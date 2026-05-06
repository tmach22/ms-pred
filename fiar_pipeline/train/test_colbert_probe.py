"""
Phase 3 — Symmetric ColBERT Probe Evaluation
=============================================
Evaluates a trained Phase3ColBERTModel checkpoint against a held-out test set
and writes a predictions CSV.

Key differences from the deprecated test_multitask_binary.py
-------------------------------------------------------------
- Uses SiameseFragmentDataset (ICEBERG cache aware) instead of PairedSiameseDataset
- Instantiates Phase3ColBERTModel instead of Phase3_LinearProbe_SiameseNetwork
- Forward pass returns a single logits tensor [batch, 1]; no tau, no cont_pred
- Fatal guard: exits if colbert_head.binary_head.weight is absent from checkpoint
- Output CSV drops continuous_pred; keeps true_label, prob_similarity, predicted_label

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/train/test_colbert_probe.py \\
    --config              fiar_pipeline/configs/fiar_nist20.yml \\
    --phase2_checkpoint   /path/to/phase2_best.pt \\
    --model_ckpt          fiar_pipeline/results/phase3_probe/fiar_phase3_colbert_best.pt \\
    --test_pairs          /path/to/test_pairs.feather \\
    --fragment_cache_path fiar_pipeline/data/fragment_cache.pt \\
    --output_dir          fiar_pipeline/results/phase3_probe/eval
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*nested tensors.*")

# ── Repo paths ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SSP  = Path("/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor")
_MF   = Path("/data/nas-gpu/wang/tmach007/massformer/src")

sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

# SpectralSimilarityPredictor model directories
sys.path.insert(0, str(_SSP / "model" / "fiar"))
sys.path.insert(0, str(_SSP / "model" / "transport_model" / "bifurcate"))

# MassFormer package
sys.path.insert(0, str(_MF))

from fiar_pipeline.data_loaders.siamese_fragment_dataloader import (  # noqa: E402
    SiameseFragmentDataset,
    siamese_frag_collate_fn,
)
from fiar_pipeline.model.symmetric_colbert_probe import Phase3ColBERTModel  # noqa: E402


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Evaluation function ───────────────────────────────────────────────────────

def test_colbert_probe(args: argparse.Namespace) -> None:
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[*] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    full_config = load_config(args.config)
    data_cfg    = full_config["data"]
    iceberg_cfg = full_config["iceberg"]
    phase3_cfg  = full_config.get("phase3", {})

    max_k = phase3_cfg.get("max_fragments", 50)
    fragment_cache = (
        args.fragment_cache_path or iceberg_cfg.get("fragment_cache_path")
    )

    # ── Dataset & loader ──────────────────────────────────────────────────────
    print(f"\n[*] Initialising test dataset from: {args.test_pairs}")
    test_dataset = SiameseFragmentDataset(
        feather_path        = args.test_pairs,
        graphs_path         = args.graphs_path or data_cfg["graphs"],
        spec_df_path        = data_cfg["spec_df"],
        mol_df_path         = data_cfg["mol_df"],
        phase2_loader_dir   = data_cfg["phase2_loader_dir"],
        fragment_cache_path = fragment_cache,
        max_k               = max_k,
        morgan_nbits        = iceberg_cfg.get("morgan_nbits", 2048),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        collate_fn  = siamese_frag_collate_fn,
        num_workers = 6,
        pin_memory  = True,
    )

    # ── Model instantiation ───────────────────────────────────────────────────
    print("\n[*] Initialising Phase3ColBERTModel (frozen backbone + ColBERT head)...")
    model = Phase3ColBERTModel(
        cfg           = full_config,
        phase2_ckpt   = args.phase2_checkpoint,
        device        = device,
        max_fragments = max_k,
    ).to(device)

    # ── Load Phase 3 weights (strict=False) + namespace check ─────────────────
    print(f"[*] Loading Phase 3 ColBERT weights from: {args.model_ckpt}")

    # Pre-flight: catch corrupted / missing / text files before torch.load
    # produces a cryptic UnpicklingError.
    _ckpt_path = Path(args.model_ckpt)
    if not _ckpt_path.exists():
        print(f"\n[!] FATAL: Checkpoint not found: {_ckpt_path}")
        print("    -> Verify the path or check that training completed successfully.")
        sys.exit(1)
    if _ckpt_path.stat().st_size == 0:
        print(f"\n[!] FATAL: Checkpoint file is empty (0 bytes): {_ckpt_path}")
        print("    -> Training was likely interrupted before the first save.")
        sys.exit(1)
    with open(_ckpt_path, "rb") as _fh:
        _magic = _fh.read(2)
    # PyTorch checkpoints are either legacy pickle (\x80 first byte) or
    # ZIP archives (PK header, used by torch.save since PyTorch 1.6).
    # Compare the right slice width for each magic sequence.
    if _magic[:1] != b"\x80" and _magic[:2] != b"PK":
        print(f"\n[!] FATAL: '{_ckpt_path}' does not look like a PyTorch checkpoint.")
        print(f"    First bytes: {_magic!r}")
        print("    Likely causes:")
        print("    1. Training was interrupted before the first epoch completed")
        print("       (no checkpoint was ever saved at this path).")
        print("    2. --model_ckpt points to a config or text file instead of a .pt binary.")
        print(f"    -> List available checkpoints: ls -lh {_ckpt_path.parent}/")
        sys.exit(1)

    p3_state = torch.load(args.model_ckpt, map_location=device, weights_only=False)
    if isinstance(p3_state, dict) and "state_dict" in p3_state:
        p3_state = p3_state["state_dict"]

    load_result = model.load_state_dict(p3_state, strict=False)

    print("\n--- Weight Loading Diagnostics ---")
    if load_result.missing_keys:
        print(f"Missing keys ({len(load_result.missing_keys)}):")
        for k in load_result.missing_keys:
            print(f"    {k}")
    else:
        print("Missing keys  : none ✓")

    if load_result.unexpected_keys:
        print(f"Unexpected keys ({len(load_result.unexpected_keys)}):")
        for k in load_result.unexpected_keys:
            print(f"    {k}")
    else:
        print("Unexpected keys: none ✓")
    print("----------------------------------\n")

    # Fatal guard — wrong checkpoint passed
    if "colbert_head.binary_head.weight" in load_result.missing_keys:
        print("[!] FATAL ERROR: 'colbert_head.binary_head.weight' is missing.")
        print("    -> You likely passed the Phase 2.5 checkpoint instead of "
              "the trained Phase 3 ColBERT checkpoint.")
        sys.exit(1)

    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    all_binary_probs: list = []

    print("[*] Commencing evaluation...")
    test_bar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch_A, batch_B, _targets_sim, _targets_label in test_bar:
            if batch_A is None or batch_B is None:
                continue

            batch_A = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_A.items()
            }
            batch_B = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_B.items()
            }

            logits      = model(batch_A, batch_B).view(-1)   # [batch]
            binary_prob = torch.sigmoid(logits)

            all_binary_probs.extend(binary_prob.cpu().numpy().flatten())

    prob_similarity = np.array(all_binary_probs)
    predicted_label = (prob_similarity >= 0.5).astype(float)

    # ── Ground-truth labels ───────────────────────────────────────────────────
    print("[*] Loading ground-truth labels from pairs file...")
    pairs_df = (
        pd.read_feather(args.test_pairs)
        if str(args.test_pairs).endswith(".feather")
        else pd.read_csv(args.test_pairs)
    )

    if "label" not in pairs_df.columns:
        raise KeyError("Could not find 'label' column in the test pairs file.")
    true_label = pairs_df["label"].values.astype(float)

    # Truncate to inference length in case some batches were skipped
    n = min(len(true_label), len(prob_similarity))
    true_label      = true_label[:n]
    prob_similarity = prob_similarity[:n]
    predicted_label = predicted_label[:n]

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(true_label, predicted_label)
    prec = precision_score(true_label, predicted_label, zero_division=0)
    rec  = recall_score(true_label, predicted_label, zero_division=0)
    f1   = f1_score(true_label, predicted_label, zero_division=0)
    try:
        roc_auc = roc_auc_score(true_label, prob_similarity)
    except ValueError:
        roc_auc = float("nan")

    print("\n" + "=" * 50)
    print("  SYMMETRIC COLBERT PROBE METRICS")
    print("=" * 50)
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1-Score        : {f1:.4f}")
    print(f"  ROC-AUC Score   : {roc_auc:.4f}")
    print("=" * 50)

    # ── CSV output ────────────────────────────────────────────────────────────
    final_df = pd.DataFrame({
        "name_main":       pairs_df["name_main"].iloc[:n].values,
        "name_sub":        pairs_df["name_sub"].iloc[:n].values,
        "cosine_similarity": pairs_df.get("cosine_similarity", pd.Series([float("nan")] * n)).iloc[:n].values,
        "true_label":      true_label,
        "prob_similarity": prob_similarity,
        "predicted_label": predicted_label,
    })

    suffix = "_nist" if "nist" in str(args.test_pairs).lower() else ""
    results_path = os.path.join(
        args.output_dir, f"test_colbert_probe_predictions{suffix}.csv"
    )
    final_df.to_csv(results_path, index=False)
    print(f"\n[+] Predictions saved to: {results_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate the Phase 3 Symmetric ColBERT Probe."
    )
    p.add_argument("--config", default="fiar_pipeline/configs/fiar_nist20.yml",
                   help="Path to YAML config (default: fiar_nist20.yml)")
    p.add_argument("--test_pairs",        type=str, required=True,
                   help=".feather or .csv file with name_main, name_sub, label columns")
    p.add_argument("--graphs_path",       type=str, default=None,
                   help="Override data.graphs in config")
    p.add_argument("--phase2_checkpoint", type=str, required=True,
                   help="Path to the Phase 2.5 backbone checkpoint (.pt)")
    p.add_argument("--model_ckpt",        type=str, required=True,
                   help="Path to the trained Phase 3 ColBERT checkpoint (.pt)")
    p.add_argument("--fragment_cache_path", type=str, default=None,
                   help="Override iceberg.fragment_cache_path in config")
    p.add_argument("--output_dir",        type=str,
                   default="fiar_pipeline/results/phase3_probe/eval",
                   help="Directory for output CSV")
    p.add_argument("--batch_size",        type=int, default=128)
    p.add_argument("--gpu_id",            type=int, default=0)
    args = p.parse_args()
    test_colbert_probe(args)
