"""
Phase 4.1 — Unbalanced Gated Sinkhorn Evaluation Script
=======================================================
Evaluates the trained SinkhornFiLMProbe on a holdout test set and exports
the predictions to a CSV file.

Output Format:
--------------
name_main, name_sub, cosine_similarity, true_label, prob_similarity, predicted_label
"""

from __future__ import annotations

# ── 1:1 MIRRORED TRAINING IMPORTS TO PROTECT THE NAMESPACE ──
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Repo paths ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SSP  = Path("/data/nas-gpu/wang/tmach007/SpectralSimilarityPredictor")
_MF   = Path("/data/nas-gpu/wang/tmach007/massformer/src")

sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SSP / "model" / "fiar"))
sys.path.insert(0, str(_SSP / "model" / "transport_model" / "bifurcate"))
sys.path.insert(0, str(_MF))

# ── NAMESPACE INJECTION (Fixes NameError: 'Predictor' is not defined) ─────────
from fiar_pipeline.data_loaders.siamese_fragment_dataloader import SiameseFragmentDataset
from fiar_pipeline.model.symmetric_colbert_probe import Phase3ColBERTModel

# ── Phase 4.1 Sinkhorn Imports ────────────────────────────────────────────────
from fiar_pipeline.data_loaders.phase4_dataloader import (  # noqa: E402
    SinkhornFragmentDataset,
    sinkhorn_collate_fn,
)
from fiar_pipeline.model.phase4_colbert import SinkhornFiLMProbe   # noqa: E402


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test(args: argparse.Namespace) -> None:
    cfg         = load_config(args.config)
    data_cfg    = cfg["data"]
    train_cfg   = cfg["train"]
    iceberg_cfg = cfg["iceberg"]
    phase3_cfg  = cfg.get("phase3", {})
    phase4_cfg  = cfg.get("phase4", {})

    device = torch.device(
        f"cuda:{train_cfg['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Test] Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    fragment_cache = args.fragment_cache_path or iceberg_cfg.get("fragment_cache_path")
    max_k = phase4_cfg.get("max_fragments", phase3_cfg.get("max_fragments", 50))

    test_pairs_path = args.test_pairs or data_cfg.get("pairs_test")
    if not test_pairs_path or not Path(test_pairs_path).exists():
        raise FileNotFoundError(f"Test pairs file not found: {test_pairs_path}")

    print(f"[Test] Loading dataset from: {test_pairs_path}")

    common_kwargs = dict(
        spec_df_path        = data_cfg["spec_df"],
        mol_df_path         = data_cfg["mol_df"],
        phase2_loader_dir   = data_cfg["phase2_loader_dir"],
        fragment_cache_path = fragment_cache,
        max_k               = max_k,
        morgan_nbits        = iceberg_cfg.get("morgan_nbits", 2048),
    )

    test_ds = SinkhornFragmentDataset(
        feather_path = test_pairs_path,
        graphs_path  = data_cfg["graphs"],
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,  # Strict False for CSV alignment
        collate_fn=sinkhorn_collate_fn,
        num_workers=6,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    phase2_ckpt = args.phase2_checkpoint or train_cfg["phase2_checkpoint"]

    # Extract Unbalanced Architecture Parameters
    s_eps   = phase4_cfg.get("sinkhorn_eps", 0.01)
    s_iters = phase4_cfg.get("sinkhorn_iters", 20)
    n_heads = phase4_cfg.get("num_heads", 8)
    rho_val = phase4_cfg.get("rho_init", 1.0)
    tau_val = phase4_cfg.get("tau_init", 5.0)

    model = SinkhornFiLMProbe(
        cfg=cfg,
        phase2_ckpt=phase2_ckpt,
        device=device,
        max_fragments=max_k,
        sinkhorn_eps=s_eps,
        sinkhorn_iters=s_iters,
        num_heads=n_heads,
        load_backbone=False, # <-- MUST BE FALSE FOR TESTING
        rho_init=rho_val,
        tau_init=tau_val
    ).to(device)

    if not Path(args.test_checkpoint).exists():
        raise FileNotFoundError(f"Trained Phase 4 checkpoint not found: {args.test_checkpoint}")

    print(f"[Test] Loading trained weights from: {args.test_checkpoint}")
    state_dict = torch.load(args.test_checkpoint, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # strict=False is required because the backbone is not loaded
    model.load_state_dict(state_dict, strict=False)

    # Print loaded tension parameters
    print(f"[Test] Loaded Tau: {model.tau.item():.3f}")
    import torch.nn.functional as F
    print(f"[Test] Loaded Rho: {F.softplus(model.rho_raw).item():.3f}")

    # ── Evaluation Loop ───────────────────────────────────────────────────────
    model.eval()

    all_probs = []
    all_labels = []
    all_sims = []

    print("[Test] Running inference...")
    with torch.no_grad():
        for batch_A, batch_B, t_sim, t_lbl in tqdm(test_loader, desc="[Test]"):
            if batch_A is None:
                continue

            def _to_device(d: dict) -> dict:
                return {
                    k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in d.items()
                }

            batch_A = _to_device(batch_A)
            batch_B = _to_device(batch_B)

            # Forward pass
            logits, _ = model(batch_A, batch_B)
            probs = torch.sigmoid(logits).view(-1)

            # Store predictions and ground truths
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(t_lbl.view(-1).numpy())
            all_sims.extend(t_sim.view(-1).numpy())

    # ── Calculate Metrics ─────────────────────────────────────────────────────
    all_probs_arr = np.array(all_probs)
    all_labels_arr = np.array(all_labels)

    # Map probabilities to rigid 1.0 or 0.0 labels for hard metrics
    predicted_labels = (all_probs_arr >= 0.5).astype(float)

    try:
        auc  = roc_auc_score(all_labels_arr, all_probs_arr)
        acc  = accuracy_score(all_labels_arr, predicted_labels)
        prec = precision_score(all_labels_arr, predicted_labels, zero_division=0)
        rec  = recall_score(all_labels_arr, predicted_labels, zero_division=0)
        f1   = f1_score(all_labels_arr, predicted_labels, zero_division=0)
    except ValueError:
        auc = acc = prec = rec = f1 = 0.0

    print(f"\n{'='*50}")
    print(f"  PHASE 4.1 UOT-GATED SINKHORN PROBE METRICS")
    print(f"{'='*50}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  F1-Score        : {f1:.4f}")
    print(f"  ROC-AUC Score   : {auc:.4f}")
    print(f"{'='*50}\n")

    # ── Export Results to CSV ─────────────────────────────────────────────────
    print(f"[Test] Compiling strictly formatted CSV export...")

    # Import pandas locally to guarantee it doesn't disrupt early sys.modules
    import pandas as pd

    # Isolate original dataframe and reset index to ensure clean alignment
    pairs_df = test_ds.pairs_df.reset_index(drop=True)

    # In the rare event the dataloader dropped a batch due to an error,
    # safely truncate the text dataframe to match the returned arrays.
    valid_len = len(all_probs)

    out_df = pd.DataFrame({
        "name_main": pairs_df["name_main"].iloc[:valid_len],
        "name_sub": pairs_df["name_sub"].iloc[:valid_len],
        "cosine_similarity": all_sims,
        "true_label": all_labels,
        "prob_similarity": all_probs_arr,
        "predicted_label": predicted_labels
    })

    out_df.to_csv(args.out_csv, index=False)
    print(f"[Test] Saved formatted results to: {args.out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4.1: Unbalanced Gated Sinkhorn Testing Script")
    p.add_argument("--config", required=True,
                   help="Path to YAML config (e.g., fiar_pipeline/configs/fiar_nist20.yml)")
    p.add_argument("--test_checkpoint", required=True,
                   help="Path to the trained Phase 4 .pt file to evaluate.")
    p.add_argument("--test_pairs", default=None,
                   help="Path to the test set feather/csv file.")
    p.add_argument("--out_csv", default="fiar_pipeline/results/phase4_probe/phase4_test_results.csv",
                   help="Path to save the resulting CSV file.")
    p.add_argument("--phase2_checkpoint", default=None,
                   help="Override train.phase2_checkpoint in config.")
    p.add_argument("--fragment_cache_path", default=None,
                   help="Override iceberg.fragment_cache_path in config.")
    return p.parse_args()


if __name__ == "__main__":
    test(_cli())
