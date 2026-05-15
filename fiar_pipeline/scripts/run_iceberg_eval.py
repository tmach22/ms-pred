"""
run_iceberg_eval.py — ICEBERG MS2 Prediction: Baseline vs. Fine-Tuned
======================================================================
Ground truth: data/MSnLib/libraries/mgf/20241003_nihnp_pos_ms2.mgf
              NIH Natural Products Library, Pluskal lab, Orbitrap ID-X
              Filtered to SPECTYPE=SINGLE_BEST_SCAN or SAME_ENERGY
              (pure observed MS2 scans — no pseudo-MS2 from MSn trees)

Both models predict the MS2 fragmentation spectrum of the parent SMILES.
Predicted prob_gen values (shifted to m/z space via +proton mass) are
compared against the actual measured MS2 m/z+intensity spectrum.

Metrics:
  • Mean Cosine Similarity   (L2-normalised dot product in m/z bin space)
  • Mean Entropy Similarity  (Jensen-Shannon divergence, bounded [0,1])
  • Fragment Recall@K        (fraction of GT peaks explained by top-K
                              predicted fragments, ±0.5 Da, K=5/10/20)

Results:
  fiar_pipeline/results/iceberg_eval/baseline/pred_eval.yaml
  fiar_pipeline/results/iceberg_eval/finetuned/pred_eval.yaml
  fiar_pipeline/results/iceberg_eval/comparison_report.md

Usage:
    cd /data/nas-gpu/wang/tmach007/ms-pred
    python fiar_pipeline/scripts/run_iceberg_eval.py \\
        --baseline_ckpt  weights/nist_iceberg_generate.ckpt \\
        --finetuned_ckpt weights/reward_scaled/reward_scaled_best.ckpt \\
        --mgf_path       data/MSnLib/libraries/mgf/20241003_nihnp_pos_ms2.mgf \\
        --n_sample       200 \\
        --max_nodes      100
"""
from __future__ import annotations

import argparse
import ast
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from numpy.linalg import norm
from scipy.stats import sem

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(_REPO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ms_pred.dag_pred.gen_model import FragGNN
import ms_pred.common as common

SEED            = 42
N_BINS          = 2000          # 1 Da resolution, 0–2000 Da
PROTON_MASS     = 1.007276      # shifts predicted neutral formula mass → [M+H]+ m/z space
MASS_TOL_DA     = 0.5           # tolerance for fragment mass matching
MIN_REL_INTEN   = 0.01          # GT peaks below 1 % of base peak are ignored
VALID_SPECTYPES = {"SINGLE_BEST_SCAN", "SAME_ENERGY"}
DEFAULT_ADDUCT  = "[M+H]+"


# ── MGF parser ────────────────────────────────────────────────────────────────

def parse_mgf(path: str, valid_spectypes=None, seed: int = SEED) -> list[dict]:
    """Parse MGF and return list of entry dicts (filtered by SPECTYPE)."""
    entries = []
    meta: dict = {}
    peaks: list = []
    in_peaks = False

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                meta, peaks, in_peaks = {}, [], False
            elif line == "END IONS":
                if peaks:
                    meta["peaks"] = peaks
                    if valid_spectypes is None or meta.get("SPECTYPE") in valid_spectypes:
                        entries.append(dict(meta))
            elif line.startswith("Num peaks="):
                meta["Num peaks"] = line.split("=", 1)[1]
                in_peaks = True
            elif in_peaks:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
            elif "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v

    return entries


def _parse_ce(val: str) -> float:
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return float(np.mean(parsed)) if parsed else 40.0
        return float(parsed)
    except Exception:
        return 40.0


# ── Spectrum helpers ──────────────────────────────────────────────────────────

def _bin_predicted(pred_entries: dict) -> np.ndarray:
    """Bin predicted prob_gen into m/z space (neutral mass + proton)."""
    vec = np.zeros(N_BINS)
    for entry in pred_entries.values():
        form = entry.get("form", "")
        if not form:
            continue
        try:
            mz_idx = int(round(common.formula_mass(form) + PROTON_MASS))
        except Exception:
            continue
        prob = float(entry.get("prob_gen", 0.0))
        if 0 <= mz_idx < N_BINS:
            vec[mz_idx] += prob
    return vec


def _bin_gt(peaks: list[tuple], min_rel: float = MIN_REL_INTEN) -> np.ndarray:
    """Bin observed MS2 peaks; filter below min_rel * base-peak intensity."""
    if not peaks:
        return np.zeros(N_BINS)
    max_inten = max(i for _, i in peaks)
    if max_inten <= 0:
        return np.zeros(N_BINS)
    vec = np.zeros(N_BINS)
    for mz, inten in peaks:
        if inten / max_inten >= min_rel:
            idx = int(round(mz))
            if 0 <= idx < N_BINS:
                vec[idx] += inten
    return vec


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = norm(a), norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def entropy_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon spectral entropy similarity, bounded [0, 1]."""
    def _norm(v):
        s = v.sum()
        return v / s if s > 1e-22 else v

    def _H(v):
        return float(-np.sum(v * np.log(v + 1e-22)))

    na, nb = _norm(a), _norm(b)
    if na.sum() < 1e-12 or nb.sum() < 1e-12:
        return 0.0
    mix = (na + nb) / 2.0
    return float(1.0 - (2 * _H(mix) - _H(na) - _H(nb)) / np.log(4))


def fragment_recall_at_k(
    pred_entries: dict,
    gt_peaks:     list[tuple],
    k:            int,
    min_rel:      float = MIN_REL_INTEN,
) -> float:
    """Fraction of significant GT peaks explained by top-K predicted fragments.

    A GT peak is 'explained' if any top-K predicted fragment's m/z (formula
    mass + proton) falls within MASS_TOL_DA of the GT peak m/z.
    """
    if not pred_entries or not gt_peaks:
        return 0.0

    max_inten = max(i for _, i in gt_peaks)
    sig_gt = [mz for mz, i in gt_peaks if i / max_inten >= min_rel] if max_inten > 0 else []
    if not sig_gt:
        return 0.0

    top_k = sorted(pred_entries.values(), key=lambda e: float(e.get("prob_gen", 0.0)), reverse=True)[:k]
    pred_mzs = []
    for entry in top_k:
        form = entry.get("form", "")
        if not form:
            continue
        try:
            pred_mzs.append(common.formula_mass(form) + PROTON_MASS)
        except Exception:
            continue

    if not pred_mzs:
        return 0.0

    hits = sum(
        1 for gt_mz in sig_gt
        if any(abs(gt_mz - pm) <= MASS_TOL_DA for pm in pred_mzs)
    )
    return hits / len(sig_gt)


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_baseline(ckpt: str, device: torch.device) -> FragGNN:
    model = FragGNN.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    return model


def load_finetuned(baseline_ckpt: str, finetuned_ckpt: str, device: torch.device) -> FragGNN:
    model = FragGNN.load_from_checkpoint(baseline_ckpt, map_location=device)
    ft = torch.load(finetuned_ckpt, map_location=device, weights_only=False)
    sd = ft.get("state_dict", ft)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [WARN] Missing keys: {missing[:3]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys: {unexpected[:3]}")
    model.eval().to(device)
    return model


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_model(
    model:   FragGNN,
    entries: list[dict],
    device:  torch.device,
    max_nodes: int,
    label:   str,
) -> dict:
    cos_sims, ent_sims = [], []
    rec5, rec10, rec20 = [], [], []
    n_skipped = 0
    t0 = time.time()

    for i, entry in enumerate(entries, 1):
        if i % 25 == 0:
            elapsed = (time.time() - t0) / 60
            print(
                f"  [{label}] {i}/{len(entries)}  elapsed={elapsed:.1f}min"
                f"  cos={np.mean(cos_sims):.4f}  ent={np.mean(ent_sims):.4f}"
                f"  rec@10={np.mean(rec10):.4f}",
                flush=True,
            )

        smi      = entry.get("SMILES", "")
        pepmass  = entry.get("PEPMASS", "")
        ce_str   = entry.get("COLLISION_ENERGY", "40.0")
        peaks    = entry.get("peaks", [])

        if not smi or not peaks:
            n_skipped += 1
            continue

        try:
            precursor_mz = float(pepmass.split()[0])  # PEPMASS may be "mz charge"
        except Exception:
            n_skipped += 1
            continue

        ce = _parse_ce(ce_str)

        try:
            pred_entries = model.predict_mol(
                root_smi      = smi,
                collision_eng = ce,
                precursor_mz  = precursor_mz,
                adduct        = DEFAULT_ADDUCT,
                threshold     = 0.0,
                device        = str(device),
                max_nodes     = max_nodes,
            )
        except Exception:
            n_skipped += 1
            continue

        if not pred_entries:
            n_skipped += 1
            continue

        pred_vec = _bin_predicted(pred_entries)
        gt_vec   = _bin_gt(peaks)

        cos_sims.append(cosine_sim(pred_vec, gt_vec))
        ent_sims.append(entropy_sim(pred_vec, gt_vec))
        rec5.append(fragment_recall_at_k(pred_entries, peaks, k=5))
        rec10.append(fragment_recall_at_k(pred_entries, peaks, k=10))
        rec20.append(fragment_recall_at_k(pred_entries, peaks, k=20))

    elapsed_total = (time.time() - t0) / 60
    n = len(cos_sims)

    return {
        "label":           label,
        "n_evaluated":     n,
        "n_skipped":       n_skipped,
        "elapsed_min":     round(elapsed_total, 2),
        "avg_cosine_sim":  float(np.mean(cos_sims))  if cos_sims else None,
        "sem_cosine_sim":  float(sem(cos_sims))      if cos_sims else None,
        "avg_entropy_sim": float(np.mean(ent_sims))  if ent_sims else None,
        "sem_entropy_sim": float(sem(ent_sims))      if ent_sims else None,
        "recall_at_5":     float(np.mean(rec5))      if rec5 else None,
        "recall_at_10":    float(np.mean(rec10))     if rec10 else None,
        "recall_at_20":    float(np.mean(rec20))     if rec20 else None,
        "sem_recall_at_5": float(sem(rec5))          if rec5 else None,
        "sem_recall_at_10":float(sem(rec10))         if rec10 else None,
        "sem_recall_at_20":float(sem(rec20))         if rec20 else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICEBERG MS2 Eval: Baseline vs. Fine-Tuned")
    parser.add_argument("--baseline_ckpt",  required=True)
    parser.add_argument("--finetuned_ckpt", required=True)
    parser.add_argument("--mgf_path",       required=True,
                        help="Path to pos_ms2 MGF (e.g. 20241003_nihnp_pos_ms2.mgf)")
    parser.add_argument("--n_sample",       default=200, type=int,
                        help="Number of spectra to sample (default: 200)")
    parser.add_argument("--max_nodes",      default=100, type=int)
    parser.add_argument("--out_dir",        default="fiar_pipeline/results/iceberg_eval")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    out_dir = Path(args.out_dir)
    (out_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (out_dir / "finetuned").mkdir(parents=True, exist_ok=True)

    log_path = Path("fiar_pipeline/logs/iceberg_eval.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device      : {device}")
    log(f"Baseline    : {args.baseline_ckpt}")
    log(f"Fine-tuned  : {args.finetuned_ckpt}")
    log(f"MGF source  : {args.mgf_path}")

    # ── Parse & sample ────────────────────────────────────────────────────────
    log(f"Parsing MGF (SPECTYPE filter: {sorted(VALID_SPECTYPES)})...")
    all_entries = parse_mgf(args.mgf_path, valid_spectypes=VALID_SPECTYPES)
    log(f"  Clean MS2 entries available: {len(all_entries)}")

    random.shuffle(all_entries)
    entries = all_entries[:args.n_sample]
    log(f"  Sampled for evaluation     : {len(entries)}")

    # ── Load models ───────────────────────────────────────────────────────────
    log("Loading baseline model...")
    baseline_model = load_baseline(args.baseline_ckpt, device)
    log("Loading fine-tuned model...")
    finetuned_model = load_finetuned(args.baseline_ckpt, args.finetuned_ckpt, device)

    ft_meta  = torch.load(args.finetuned_ckpt, map_location="cpu", weights_only=False)
    ft_epoch = ft_meta.get("epoch", "?")
    ft_loss  = ft_meta.get("loss", float("nan"))
    log(f"Fine-tuned checkpoint: epoch={ft_epoch}, train_loss={ft_loss:.6f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log("=" * 60)
    log("EVALUATING: BASELINE")
    b_res = evaluate_model(baseline_model, entries, device, args.max_nodes, "baseline")
    log(f"Baseline   cos={b_res['avg_cosine_sim']:.4f}  ent={b_res['avg_entropy_sim']:.4f}"
        f"  rec@5={b_res['recall_at_5']:.4f}  rec@10={b_res['recall_at_10']:.4f}"
        f"  rec@20={b_res['recall_at_20']:.4f}")
    with open(out_dir / "baseline" / "pred_eval.yaml", "w") as fh:
        yaml.dump(b_res, fh, default_flow_style=False)

    log("=" * 60)
    log("EVALUATING: FINE-TUNED")
    ft_res = evaluate_model(finetuned_model, entries, device, args.max_nodes, "finetuned")
    log(f"Fine-tuned cos={ft_res['avg_cosine_sim']:.4f}  ent={ft_res['avg_entropy_sim']:.4f}"
        f"  rec@5={ft_res['recall_at_5']:.4f}  rec@10={ft_res['recall_at_10']:.4f}"
        f"  rec@20={ft_res['recall_at_20']:.4f}")
    with open(out_dir / "finetuned" / "pred_eval.yaml", "w") as fh:
        yaml.dump(ft_res, fh, default_flow_style=False)

    # ── Comparison report ─────────────────────────────────────────────────────
    def _fmt(v, d=4):
        return f"{v:.{d}f}" if v is not None else "N/A"

    def _delta(a, b):
        if a is None or b is None:
            return "N/A"
        d = b - a
        return f"{'+'if d>=0 else ''}{d:.4f} {'↑' if d>0 else '↓'}"

    b, ft = b_res, ft_res
    mgf_name = Path(args.mgf_path).name

    report = f"""# ICEBERG MS2 Evaluation — Baseline vs. Fine-Tuned

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Device**: {device}
**Ground truth**: `{mgf_name}`
  NIH Natural Products Library (Pluskal lab, Orbitrap ID-X, HCD)
  SPECTYPE filter: SINGLE_BEST_SCAN + SAME_ENERGY (pure observed MS2)
  Entries evaluated: {len(entries)} / {len(all_entries)} available
**Baseline checkpoint**: `{args.baseline_ckpt}`
**Fine-tuned checkpoint**: `{args.finetuned_ckpt}`
  Epoch {ft_epoch}, Training Loss {ft_loss:.6f}

## Evaluation Methodology

Both models run `FragGNN.predict_mol()` on the parent SMILES to produce a
fragmentation tree. Each predicted fragment's neutral formula mass is shifted
to m/z space (+{PROTON_MASS} Da proton) and binned at 1 Da resolution into a
probability vector (prob\\_gen). The observed MS2 spectrum is intensity-binned
at the same resolution (peaks below {int(MIN_REL_INTEN*100)}% of base peak discarded).

**Cosine / Entropy Similarity**: vectorised spectrum comparison in shared m/z bin space.
**Fragment Recall@K**: fraction of significant GT peaks (≥{int(MIN_REL_INTEN*100)}% base peak)
  whose m/z is within ±{MASS_TOL_DA} Da of any top-K predicted fragment (ranked by prob\\_gen).

---

## Results

| Metric | Baseline | Fine-Tuned | Δ (FT − Base) |
|---|---|---|---|
| **Entropy Similarity** | {_fmt(b['avg_entropy_sim'])} ± {_fmt(b['sem_entropy_sim'])} | {_fmt(ft['avg_entropy_sim'])} ± {_fmt(ft['sem_entropy_sim'])} | {_delta(b['avg_entropy_sim'], ft['avg_entropy_sim'])} |
| **Cosine Similarity** | {_fmt(b['avg_cosine_sim'])} ± {_fmt(b['sem_cosine_sim'])} | {_fmt(ft['avg_cosine_sim'])} ± {_fmt(ft['sem_cosine_sim'])} | {_delta(b['avg_cosine_sim'], ft['avg_cosine_sim'])} |
| **Fragment Recall@5** | {_fmt(b['recall_at_5'])} ± {_fmt(b['sem_recall_at_5'])} | {_fmt(ft['recall_at_5'])} ± {_fmt(ft['sem_recall_at_5'])} | {_delta(b['recall_at_5'], ft['recall_at_5'])} |
| **Fragment Recall@10** | {_fmt(b['recall_at_10'])} ± {_fmt(b['sem_recall_at_10'])} | {_fmt(ft['recall_at_10'])} ± {_fmt(ft['sem_recall_at_10'])} | {_delta(b['recall_at_10'], ft['recall_at_10'])} |
| **Fragment Recall@20** | {_fmt(b['recall_at_20'])} ± {_fmt(b['sem_recall_at_20'])} | {_fmt(ft['recall_at_20'])} ± {_fmt(ft['sem_recall_at_20'])} | {_delta(b['recall_at_20'], ft['recall_at_20'])} |

> ± values are SEM. Recall@K = fraction of significant GT peaks explained
> by top-K predicted fragments at ±{MASS_TOL_DA} Da tolerance.

## Run Statistics

| | Baseline | Fine-Tuned |
|---|---|---|
| Spectra evaluated | {b['n_evaluated']} | {ft['n_evaluated']} |
| Spectra skipped | {b['n_skipped']} | {ft['n_skipped']} |
| Runtime (min) | {b['elapsed_min']} | {ft['elapsed_min']} |
"""

    report_path = out_dir / "comparison_report.md"
    with open(report_path, "w") as fh:
        fh.write(report)
    log(f"Comparison report → {report_path}")

    log("=" * 60)
    log("EVALUATION COMPLETE")
    print(f"\n{'='*64}")
    print("  ICEBERG MS2 EVALUATION — RESULTS")
    print(f"  Ground truth: {mgf_name}")
    print(f"{'='*64}")
    print(f"  {'Metric':<30} {'Baseline':>10} {'Fine-Tuned':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Entropy Similarity':<30} {_fmt(b['avg_entropy_sim']):>10} {_fmt(ft['avg_entropy_sim']):>10}")
    print(f"  {'Cosine Similarity':<30} {_fmt(b['avg_cosine_sim']):>10} {_fmt(ft['avg_cosine_sim']):>10}")
    print(f"  {'Fragment Recall@5':<30} {_fmt(b['recall_at_5']):>10} {_fmt(ft['recall_at_5']):>10}")
    print(f"  {'Fragment Recall@10':<30} {_fmt(b['recall_at_10']):>10} {_fmt(ft['recall_at_10']):>10}")
    print(f"  {'Fragment Recall@20':<30} {_fmt(b['recall_at_20']):>10} {_fmt(ft['recall_at_20']):>10}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
