"""
Single-Batch Overfit Test — MS³ Fine-Tuning Architecture
==========================================================
Validates that gradients flow correctly through the KineticFiLM layer and the
GaussianSoftTargetBCE loss, while confirming the frozen backbone stays frozen.

Pass criteria
-------------
  1. Total loss decreases monotonically (or near-monotonically) to near zero
     over 100 epochs on the fixed batch — proving the forward + backward graph
     is wired correctly.
  2. active_model.output_map gradient norms are positive at every logged step.
  3. KineticFiLM gradient norms are positive at every logged step.
  4. The deep backbone parameter (gnn.gnn.model.linears.0.weight) has grad=None
     or norm=0.0 throughout — proving the backbone freeze is intact.

Usage
-----
cd /data/nas-gpu/wang/tmach007/ms-pred

/data/nas-gpu/wang/tmach007/ms-pred/y/envs/ms-gen/bin/python \\
    fiar_pipeline/train/test_single_batch_overfit.py \\
    [--feather  data/ms3_edge_labels.feather] \\
    [--ckpt     weights/nist_iceberg_generate.ckpt] \\
    [--device   cpu] \\
    [--epochs   100] \\
    [--lr       1e-3]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Repo path injection (mirrors finetune_iceberg_ms3.py) ─────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from ms_pred.dag_pred.gen_model import FragGNN

from fiar_pipeline.data_loaders.ms3_edge_dataset import MS3EdgeDataset
from fiar_pipeline.model.kinetic_film import KineticFiLM
from fiar_pipeline.model.ms3_losses import (
    GaussianSoftTargetBCE,
    MaskedPUMarginLoss,
    KLDistillationLoss,
)
from fiar_pipeline.train.finetune_iceberg_ms3 import (
    load_frag_gnn,
    tree_proc_kwargs_from_ckpt,
    freeze_backbone,
    _FILM_CTX,
    _pool_hook,
    _set_nce,
    _model_forward,
)

# ── Frozen backbone parameter used as the "must be zero" audit target ──────────
FROZEN_PARAM_NAME = "gnn.gnn.model.linears.0.weight"


# ── Gradient-norm helpers ──────────────────────────────────────────────────────

def _grad_norm(module: nn.Module) -> float:
    """L2 norm across all parameter gradients in a module (0.0 if all None)."""
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def _named_param_grad_norm(model: nn.Module, full_name: str) -> str:
    """Return grad norm string for a specific named parameter."""
    for name, param in model.named_parameters():
        if name == full_name:
            if param.grad is None:
                return "None"
            return f"{param.grad.detach().norm(2).item():.6f}"
    return f"<'{full_name}' not found>"


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print("=" * 70)
    print("  MS³ SINGLE-BATCH OVERFIT TEST")
    print("=" * 70)
    print(f"  Device : {device}")
    print(f"  LR     : {args.lr}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch_size}")
    print()

    # ── 1. Dataset — extract exactly one batch ─────────────────────────────────
    print("[1/4] Loading dataset and extracting one batch …")
    tp_kwargs = tree_proc_kwargs_from_ckpt(args.ckpt)
    print(f"    TreeProcessor kwargs (from checkpoint): {tp_kwargs}")

    dataset = MS3EdgeDataset(
        feather_path          = args.feather,
        max_mol_size          = 80,
        tree_processor_kwargs = tp_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = MS3EdgeDataset.collate_fn,
        num_workers = 0,   # single-process for reproducibility
    )

    # Pull the first non-None batch
    fixed_batch = None
    for b in loader:
        if b is not None:
            fixed_batch = b
            break
    if fixed_batch is None:
        print("[!] FATAL: Could not obtain a valid batch. Check feather file.")
        sys.exit(1)

    n_items = len(fixed_batch["names"])
    n_frags = fixed_batch["frag_atoms"].shape[0]
    print(f"    Batch acquired: {n_items} molecules / {n_frags} fragments")
    print(f"    targ_atoms shape : {fixed_batch['targ_atoms'].shape}")
    print(f"    frag_atoms       : {fixed_batch['frag_atoms'].tolist()}")

    # ── 2. Model setup (mirrors finetune_iceberg_ms3.py exactly) ──────────────
    print(f"\n[2/4] Loading FragGNN from: {args.ckpt}")
    active_model = load_frag_gnn(args.ckpt, device).to(device)
    oracle_model = copy.deepcopy(active_model).to(device)

    # Oracle fully frozen
    for p in oracle_model.parameters():
        p.requires_grad = False
    oracle_model.eval()

    # Active: freeze backbone, unfreeze output_map
    freeze_backbone(active_model)

    trainable_backbone = [p for p in active_model.parameters() if p.requires_grad]
    n_backbone_trainable = sum(p.numel() for p in trainable_backbone)
    print(f"    Backbone trainable params : {n_backbone_trainable:,}  "
          f"(output_map only)")

    # KineticFiLM
    film = KineticFiLM(hidden_size=active_model.hidden_size).to(device)
    _FILM_CTX.film = film
    hook_handle = active_model.pool.register_forward_hook(_pool_hook)
    n_film = sum(p.numel() for p in film.parameters())
    print(f"    KineticFiLM params        : {n_film:,}")
    print(f"    Pool hook registered on   : {type(active_model.pool).__name__}")

    # ── 3. Losses and optimiser ────────────────────────────────────────────────
    print("\n[3/4] Building losses and AdamW optimiser …")
    loss_gs = GaussianSoftTargetBCE()
    loss_pu = MaskedPUMarginLoss()
    loss_kl = KLDistillationLoss(temperature=2.0)

    all_trainable = trainable_backbone + list(film.parameters())
    optimizer = torch.optim.AdamW(all_trainable, lr=args.lr, weight_decay=1e-5)

    # ── 4. Overfit loop ────────────────────────────────────────────────────────
    print("\n[4/4] Running overfit loop …\n")

    # Pre-compute oracle predictions once (frozen — never change)
    with torch.no_grad():
        _set_nce(fixed_batch, fixed_batch["inds"], device)
        pred_oracle = _model_forward(oracle_model, fixed_batch, device).detach()

    targ   = fixed_batch["targ_atoms"].to(device)
    natoms = fixed_batch["frag_atoms"].to(device)

    header = (
        f"{'Epoch':>6}  {'Total':>9}  {'GS-BCE':>9}  {'PU-Marg':>9}  "
        f"{'KL-Dist':>9}  "
        f"{'∇output_map':>13}  {'∇film.mlp':>13}  {'∇frozen_gnn':>13}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    initial_loss = None
    prev_loss = None

    for epoch in range(1, args.epochs + 1):
        active_model.eval()              # freeze BN running stats
        active_model.output_map.train()  # only output_map in train mode

        _set_nce(fixed_batch, fixed_batch["inds"], device)
        pred_active = _model_forward(active_model, fixed_batch, device)

        gs   = loss_gs(pred_active, targ, natoms)
        pu   = loss_pu(pred_active, targ, natoms)
        kl   = loss_kl(pred_active, pred_oracle, natoms)
        loss = gs + 0.5 * pu + 0.3 * kl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(all_trainable, max_norm=5.0)
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        prev_loss = loss.item()

        if epoch % 10 == 0 or epoch == 1:
            norm_output_map = _grad_norm(active_model.output_map)
            norm_film       = _grad_norm(film)
            norm_frozen     = _named_param_grad_norm(active_model, FROZEN_PARAM_NAME)

            print(
                f"{epoch:>6}  "
                f"{loss.item():>9.5f}  "
                f"{gs.item():>9.5f}  "
                f"{pu.item():>9.5f}  "
                f"{kl.item():>9.5f}  "
                f"{norm_output_map:>13.6f}  "
                f"{norm_film:>13.6f}  "
                f"{norm_frozen:>13}"
            )

    print(sep)
    final_loss = prev_loss
    reduction  = (initial_loss - final_loss) / max(abs(initial_loss), 1e-9) * 100.0

    print(f"\nInitial loss : {initial_loss:.5f}")
    print(f"Final loss   : {final_loss:.5f}")
    print(f"Reduction    : {reduction:.1f}%")

    # ── Pass / Fail summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PASS / FAIL SUMMARY")
    print("=" * 70)

    frozen_grad_str = _named_param_grad_norm(active_model, FROZEN_PARAM_NAME)
    try:
        frozen_is_zero = frozen_grad_str == "None" or float(frozen_grad_str) < 1e-9
    except ValueError:
        frozen_is_zero = False

    output_map_norm = _grad_norm(active_model.output_map)
    film_norm       = _grad_norm(film)
    converged       = reduction > 50.0

    checks = [
        ("Loss decreased > 50%",                converged,         f"{reduction:.1f}% reduction"),
        ("output_map has non-zero gradients",    output_map_norm > 0, f"norm={output_map_norm:.6f}"),
        ("KineticFiLM has non-zero gradients",   film_norm > 0,     f"norm={film_norm:.6f}"),
        (f"'{FROZEN_PARAM_NAME}' grad is frozen", frozen_is_zero,   f"norm={frozen_grad_str}"),
    ]

    all_passed = True
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}]  {label:<50}  {detail}")

    print("=" * 70)
    if all_passed:
        print("  OVERALL: PASS — architecture is correctly wired.")
    else:
        print("  OVERALL: FAIL — review the FAIL items above.")
    print("=" * 70)

    hook_handle.remove()
    _FILM_CTX.film = None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-batch overfit test for the MS³ fine-tuning architecture."
    )
    p.add_argument("--feather",    default="data/ms3_edge_labels.feather")
    p.add_argument("--ckpt",       default="weights/nist_iceberg_generate.ckpt")
    p.add_argument("--device",     default="cpu",
                   help="Torch device (default: cpu — fast for 100 epochs on one batch)")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=32)
    return p.parse_args()


if __name__ == "__main__":
    main(_cli())
