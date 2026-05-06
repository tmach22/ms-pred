"""
MS³ Fine-Tuning Losses
======================
Three complementary losses for supervising FragGNN with empirical MS³ edges.

GaussianSoftTargetBCE
    Replaces hard binary targets with continuous Gaussian weights based on
    how closely each candidate fragment's mass matches the empirical child_mz.
    Handles hydrogen mobility (±2 Da) gracefully.

MaskedPUMarginLoss
    Positive-Unlabeled margin loss.  Atoms with high soft-target scores are
    treated as "positives"; atoms with near-zero soft-target scores are
    "negatives" (ghost fragments, structural isomers).  Atoms in the
    ambiguous middle range receive zero gradient, preventing the model from
    being pushed by uncertain labels.

KLDistillationLoss
    Temperature-scaled KL divergence from the frozen oracle FragGNN to the
    active fine-tuned model.  Penalises catastrophic forgetting of the
    original fragmentation prior learned from MS2 spectra.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSoftTargetBCE(nn.Module):
    """Binary cross-entropy with Gaussian soft targets in [0, 1].

    Loss_i = -(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

    where y_i is the soft target (0 = definitely absent, 1 = definitely
    present as a leaving atom) and p_i is the model's sigmoid output.

    Padding positions (beyond natoms) are masked out.
    """

    def forward(
        self,
        pred: torch.Tensor,      # [batch, max_atoms] — sigmoid probabilities
        soft_targ: torch.Tensor, # [batch, max_atoms] — Gaussian weights in [0, 1]
        natoms: torch.Tensor,    # [batch] — number of valid atoms per fragment
    ) -> torch.Tensor:
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)
        loss = -(
            soft_targ * torch.log(pred)
            + (1.0 - soft_targ) * torch.log(1.0 - pred)
        )
        is_valid = (
            torch.arange(loss.shape[1], device=loss.device)[None, :]
            < natoms[:, None]
        )
        total_atoms = natoms.sum().clamp(min=1)
        return torch.sum(loss * is_valid) / total_atoms


class MaskedPUMarginLoss(nn.Module):
    """Positive-Unlabeled hinge margin loss for atom-leaving probabilities.

    Atoms are partitioned into three regions based on their soft target y_i:
      - Positive  (y_i >= pos_threshold): empirical leaving atoms
      - Negative  (y_i <  neg_threshold): structural isomers / ghost fragments
      - Unlabeled (between thresholds):   ambiguous; gradient is zeroed

    The per-example loss is:
        max(0, margin - mean_score(positives) + mean_score(negatives))

    Examples with no positive atoms (no mass match found) are excluded.

    Args:
        margin:        Minimum gap enforced between positive and negative scores.
        pos_threshold: Soft-target threshold for a "positive" atom.
        neg_threshold: Soft-target threshold below which an atom is "negative".
    """

    def __init__(
        self,
        margin: float = 0.3,
        pos_threshold: float = 0.5,
        neg_threshold: float = 0.05,
    ):
        super().__init__()
        self.margin = margin
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def forward(
        self,
        pred: torch.Tensor,      # [batch, max_atoms] — sigmoid probabilities
        soft_targ: torch.Tensor, # [batch, max_atoms] — Gaussian weights in [0, 1]
        natoms: torch.Tensor,    # [batch] — number of valid atoms per fragment
    ) -> torch.Tensor:
        is_valid = (
            torch.arange(pred.shape[1], device=pred.device)[None, :]
            < natoms[:, None]
        ).float()

        is_pos = (soft_targ >= self.pos_threshold).float() * is_valid
        is_neg = (soft_targ <  self.neg_threshold).float() * is_valid

        # Per-example mean scores for positive and negative atoms
        n_pos = is_pos.sum(dim=1).clamp(min=1)
        n_neg = is_neg.sum(dim=1).clamp(min=1)
        pos_score = (pred * is_pos).sum(dim=1) / n_pos
        neg_score = (pred * is_neg).sum(dim=1) / n_neg

        margin_loss = F.relu(self.margin - pos_score + neg_score)

        # Only supervise examples that have at least one positive atom
        has_pos = ((soft_targ >= self.pos_threshold) * is_valid).any(dim=1).float()
        n_supervised = has_pos.sum().clamp(min=1)
        return (margin_loss * has_pos).sum() / n_supervised


class KLDistillationLoss(nn.Module):
    """Temperature-scaled KL divergence from frozen oracle to active model.

    KL(p_oracle_soft || p_active) is computed atom-wise as a binary KL
    (each atom position is an independent Bernoulli variable).

    Temperature softening applied to the oracle only:
        p_soft = p_oracle^(1/T) / (p_oracle^(1/T) + (1 - p_oracle)^(1/T))

    Higher T → softer oracle distribution → gentler regularisation.

    Args:
        temperature: Softening temperature T (>= 1.0).  T=1 recovers the
            unmodified oracle distribution; T=2 is a mild smoothing.
    """

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        if temperature < 1.0:
            raise ValueError(f"temperature must be >= 1.0, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        active_pred: torch.Tensor,  # [batch, max_atoms] — active model sigmoid outputs
        oracle_pred: torch.Tensor,  # [batch, max_atoms] — frozen oracle sigmoid outputs
        natoms: torch.Tensor,       # [batch] — number of valid atoms per fragment
    ) -> torch.Tensor:
        is_valid = (
            torch.arange(active_pred.shape[1], device=active_pred.device)[None, :]
            < natoms[:, None]
        ).float()

        eps = 1e-7
        T = self.temperature

        p_o = oracle_pred.detach().clamp(eps, 1.0 - eps)  # frozen oracle
        p_a = active_pred.clamp(eps, 1.0 - eps)           # active model

        # Temperature-softened oracle (Bernoulli parameterisation)
        po_T = p_o ** (1.0 / T)
        qo_T = (1.0 - p_o) ** (1.0 / T)
        denom = (po_T + qo_T).clamp(min=eps)
        p_soft = po_T / denom
        q_soft = qo_T / denom

        # Binary KL: sum over both Bernoulli outcomes
        kl = p_soft * (torch.log(p_soft) - torch.log(p_a)) + \
             q_soft * (torch.log(q_soft) - torch.log(1.0 - p_a))

        total_atoms = natoms.sum().clamp(min=1)
        return torch.sum(kl * is_valid) / total_atoms
