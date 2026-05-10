"""
Phase 4 Losses
==============
EntropyRegularizedFocalLoss
    Extends BinaryFocalLoss (γ=3.0, α=0.75) with a Shannon-entropy bonus on
    the ColBERT attention weights.  High entropy ↔ spread attention across
    all fragment pairs; the penalty discourages the model from collapsing onto
    a single dominant fragment match.

Loss formula
------------
    L_total = L_focal(logits, targets)
            - λ · mean_over_elements( A · log(A + ε) )

Since  -mean(A·log A) = H(A)/n  is the mean per-element entropy (positive for
A ∈ (0,1)), subtracting λ·Σ(A·log A) adds λ·H to the total loss.  Minimising
L_total therefore simultaneously minimises the focal classification error and
maximises the attention entropy — encouraging diverse fragment-pair matching.

Usage
-----
criterion = EntropyRegularizedFocalLoss(lambda_ent=0.01)
loss = criterion(logits, labels, attention_weights=A)  # A: [B, K, K]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Base focal loss (Phase 3 definition, copied verbatim for lineage clarity) ─

class BinaryFocalLoss(nn.Module):
    """Binary focal loss — γ=3.0, α=0.75 by default (Phase 3 spec)."""

    def __init__(self, alpha: float = 0.75, gamma: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self._bce  = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self._bce(logits, targets)
        pt  = torch.exp(-bce)
        return (self.alpha * (1.0 - pt) ** self.gamma * bce).mean()


# ── Phase 4 loss ──────────────────────────────────────────────────────────────

class EntropyRegularizedFocalLoss(nn.Module):
    """BinaryFocalLoss + Shannon-entropy penalty on ColBERT attention weights.

    Args:
        alpha      : Focal-loss α parameter (default 0.75).
        gamma      : Focal-loss γ parameter (default 3.0).
        lambda_ent : Entropy-regularisation coefficient λ (default 0.01).
                     Set to 0.0 to recover plain BinaryFocalLoss.

    Forward signature
    -----------------
    forward(logits, targets, attention_weights) → scalar loss

        logits            : [batch]   or [batch, 1]  — raw sigmoid inputs
        targets           : [batch]   or [batch, 1]  — binary labels {0, 1}
        attention_weights : [batch, K_Q, K_D]        — softmax-normalised
                            interaction matrix from Phase4ColBERTHead.
                            Values must lie in (0, 1] and sum to 1 along the
                            last dimension (softmax contract).
    """

    _LOG_EPS: float = 1e-8

    def __init__(
        self,
        alpha:      float = 0.75,
        gamma:      float = 3.0,
        lambda_ent: float = 0.01,
    ):
        super().__init__()
        self.focal     = BinaryFocalLoss(alpha=alpha, gamma=gamma)
        self.lambda_ent = lambda_ent

    def forward(
        self,
        logits:            torch.Tensor,   # [batch] or [batch, 1]
        targets:           torch.Tensor,   # [batch] or [batch, 1]
        attention_weights: torch.Tensor,   # [batch, K_Q, K_D]
    ) -> torch.Tensor:
        # ── Focal classification term ─────────────────────────────────────────
        L_focal = self.focal(logits.view(-1), targets.view(-1).float())

        if self.lambda_ent == 0.0:
            return L_focal

        # ── Shannon entropy bonus ─────────────────────────────────────────────
        # A · log(A + ε) is negative for A ∈ (0,1).
        # mean(A · log A) < 0  →  -λ · mean(A · log A) > 0  →  adds to loss.
        # Minimising L_total therefore maximises entropy H(A).
        A = attention_weights.clamp(min=self._LOG_EPS)  # numerical safety
        ent_term = (A * torch.log(A)).mean()            # negative scalar

        L_total = L_focal - self.lambda_ent * ent_term
        return L_total
