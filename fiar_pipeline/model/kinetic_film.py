"""
KineticFiLM — NCE-Conditioned Feature-wise Linear Modulation
=============================================================
Injects thermodynamic context (normalized collision energy) into fragment
pooled embeddings via a residual Hadamard gate:

    G_final = G_p + G_p * Sigmoid(projector(NCE))

The projector maps a scalar NCE value (CE / 50.0) to a dense gate vector
of the same width as the pooled fragment embedding.  The residual form
(G_p + G_p * gate) is equivalent to G_p * (1 + gate), ensuring that at
initialization (gate ≈ 0.5) the embeddings are scaled by ~1.5x rather
than zeroed, giving a stable training start.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class KineticFiLM(nn.Module):
    """NCE-conditioned multiplicative gate for pooled fragment embeddings.

    Args:
        hidden_size: Dimensionality of the pooled fragment embedding (must
            match FragGNN.hidden_size).
        bottleneck:  Width of the internal projection layer.
    """

    def __init__(self, hidden_size: int, bottleneck: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.projector = nn.Sequential(
            nn.Linear(1, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_size),
        )
        # Zero-init output layer so the gate starts near 0.5 (sigmoid(0))
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)

    def forward(
        self,
        avg_frags: torch.Tensor,     # [n_frags, hidden_size]
        nce_per_frag: torch.Tensor,  # [n_frags] or [n_frags, 1]
    ) -> torch.Tensor:
        """Apply NCE-conditioned residual gate.

        Args:
            avg_frags:    Pooled fragment embeddings from dgl_nn.AvgPooling /
                          GlobalAttentionPooling.  Shape [n_frags, hidden_size].
            nce_per_frag: Per-fragment NCE values (CE_eV / 50.0).  Broadcast-
                          aligned to [n_frags, 1] internally.

        Returns:
            Modulated embeddings of the same shape as avg_frags.
        """
        nce = nce_per_frag.view(-1, 1).float()    # [n_frags, 1]
        E_kinetic = self.projector(nce)            # [n_frags, hidden_size]
        gate = torch.sigmoid(E_kinetic)            # [n_frags, hidden_size]
        return avg_frags + avg_frags * gate        # residual Hadamard product
