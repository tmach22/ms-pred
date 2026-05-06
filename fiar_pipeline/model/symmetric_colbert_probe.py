"""
Phase 3: Symmetric ColBERT Probe
==================================
Self-contained model for Phase 3 training.  Loads the Phase 2.5 MassFormer +
Thermodynamic Adapter backbone with strict=False, freezes it entirely, and
adds a trainable SymmetricColBERTHead on top.

Architecture
------------
Frozen  : MassFormer backbone + Phase 2.5 Thermodynamic Adapters
Trainable: SymmetricColBERTHead (DynamicWeightingMLP + binary_head)

The ColBERT head treats each molecule's Top-K cached ICEBERG fragments as
token embeddings (Morgan FPs, shape [batch, K, 2048]) and computes a
dynamically-weighted symmetric MaxSim score between the two fragment sets.

Metadata tensor (per fragment, shape [batch, K, 3]):
  [0] relative_mass  = frag_exact_mass / precursor_mz
  [1] log_prob_gen   = log(prob_gen + 1e-8)
  [2] NCE            = collision_energy / 50.0

Usage
-----
from fiar_pipeline.model.symmetric_colbert_probe import Phase3ColBERTModel

model = Phase3ColBERTModel(cfg, phase2_ckpt, device, max_fragments=50)
logits = model(batch_A, batch_B)  # [batch, 1]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── ColBERT head (exact spec) ─────────────────────────────────────────────────

class DynamicWeightingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: relative_mass, log(prob_gen), normalized_collision_energy (NCE)
        self.mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, metadata_tensor: torch.Tensor) -> torch.Tensor:
        # metadata_tensor shape: [batch_size, num_fragments, 3]
        raw_weights = self.mlp(metadata_tensor).squeeze(-1)  # [batch, num_frags]
        # Softmax over the fragments to get competitive importance
        return F.softmax(raw_weights, dim=-1)


class SymmetricColBERTHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weighting_mlp = DynamicWeightingMLP()
        # The Final Focal Scalpel (Linear binary head on top of the unbounded MaxSim)
        self.binary_head = nn.Linear(1, 1)

    def asymmetric_maxsim(
        self,
        embeds_A: torch.Tensor,   # [batch, num_frags_A, dim]
        embeds_B: torch.Tensor,   # [batch, num_frags_B, dim]
        weights_A: torch.Tensor,  # [batch, num_frags_A]
    ) -> torch.Tensor:            # [batch]
        # Normalize embeddings for cosine similarity
        A_norm = F.normalize(embeds_A, p=2, dim=-1)
        B_norm = F.normalize(embeds_B, p=2, dim=-1)

        # Batch matrix multiplication: [batch, frags_A, frags_B]
        sim_matrix = torch.bmm(A_norm, B_norm.transpose(1, 2))

        # Max over B's fragments for each of A's fragments: [batch, frags_A]
        max_sims, _ = torch.max(sim_matrix, dim=-1)

        # Multiply by the dynamically learned weights and sum
        return torch.sum(max_sims * weights_A, dim=-1)  # [batch]

    def forward(
        self,
        embeds_A: torch.Tensor,  # [batch, K, dim]
        meta_A:   torch.Tensor,  # [batch, K, 3]
        embeds_B: torch.Tensor,  # [batch, K, dim]
        meta_B:   torch.Tensor,  # [batch, K, 3]
    ) -> torch.Tensor:           # [batch, 1]
        # 1. Calculate dynamic weights
        w_A = self.weighting_mlp(meta_A)
        w_B = self.weighting_mlp(meta_B)

        # 2. Calculate Bidirectional Asymmetric MaxSim
        S_A_to_B = self.asymmetric_maxsim(embeds_A, embeds_B, w_A)
        S_B_to_A = self.asymmetric_maxsim(embeds_B, embeds_A, w_B)

        # 3. The Symmetric Average
        S_sym = 0.5 * (S_A_to_B + S_B_to_A)  # [batch]

        # 4. The Focal Scalpel
        logits = self.binary_head(S_sym.unsqueeze(-1))  # [batch, 1]
        return logits


# ── Full Phase 3 model ────────────────────────────────────────────────────────

class Phase3ColBERTModel(nn.Module):
    """
    Phase 3 model: frozen Phase 2.5 backbone + trainable SymmetricColBERTHead.

    Backbone loading
    ----------------
    The Phase 2.5 checkpoint is loaded with strict=False. Missing keys
    (colbert_head.*) and unexpected keys are printed explicitly so the caller
    can verify the backbone loaded perfectly and only the new head was
    initialized from scratch.

    BatchNorm patch (caller responsibility)
    ----------------------------------------
    Call model.eval() globally at the start of each training epoch, then
    model.colbert_head.train() to keep only the ColBERT head in training mode.
    This prevents frozen BatchNorm layers in the backbone from updating their
    running statistics and corrupting the learned manifold.

    Forward inputs (from siamese_frag_collate_fn output)
    -----------------------------------------------------
    batch_A / batch_B must contain:
      frag_fps    : FloatTensor [batch, max_k, 2048]  — Morgan FP embeddings
      frag_masses : FloatTensor [batch, max_k]        — exact fragment masses
      frag_probs  : FloatTensor [batch, max_k]        — prob_gen per fragment
      prec_mz     : FloatTensor [batch]               — precursor m/z
      nce         : FloatTensor [batch]               — normalized CE (CE/50)
    """

    _LOG_EPS = 1e-8

    def __init__(
        self,
        cfg: dict,
        phase2_ckpt: str,
        device: torch.device,
        max_fragments: int = 50,
    ):
        super().__init__()
        self.max_fragments = max_fragments

        # ── Locate and import backbone class ─────────────────────────────────
        phase2_loader_dir = cfg["data"]["phase2_loader_dir"]
        for search_path in [
            phase2_loader_dir,
            str(Path(phase2_loader_dir).parents[1] / "model" / "fiar"),
        ]:
            abs_p = str(Path(search_path).resolve())
            if abs_p not in sys.path:
                sys.path.insert(0, abs_p)

        try:
            from phase3_multitask_siamese import (  # type: ignore
                Phase3_LinearProbe_SiameseNetwork,
            )
        except ImportError as exc:
            raise ImportError(
                f"Cannot import Phase3_LinearProbe_SiameseNetwork.\n"
                f"Check data.phase2_loader_dir in config: {phase2_loader_dir}\n"
                f"Original error: {exc}"
            )

        # Force the exact Phase 2.5 architectural config to prevent MassFormer
        # attribute errors — the backbone is frozen so these must match the
        # state it was trained with exactly.
        cfg["model"] = {
            "embed_types":              ["gf_v2"],
            "gf_model_name":            "graphormer_base",
            "gf_pretrain_name":         "none",
            "fix_num_pt_layers":        0,
            "reinit_num_pt_layers":     0,
            "reinit_layernorm":         False,
            "embed_dim":                -1,
            "embed_linear":             False,
            "ff_layer_type":            "neims",
            "ff_h_dim":                 1000,
            "ff_num_layers":            4,
            "ff_skip":                  True,
            "output_normalization":     "l1",
            "bidirectional_prediction": True,
            "spectrum_attention":       False,
            "gate_prediction":          False,
            "model_seed":               0,
            "dropout":                  0.15,
        }

        # Construct backbone — this internally loads the Phase 2.5 checkpoint.
        self.backbone = Phase3_LinearProbe_SiameseNetwork(
            cfg,
            phase2_ckpt,
            device,
            max_fragments=cfg["model"].get("max_fragments", 10),
        )

        # ── Namespace check: strict=False analysis ────────────────────────────
        # Load the raw checkpoint and diff its keys against the backbone's
        # state_dict so we can verify that all backbone weights transferred
        # perfectly and that the only new keys are from colbert_head.
        print(f"\n[Phase3ColBERT] ── Namespace check (strict=False) ────────────")
        print(f"[Phase3ColBERT] Checkpoint: {phase2_ckpt}")

        raw_ckpt = torch.load(phase2_ckpt, map_location=device, weights_only=False)
        if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
            raw_ckpt = raw_ckpt["state_dict"]

        backbone_keys = set(self.backbone.state_dict().keys())
        ckpt_keys     = set(raw_ckpt.keys())
        missing_keys  = sorted(backbone_keys - ckpt_keys)
        unexpected_keys = sorted(ckpt_keys - backbone_keys)

        if missing_keys:
            print(f"[Phase3ColBERT] missing_keys  ({len(missing_keys)}):")
            for k in missing_keys:
                print(f"    {k}")
        else:
            print("[Phase3ColBERT] missing_keys  : none ✓")

        if unexpected_keys:
            print(f"[Phase3ColBERT] unexpected_keys ({len(unexpected_keys)}):")
            for k in unexpected_keys:
                print(f"    {k}")
        else:
            print("[Phase3ColBERT] unexpected_keys: none ✓")
        print("[Phase3ColBERT] ─────────────────────────────────────────────────\n")

        # ── Freeze backbone entirely ──────────────────────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False
        frozen_n = sum(p.numel() for p in self.backbone.parameters())
        print(f"[Phase3ColBERT] Frozen backbone params  : {frozen_n:,}")

        # ── Trainable ColBERT head ────────────────────────────────────────────
        self.colbert_head = SymmetricColBERTHead()
        trainable_n = sum(p.numel() for p in self.colbert_head.parameters())
        print(f"[Phase3ColBERT] Trainable ColBERT params: {trainable_n:,}\n")

    # ── Metadata builder ──────────────────────────────────────────────────────

    def _build_metadata(
        self,
        frag_masses: torch.Tensor,  # [batch, K]
        frag_probs:  torch.Tensor,  # [batch, K]
        prec_mz:     torch.Tensor,  # [batch]
        nce:         torch.Tensor,  # [batch]
    ) -> torch.Tensor:              # [batch, K, 3]
        K = frag_masses.size(1)
        rel_mass    = frag_masses / prec_mz.unsqueeze(1).clamp(min=1.0)
        log_prob    = torch.log(frag_probs + self._LOG_EPS)
        nce_expand  = nce.unsqueeze(1).expand(-1, K)
        return torch.stack([rel_mass, log_prob, nce_expand], dim=-1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch_A: dict, batch_B: dict) -> torch.Tensor:
        """
        Run the SymmetricColBERTHead on the cached ICEBERG fragment FPs.
        The frozen backbone is NOT invoked in the forward pass — it is carried
        in the checkpoint for lineage purposes only.

        Returns logits of shape [batch, 1].
        """
        K = self.max_fragments

        embeds_A = batch_A["frag_fps"][:, :K]     # [batch, K, 2048]
        embeds_B = batch_B["frag_fps"][:, :K]

        meta_A = self._build_metadata(
            batch_A["frag_masses"][:, :K],
            batch_A["frag_probs"][:, :K],
            batch_A["prec_mz"],
            batch_A["nce"],
        )
        meta_B = self._build_metadata(
            batch_B["frag_masses"][:, :K],
            batch_B["frag_probs"][:, :K],
            batch_B["prec_mz"],
            batch_B["nce"],
        )

        return self.colbert_head(embeds_A, meta_A, embeds_B, meta_B)
