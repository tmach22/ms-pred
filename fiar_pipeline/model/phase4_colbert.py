import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornFiLMProbe(nn.Module):
    def __init__(
        self,
        cfg: dict,
        phase2_ckpt: str,
        device: torch.device,
        max_fragments: int = 50,
        sinkhorn_eps: float = 0.01,
        sinkhorn_iters: int = 20,
        num_heads: int = 8,
        load_backbone: bool = True,
        rho_init: float = 1.0,    # NOW DYNAMIC: Initial Unbalanced OT mass-forgiveness
        tau_init: float = 5.0     # Initial steepness for Exponential Cost Gate
    ):
        super().__init__()
        self.device = device
        self.max_fragments = max_fragments
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.num_heads = num_heads

        # 2048 / num_heads dimensions per head
        self.head_dim = 2048 // self.num_heads

        # 1. Conditionally Load the frozen Phase 2.5 Backbone
        if load_backbone:
            from fiar_pipeline.model.symmetric_colbert_probe import Phase3ColBERTModel
            self._dummy_phase3 = Phase3ColBERTModel(cfg, phase2_ckpt, device, max_fragments)
            self.backbone = self._dummy_phase3.backbone

            # Strictly freeze the backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            self.backbone = nn.Module()

        # 2. Thermodynamic FiLM Modulator (~65k params) with Regularization
        self.film_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.Dropout(p=0.2), # <--- ADDED: Prevents Thermodynamic Memorization
            nn.ReLU(),
            nn.Linear(16, 4096)
        )

        # Identity Initialization: gamma=1, beta=0
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.constant_(self.film_mlp[-1].bias[:2048], 1.0)
        nn.init.constant_(self.film_mlp[-1].bias[2048:], 0.0)

        # 3. Learnable Tension Parameters (The Sinkhorn Trap Fix)
        self.tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32))
        self.rho_raw = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))

        # 4. The Multi-Head Non-Linear Metric Probe
        # Upgraded to bend the latent space and separate compressed distributions
        self.wasserstein_probe = nn.Sequential(
            nn.Linear(self.num_heads, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )

    def _compute_gated_multihead_cost(self, E_A: torch.Tensor, E_B: torch.Tensor) -> torch.Tensor:
        """
        Computes the Exponentially Gated Cost Matrix.
        Outputs: [B, H, K, K]
        """
        B, K, _ = E_A.shape
        H, D = self.num_heads, self.head_dim

        E_A_h = E_A.view(B, K, H, D).permute(0, 2, 1, 3)
        E_B_h = E_B.view(B, K, H, D).permute(0, 2, 1, 3)

        E_A_norm = F.normalize(E_A_h, p=2, dim=-1)
        E_B_norm = F.normalize(E_B_h, p=2, dim=-1)

        cos_sim = torch.matmul(E_A_norm, E_B_norm.transpose(-1, -2))
        base_cost = 1.0 - cos_sim

        # Exponential Gate
        gated_cost = torch.exp(self.tau * base_cost) - 1.0
        return gated_cost

    def _unbalanced_sinkhorn_knopp_log(self, C: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable Log-Domain Unbalanced Sinkhorn Optimal Transport.
        """
        mu = (mu + 1e-8) / (mu + 1e-8).sum(dim=-1, keepdim=True)
        nu = (nu + 1e-8) / (nu + 1e-8).sum(dim=-1, keepdim=True)

        log_mu = torch.log(mu).unsqueeze(1)
        log_nu = torch.log(nu).unsqueeze(1)

        u = torch.zeros_like(log_mu).expand(-1, self.num_heads, -1).clone()
        v = torch.zeros_like(log_nu).expand(-1, self.num_heads, -1).clone()

        # Enforce strict positivity for the learnable rho parameter via softplus
        active_rho = F.softplus(self.rho_raw)

        # The Dynamic Unbalanced OT Scaling Factor
        f = active_rho / (active_rho + self.sinkhorn_eps)

        for _ in range(self.sinkhorn_iters):
            arg_u = (-C / self.sinkhorn_eps) + v.unsqueeze(-2)
            u = f * (log_mu - torch.logsumexp(arg_u, dim=-1))

            arg_v = (-C / self.sinkhorn_eps) + u.unsqueeze(-1)
            v = f * (log_nu - torch.logsumexp(arg_v, dim=-2))

        log_P = (-C / self.sinkhorn_eps) + u.unsqueeze(-1) + v.unsqueeze(-2)
        P = torch.exp(log_P)

        wasserstein_dist = torch.sum(P * C, dim=(-2, -1))
        return wasserstein_dist

    def forward(self, batch_A: dict, batch_B: dict):
        E_frozen_A = batch_A["frag_fps"].float()
        E_frozen_B = batch_B["frag_fps"].float()

        T_A = batch_A["thermo_state"]
        T_B = batch_B["thermo_state"]

        film_out_A = self.film_mlp(T_A)
        gamma_A, beta_A = torch.split(film_out_A, 2048, dim=-1)
        E_adapted_A = gamma_A * E_frozen_A + beta_A

        film_out_B = self.film_mlp(T_B)
        gamma_B, beta_B = torch.split(film_out_B, 2048, dim=-1)
        E_adapted_B = gamma_B * E_frozen_B + beta_B

        gated_cost = self._compute_gated_multihead_cost(E_adapted_A, E_adapted_B)

        mass_A = batch_A["frag_mass_fractions"]
        mass_B = batch_B["frag_mass_fractions"]

        w_dist = self._unbalanced_sinkhorn_knopp_log(gated_cost, mass_A, mass_B)

        logits = self.wasserstein_probe(-w_dist)
        return logits, w_dist
