"""P-A-01 — MI diagnostics for the IB bottleneck.

Three estimators, ordered by cost:

1. `effective_rank` — Roy-Vetterli effective rank of a (B, D) matrix. Free
   diagnostic; high effective rank ≈ low dimensional collapse.
2. `infonce_lower_bound` — van den Oord et al. 2018, no extra parameters.
   Treats in-batch (mu_i, z_partial_i) as positives and cross-batch as
   negatives. Gives a lower bound `log(B) - L_NCE` on I(Z; X).
3. `CLUB` — Cheng et al. 2020 contrastive log-ratio upper bound on MI.
   Trains a small variational critic q(z|x) alongside the main model.

The pretrain loop logs (1) and (2) every COLLAPSE_LOG_EVERY steps and writes
both to W&B. (3) is opt-in: if a CLUB instance is constructed and stepped,
its estimate is also logged. The blueprint's hypothesis is that V1's IB does
not compress — i.e. CLUB(Z; X) ≈ InfoNCE_LB(Z; X) ≈ I(Z_partial; X), the bound
is trivial. A V2 where the IB actually compresses should show CLUB(Z; X)
**below** I(Z_partial; X) by a measurable margin.

References:
- arxiv:1807.03748 — van den Oord et al., InfoNCE / CPC
- arxiv:2006.12013 — Cheng et al., CLUB
- doi:10.1109/EUSIPCO.2007.7099037 — Roy & Vetterli, effective rank
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(x: torch.Tensor, eps: float = 1e-9) -> float:
    """exp(-Σ p_i log p_i) where p_i = σ_i / Σ σ_j and σ_i are SVDs of `x`.

    x: (B, D). Returns the effective rank — a continuous proxy for dim collapse.
    Effective rank = D means perfectly isotropic; effective rank = 1 means rank-1
    collapse. Cheap (one SVD per call); compute on a detached, fp32 tensor.
    """
    x = x.detach().float()
    if x.shape[0] < 2 or x.shape[1] < 2:
        return 0.0
    x_c = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x_c)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    h = -(p * (p + eps).log()).sum()
    return float(torch.exp(h).item())


def infonce_lower_bound(z: torch.Tensor, x: torch.Tensor, tau: float = 0.1) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE lower bound on I(Z; X) using in-batch negatives.

    Lower bound: log(B) - L_NCE where L_NCE is the symmetric InfoNCE loss.
    z, x: (B, D_z) and (B, D_x). They are L2-normalized first; the bound is
    therefore on the cosine-similarity-based contrastive task.

    Returns:
      L_NCE   the loss tensor (gradient-bearing if z or x requires grad)
      lb_nats the InfoNCE lower bound on I(Z; X) in nats (float)
    """
    B = z.shape[0]
    if B < 2:
        return z.new_zeros(()), 0.0
    z_n = F.normalize(z, dim=-1)
    x_n = F.normalize(x, dim=-1)
    sim = (z_n @ x_n.T) / tau                      # (B, B)
    target = torch.arange(B, device=z.device)
    loss_zx = F.cross_entropy(sim, target)
    loss_xz = F.cross_entropy(sim.T, target)
    L_NCE = 0.5 * (loss_zx + loss_xz)
    lb_nats = float(math.log(B) - L_NCE.detach().item())
    return L_NCE, lb_nats


class CLUB(nn.Module):
    """Contrastive Log-ratio Upper Bound on I(Z; X) (Cheng et al. 2020).

    Trains a small variational critic q(z | x). The upper bound is
        I(Z; X) ≤ E_p(x,z)[log q(z|x)] - E_p(x) E_p(z)[log q(z|x)]
    estimated as a per-batch difference of paired vs unpaired log-densities.

    Usage:
        club = CLUB(dim_x=768, dim_z=256).to(device)
        club_opt = torch.optim.Adam(club.parameters(), lr=1e-4)

        # during pretrain inner loop, every N steps:
        with torch.no_grad():
            x_det = z_partial.detach()    # (B, 768)
            z_det = mu.detach()           # (B, 256)
        club_opt.zero_grad()
        club_loss = -club.learning_loss(x_det, z_det)  # MLE update
        club_loss.backward()
        club_opt.step()

        with torch.no_grad():
            mi_upper = club.mi_upper_bound(x_det, z_det)

    The critic is small (~200 K params at default sizes), so its compute is
    a rounding error vs the main encoder.
    """

    def __init__(self, dim_x: int = 768, dim_z: int = 256, hidden: int = 256):
        super().__init__()
        self.mu_head = nn.Sequential(
            nn.Linear(dim_x, hidden), nn.ReLU(),
            nn.Linear(hidden, dim_z),
        )
        self.log_sigma_head = nn.Sequential(
            nn.Linear(dim_x, hidden), nn.ReLU(),
            nn.Linear(hidden, dim_z), nn.Tanh(),   # tanh keeps log σ ∈ [-1, 1]
        )

    def _log_q(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """log q(z | x) under a diagonal Gaussian critic. (B,) returned."""
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        # Gaussian log-density, sum over dims
        return (-0.5 * ((z - mu) / log_sigma.exp()).pow(2) - log_sigma).sum(-1)

    def learning_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood for critic training. Minimize this."""
        return -self._log_q(x, z).mean()

    @torch.no_grad()
    def mi_upper_bound(self, x: torch.Tensor, z: torch.Tensor) -> float:
        """Returns the CLUB upper bound on I(Z; X) in nats."""
        # Paired log-density
        lq_pos = self._log_q(x, z)                                # (B,)
        # Unpaired: every (x_i, z_j) cross product
        B = x.shape[0]
        if B < 2:
            return 0.0
        mu = self.mu_head(x)                                      # (B, D)
        log_sigma = self.log_sigma_head(x)                        # (B, D)
        # log q(z_j | x_i) for all i, j
        diff = z.unsqueeze(0) - mu.unsqueeze(1)                   # (B, B, D)
        lq_all = (-0.5 * (diff / log_sigma.unsqueeze(1).exp()).pow(2)
                  - log_sigma.unsqueeze(1)).sum(-1)               # (B, B)
        lq_neg = lq_all.mean()
        return float((lq_pos.mean() - lq_neg).item())
