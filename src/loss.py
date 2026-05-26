import torch
import torch.nn.functional as F


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """1 − mean cosine similarity. z1, z2: (N, D)."""
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def compression_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    free_bits_nats: float = 0.0,
) -> torch.Tensor:
    """KL( N(mu, sigma²) ∥ N(0,1) ) per sample, summed over latent dims, mean over batch.

    Computed directly from log_sigma (not exp) to stay finite under bf16.
    With free_bits_nats > 0, per-dim KL is clamped from below — the encoder is
    not penalized for using at least λ nats per dim (hard elementwise free bits;
    variant of Kingma 2016). Prevents posterior collapse to N(0,1).
    """
    per_dim_kl = -0.5 * (1 + 2 * log_sigma - mu.pow(2) - torch.exp(2 * log_sigma))
    if free_bits_nats > 0:
        per_dim_kl = torch.clamp(per_dim_kl, min=free_bits_nats)
    return per_dim_kl.sum(-1).mean()


def variance_loss(z: torch.Tensor, gamma: float = 0.5, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance hinge. Encourages per-dim std(z over N) ≥ gamma.

    Below gamma, gradient pulls the encoder away from collapse. Above gamma,
    the loss is zero (no spurious pressure on already-diverse features).
    z: (N, D) where N ≥ 2 (e.g., paragraph reps within a document).
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=True) + eps)
    return F.relu(gamma - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg-style covariance regularizer. Pushes off-diagonal of cov(z) toward 0,
    decorrelating feature dimensions and preventing dimensional collapse.
    z: (N, D) where N ≥ 2.
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    z_centered = z - z.mean(dim=0, keepdim=True)
    n = z.shape[0]
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / z.shape[1]


def glocal_ib_loss(
    Z_prime:   torch.Tensor,   # (B, 768) teacher full-doc
    Z_proj:    torch.Tensor,   # (B, 768) student post-IB post-predictor
    z_partial: torch.Tensor,   # (B, 768) student pre-IB partial-pool post-predictor (for L_inter)
    s_chunks:  list,           # list[Tensor(N, 768)] student all-para reps
    t_chunks:  list,           # list[Tensor(N, 768)] teacher all-para reps
    mu:        torch.Tensor,   # (B, 256)
    log_sigma: torch.Tensor,   # (B, 256) clamped log-σ
    log_s:     torch.Tensor,   # (3,) clamped UW weights [local, inter, global]
    beta_kl: float = 1.0,
    free_bits_nats: float = 0.05,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    var_gamma: float = 0.5,
):
    """V2 loss: UW weights only the 3 alignment-family losses. L_compress is summed
    outside UW with a fixed β coefficient. Anti-collapse VICReg terms remain
    outside UW with fixed weights as in V1.

    P-G-02: the prior V1 stack put L_compress INSIDE UW. With free_bits_nats=0.5
    pinning L_compress ≈ 128 and the cosines bounded in [0, 2], the UW
    optimizer rationally learned log_s_compress ≈ log(128) ≈ 4.85, attenuating
    the compression gradient by ~e^(-4.85) ≈ 0.008. UW assumes per-loss
    likelihoods whose variance can be learned (Kendall et al. arxiv:1705.07115);
    a KL-to-prior is a regularizer, not a likelihood — feeding it to UW is a
    category error that mutes the signal it's supposed to balance.

    Components:
      L_compress    β · KL penalty (with free bits) — FIXED weight, outside UW
      L_local       align all N paragraph pairs (teacher_i vs student_i) — UW
      L_inter       align student partial-pool (predictor) vs teacher pre-IB — UW
      L_global      align student IB projection (predictor) vs teacher post-IB — UW
      L_variance    VICReg variance hinge on per-doc paragraph reps — fixed weight
      L_covariance  VICReg covariance regularizer on per-doc paragraph reps — fixed

    Returns (total, l_compress, l_local, l_inter, l_global, l_variance, l_covariance).
    The 7-tuple shape is unchanged — only the way components are combined into total.
    """
    l_compress = compression_loss(mu, log_sigma, free_bits_nats=free_bits_nats)
    l_local    = alignment_loss(torch.cat(s_chunks), torch.cat(t_chunks))
    l_inter    = alignment_loss(z_partial, Z_prime.detach())
    l_global   = alignment_loss(Z_proj,    Z_prime.detach())

    var_terms = [variance_loss(sc, gamma=var_gamma)  for sc in s_chunks]
    cov_terms = [covariance_loss(sc)                  for sc in s_chunks]
    l_variance   = torch.stack(var_terms).mean() if var_terms else Z_proj.new_zeros(())
    l_covariance = torch.stack(cov_terms).mean() if cov_terms else Z_proj.new_zeros(())

    # UW over the 3 alignment-family losses only.
    losses_uw = torch.stack([l_local, l_inter, l_global])
    uw_term = (losses_uw * torch.exp(-log_s) + log_s).sum()

    total = uw_term \
          + beta_kl * l_compress \
          + var_weight * l_variance \
          + cov_weight * l_covariance

    # Returned l_compress is the RAW KL (post free-bits clamp, pre-β). Multiply
    # by beta_kl downstream if you want the actually-applied term.
    return total, l_compress, l_local, l_inter, l_global, l_variance, l_covariance
