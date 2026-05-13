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


def glocal_ib_loss(
    Z_prime:   torch.Tensor,   # (B, 768) teacher full-doc
    Z_proj:    torch.Tensor,   # (B, 768) student post-IB
    z_partial: torch.Tensor,   # (B, 768) student pre-IB (for L_inter)
    s_chunks:  list,           # list[Tensor(N, 768)] student all-para reps
    t_chunks:  list,           # list[Tensor(N, 768)] teacher all-para reps
    mu:        torch.Tensor,   # (B, 256)
    log_sigma: torch.Tensor,   # (B, 256) clamped log-σ
    log_s:     torch.Tensor,   # (4,) clamped UW weights [compress, local, inter, global]
    beta_kl: float = 1.0,
    free_bits_nats: float = 0.5,
):
    """
    Four-component hierarchical GlocalIB loss with homoscedastic uncertainty weighting.

    L_compress  β · KL penalty (with free bits) — forces IB bottleneck to compress
    L_local     align all N paragraph pairs (teacher_i vs student_i)
    L_inter     align student partial pool vs teacher full doc (pre-IB gradient path)
    L_global    align student IB projection vs teacher full doc (post-IB)

    beta_kl scales the raw compression loss before it enters the UW stack; UW
    (exp(−log_s_compress)) adapts to whatever scale β·KL settles at. Ramping β
    from 0 prevents the large init KL from crushing μ before alignment losses
    can shape the bottleneck.

    Returns (total, l_compress, l_local, l_inter, l_global).
    """
    l_compress = beta_kl * compression_loss(mu, log_sigma, free_bits_nats=free_bits_nats)
    l_local    = alignment_loss(torch.cat(s_chunks), torch.cat(t_chunks))
    l_inter    = alignment_loss(z_partial, Z_prime.detach())
    l_global   = alignment_loss(Z_proj,    Z_prime.detach())

    losses = torch.stack([l_compress, l_local, l_inter, l_global])
    total  = (losses * torch.exp(-log_s) + log_s).sum()

    return total, l_compress, l_local, l_inter, l_global
