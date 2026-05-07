import torch
import torch.nn.functional as F


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """1 − mean cosine similarity. z1, z2: (N, D)."""
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def compression_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma²) ∥ N(0,1) ) closed form. mu, sigma: (B, H)."""
    return -0.5 * (1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2)).sum(-1).mean()


def glocal_ib_loss(
    Z_prime:   torch.Tensor,   # (B, 768) teacher full-doc
    Z_proj:    torch.Tensor,   # (B, 768) student post-IB
    z_partial: torch.Tensor,   # (B, 768) student pre-IB (for L_inter)
    s_chunks:  list,           # list[Tensor(N, 768)] student all-para reps
    t_chunks:  list,           # list[Tensor(N, 768)] teacher all-para reps
    mu:        torch.Tensor,   # (B, 256)
    sigma:     torch.Tensor,   # (B, 256)
    log_s:     torch.Tensor,   # (4,) clamped UW weights [compress, local, inter, global]
):
    """
    Four-component hierarchical GlocalIB loss with homoscedastic uncertainty weighting.

    L_compress  KL penalty — forces IB bottleneck to compress
    L_local     align all N paragraph pairs (teacher_i vs student_i)
    L_inter     align student partial pool vs teacher full doc (pre-IB gradient path)
    L_global    align student IB projection vs teacher full doc (post-IB)

    Returns (total, l_compress, l_local, l_inter, l_global).
    """
    l_compress = compression_loss(mu, sigma)
    l_local    = alignment_loss(torch.cat(s_chunks), torch.cat(t_chunks))
    l_inter    = alignment_loss(z_partial, Z_prime.detach())
    l_global   = alignment_loss(Z_proj,    Z_prime.detach())

    losses = torch.stack([l_compress, l_local, l_inter, l_global])
    total  = (losses * torch.exp(-log_s) + log_s).sum()

    return total, l_compress, l_local, l_inter, l_global
