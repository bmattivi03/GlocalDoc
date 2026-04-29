import torch
import torch.nn.functional as F


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """1 − mean cosine similarity. Inputs: (N, D)."""
    if z1.dim() == 3:
        z1 = z1.view(-1, z1.size(-1))
        z2 = z2.view(-1, z2.size(-1))
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def compression_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, sigma²) ∥ N(0,1) ) — closed form. Inputs: (B, H)."""
    return -0.5 * torch.sum(
        1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2), dim=-1
    ).mean()


def glocal_ib_loss(
    Z_prime:     torch.Tensor,           # (B, 768) — teacher full doc
    Z_proj:      torch.Tensor,           # (B, 768) — student IB projection
    Z_inter_s:   torch.Tensor,           # (B, 768) — student partial aggregate
    Z_inter_t:   torch.Tensor,           # (B, 768) — teacher partial aggregate
    chunks_s:    list,                   # list[Tensor(M, 768)]
    chunks_t:    list,                   # list[Tensor(M, 768)]
    mu:          torch.Tensor,           # (B, 256)
    sigma:       torch.Tensor,           # (B, 256)
    log_s:       torch.Tensor,           # (4,)  learnable
    disable_ib:  bool = False,
):
    """
    Four-component hierarchical GlocalIB loss with homoscedastic uncertainty weighting.

    Returns: (total, l_compress, l_local, l_inter, l_global)
    """
    l_compress = compression_loss(mu, sigma)
    l_local    = alignment_loss(torch.cat(chunks_s), torch.cat(chunks_t))
    l_inter    = alignment_loss(Z_inter_s, Z_inter_t)
    l_global   = alignment_loss(Z_proj, Z_prime)

    if disable_ib:
        return l_global, torch.zeros(1), torch.zeros(1), torch.zeros(1), l_global

    # Homoscedastic uncertainty weighting (Kendall, Gal & Cipolla, CVPR 2018)
    losses = torch.stack([l_compress, l_local, l_inter, l_global])
    total  = (losses * torch.exp(-log_s) + log_s).sum()

    return total, l_compress, l_local, l_inter, l_global
