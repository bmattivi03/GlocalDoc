import torch
import torch.nn.functional as F


def alignment_loss(z_proj, z_prime):
    """1 - mean cosine similarity. z_proj and z_prime are (B, 768)."""
    return 1.0 - F.cosine_similarity(z_proj, z_prime, dim=-1).mean()


def compression_loss(mu, sigma):
    """
    Closed-form KL( N(mu, sigma^2) || N(0,1) ).
    mu and sigma are (B, hidden_dim).
    """
    return -0.5 * torch.sum(
        1 + 2 * torch.log(sigma) - mu.pow(2) - sigma.pow(2), dim=-1
    ).mean()


def glocal_ib_loss(z_proj, z_prime, mu, sigma, beta, disable_ib=False):
    """
    Total GlocalIB loss.
    disable_ib=True → beta=0 ablation (alignment only, IB term zeroed out).
    Returns: (total_loss, l_align, l_compress)
    """
    l_align = alignment_loss(z_proj, z_prime)
    l_compress = compression_loss(mu, sigma)
    if disable_ib:
        return l_align, l_align, torch.tensor(0.0, device=z_proj.device)
    return l_align + beta * l_compress, l_align, l_compress
