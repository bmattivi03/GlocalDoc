import torch
import torch.nn.functional as F


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """1 − mean cosine similarity. Inputs: (N, D)."""
    if z1.dim() == 3:
        z1 = z1.view(-1, z1.size(-1))
        z2 = z2.view(-1, z2.size(-1))
    # FIX: Ensure we are not comparing a tensor to itself (done in model.py by using different passes)
    return 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()


def compression_loss(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL( N(mu, exp(log_sigma)²) ∥ N(0,1) ) — closed form. Inputs: (B, H)."""
    return -0.5 * torch.sum(
        1 + 2 * log_sigma - mu.pow(2) - torch.exp(2 * log_sigma), dim=-1
    ).mean()


def glocal_ib_loss(
    Z_prime:     torch.Tensor,           # (B, 768) — teacher full doc
    Z_proj:      torch.Tensor,           # (B, 768) — student IB projection
    Z_inter_s:   torch.Tensor,           # (B, 768) — student partial aggregate
    Z_inter_t:   torch.Tensor,           # (B, 768) — teacher partial aggregate
    chunks_s:    list,                   # list[Tensor(M, 768)]
    chunks_t:    list,                   # list[Tensor(M, 768)]
    mu:          torch.Tensor,           # (B, 256)
    log_sigma:   torch.Tensor,           # (B, 256)
    log_s:       torch.Tensor,           # (3,)  learnable (local, inter, global)
    beta:        float = 1e-4,           # Fixed IB trade-off
    disable_ib:  bool = False,
):
    """
    Four-component hierarchical GlocalIB loss.
    Compression is fixed (beta), while alignments use uncertainty weighting.
    """
    l_compress = compression_loss(mu, log_sigma)
    l_local    = alignment_loss(torch.cat(chunks_s), torch.cat(chunks_t))
    l_inter    = alignment_loss(Z_inter_s, Z_inter_t)
    l_global   = alignment_loss(Z_proj, Z_prime)

    if disable_ib:
        # FIX: Ensure all modules receive a gradient (even if 0) to satisfy DDP and prevent crashes.
        # This keeps predictors and heads in the graph even when they don't contribute to the primary loss.
        dummy_loss = 0.0 * (log_s.sum() + l_local + l_inter + l_compress)
        return l_global + dummy_loss, l_compress.detach(), l_local.detach(), l_inter.detach(), l_global

    # Homoscedastic uncertainty weighting for the THREE alignment terms.
    # FIX: Remove hard clamp to prevent dead gradients.
    safe_log_s = log_s
    align_losses = torch.stack([l_local, l_inter, l_global])
    weighted_align = (0.5 * align_losses * torch.exp(-safe_log_s) + 0.5 * safe_log_s).sum()

    # IB total: weighted_align + beta * l_compress
    total = weighted_align + beta * l_compress

    return total, l_compress, l_local, l_inter, l_global
