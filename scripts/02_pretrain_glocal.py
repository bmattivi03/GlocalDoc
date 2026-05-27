import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # silence unauth-request warning
# Titan Xp has no NVLink — PCIe-only P2P deadlocks the DDP param broadcast at startup.
# Force NCCL onto shared-memory/socket transports; harmless on NVLink hardware and single GPU.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
# Surface NCCL hangs as readable errors instead of silent 10-min timeouts.
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import sys
import time
import random
from collections import deque
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator, InitProcessGroupKwargs

# Suppress benign per-paragraph warnings (long tokens are sub-chunked manually,
# use_cache is explicitly disabled alongside gradient checkpointing).
transformers.logging.set_verbosity_error()

sys.path.append(".")
from src.data import load_ecthr, mask_text, get_paragraph_mask
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss, variance_loss, covariance_loss
from src.diagnostics import effective_rank, infonce_lower_bound

# ── CONFIG ────────────────────────────────────────────────────────────────────
# P-H-01: pin pretrain RNG. Without this, V1-vs-V2-vs-H-MLM comparisons are
# polluted by random init drift larger than the disputed few-shot gap
# (MDE at 5 fine-tune seeds ≈ 3.7 macro-F1).
PRETRAIN_SEED       = 1337
EPOCHS              = 5
BATCH_SIZE          = 1       # per GPU; effective = BATCH_SIZE × num_GPUs × GRAD_ACCUM
GRAD_ACCUM          = 8
# P-G-03: 4-group AdamW. Encoder LR drives the trunk; the small heads (IB,
# projector, predictor) are randomly initialized and need a higher LR to catch
# up. log_s needs ~333× encoder LR with weight_decay=0 to actually converge.
LR_ENCODER          = 3e-5
LR_IB_HEAD          = 1e-4    # ~3.3× encoder — random init, needs to move
LR_PROJ_PRED        = 1e-4    # projector + predictor: same ratio
LR_LOG_S            = 1e-2    # ~333× encoder LR — required for UW to converge
# P-G-03: per-group clip contract. Encoder gets the standard 1.0; small heads
# get tighter 0.5 so a noisy critic gradient cannot push the projector around.
CLIP_ENCODER        = 1.0
CLIP_IB_HEAD        = 0.5
CLIP_PROJ_PRED      = 0.5
# P-G-04: EMA τ ramps from EMA_TAU_START to EMA_TAU_END. Slower teacher
# evolution at the end of training stabilizes the alignment targets when the
# student is near convergence.
EMA_TAU_START       = 0.996   # DINO's initial value
EMA_TAU_END         = 0.9999
BETA_KL_FINAL       = 1.0
BETA_KL_WARMUP_FRAC = 0.25    # ramp β over first 25% of total optimizer steps
FREE_BITS_NATS      = 0.05    # P-G-01: per-dim hard floor at 0.05 nats/dim → 12.8 nat total floor.
                              # The prior value of 0.5 was the V1 design's #1 confound: per-dim KL
                              # is ≈ 0 early in training (μ→0, σ→1), so torch.clamp(min=0.5)
                              # returns 0.5 with zero subgradient through mu_head. The encoder is
                              # never penalized for storing 0 information up to the floor → IB is
                              # decorative. 0.05 is the Kingma 2016 free-bits scale (~per-group
                              # threshold); 0.5 disables compression entirely. See [arxiv:1606.04934].
VAR_WEIGHT          = 5.0     # encoder s_chunks anti-collapse
COV_WEIGHT          = 0.04    # VICReg default
VAR_GAMMA           = 0.5     # per-dim std hinge threshold (used for s_chunks AND temporal buffers)
# P-C-01: per-paragraph token MLM loss outside UW. Fixed weight — the goal is
# to match H-MLM's per-paragraph supervision-density signal alongside GlocalIB,
# not to outweigh it. β=1.0 matches the H-MLM loss weighting in the SMITH-style
# baseline (scripts/03_pretrain_mlm.py uses 0.5/0.5).
MLM_WEIGHT          = 1.0
# Temporal anti-collapse on Z_proj_pred and mu (rolling GPU buffers, gradient flows
# through current sample only). Addresses the downstream collapse mode where the
# IB+projector+predictor chain maps varied encoder outputs to constant final reps.
Z_PROJ_VAR_WEIGHT   = 25.0    # temporal variance on Z_proj_pred (VICReg paper's variance weight)
Z_PROJ_COV_WEIGHT   = 1.0     # temporal covariance on Z_proj_pred
MU_VAR_WEIGHT       = 25.0    # temporal variance on mu (high — bottleneck must remain informative)
MU_COV_WEIGHT       = 1.0     # temporal covariance on mu
TEMPORAL_BUF_LEN    = 16      # how many past samples to keep in GPU buffer
TEMPORAL_MIN_FILL   = 4       # min buffer fill before temporal loss activates
LOG_EVERY           = 1       # W&B log frequency (optimizer steps)
PRINT_EVERY         = 50      # stdout summary frequency (optimizer steps)
COLLAPSE_LOG_EVERY  = 25      # how often to compute inter-doc collapse metric (was 50 — catch earlier)
COLLAPSE_ALARM_THR  = 0.85    # warn if rolling inter-doc cosine exceeds this (was 0.95 — catch earlier)
Z_BUFFER_LEN        = 16      # rolling buffer of past Z_proj for collapse metric (cpu, diagnostic)
WANDB_PROJECT       = "glocal-nlp"
CONDITION           = "glocal_ib"
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return [item["text"] for item in batch]


def beta_kl_schedule(step: int, total_steps: int) -> float:
    warmup = max(1, int(total_steps * BETA_KL_WARMUP_FRAC))
    return BETA_KL_FINAL * min(1.0, step / warmup)


def ema_tau_schedule(step: int, total_steps: int) -> float:
    """P-G-04: linear ramp from EMA_TAU_START to EMA_TAU_END over training.
    Slower teacher evolution near the end → stable alignment targets when
    student is near convergence."""
    p = min(1.0, step / max(1, total_steps))
    return EMA_TAU_START + p * (EMA_TAU_END - EMA_TAU_START)


def mean_pairwise_cosine(buf: deque) -> float:
    """Average pairwise cosine over a deque of (D,) tensors. Returns 0.0 if too short."""
    if len(buf) < 2:
        return 0.0
    z = torch.stack(list(buf))                  # (K, D), already fp32 cpu detached
    z = F.normalize(z, dim=-1)
    sims = z @ z.T                              # (K, K)
    K = z.shape[0]
    # off-diagonal mean
    return ((sims.sum() - K) / (K * (K - 1))).item()


def kl_gini(per_dim_kl: torch.Tensor) -> float:
    """Gini coefficient on per-dim KL averaged over batch.

    P-G-01 / Achille-Soatto diagnostic: a healthy IB concentrates information
    in a small sufficient subset → Gini near 1. Diffuse KL (all dims carrying
    a little) → Gini near 0 → bottleneck not actually compressing.
    per_dim_kl: (B, D) — non-negative per-dim KL.
    """
    if per_dim_kl.numel() == 0:
        return 0.0
    v = per_dim_kl.mean(dim=0).flatten().sort().values  # (D,) sorted ascending
    n = v.numel()
    if n < 2 or v.sum() <= 0:
        return 0.0
    # Gini = (2 Σ i*v_i / (n Σ v_i)) - (n+1)/n
    i = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    return ((2 * (i * v).sum() / (n * v.sum())) - (n + 1) / n).item()


def _seed_pretrain(seed: int, rank: int) -> None:
    """Seed all RNGs that affect model init, dropout, and data shuffles.

    Per-rank offset on torch.cuda RNGs only — model init and Python RNG share
    the same seed so DDP starts from identical weights. Without this, the
    rank-0 broadcast of teacher params is the only thing keeping ranks in sync.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def train():
    # 30-min collective timeout: protects against a slow first step (kernel JIT,
    # long-doc encode) tripping the default 10-min NCCL watchdog and killing the run.
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    device = accelerator.device
    _seed_pretrain(PRETRAIN_SEED, accelerator.process_index)

    if accelerator.is_main_process:
        print(
            f"[launch] world_size={accelerator.num_processes} "
            f"mixed_precision={accelerator.mixed_precision} "
            f"device={device}",
            flush=True,
        )
        os.makedirs("checkpoints", exist_ok=True)
        wandb.init(project=WANDB_PROJECT, name=CONDITION, config={
            "condition":           CONDITION,
            "pretrain_seed":        PRETRAIN_SEED,
            "epochs":               EPOCHS,
            "batch_size":           BATCH_SIZE,
            "grad_accum":           GRAD_ACCUM,
            "world_size":           accelerator.num_processes,
            "lr_encoder":           LR_ENCODER,
            "lr_ib_head":           LR_IB_HEAD,
            "lr_proj_pred":         LR_PROJ_PRED,
            "lr_log_s":             LR_LOG_S,
            "clip_encoder":         CLIP_ENCODER,
            "clip_ib_head":         CLIP_IB_HEAD,
            "clip_proj_pred":       CLIP_PROJ_PRED,
            "ema_tau_start":        EMA_TAU_START,
            "ema_tau_end":          EMA_TAU_END,
            "beta_kl_final":        BETA_KL_FINAL,
            "beta_kl_warmup_frac":  BETA_KL_WARMUP_FRAC,
            "free_bits_nats":       FREE_BITS_NATS,
            "var_weight":           VAR_WEIGHT,
            "cov_weight":           COV_WEIGHT,
            "var_gamma":            VAR_GAMMA,
            "z_proj_var_weight":    Z_PROJ_VAR_WEIGHT,
            "z_proj_cov_weight":    Z_PROJ_COV_WEIGHT,
            "mu_var_weight":        MU_VAR_WEIGHT,
            "mu_cov_weight":        MU_COV_WEIGHT,
            "temporal_buf_len":     TEMPORAL_BUF_LEN,
            "mlm_weight":           MLM_WEIGHT,
        })
    # Block non-main ranks until main has created checkpoints/ — avoids a race
    # at the first accelerator.save_state() call.
    accelerator.wait_for_everyone()

    dataset = load_ecthr()
    model   = GlocalIBModel(ema_tau=EMA_TAU_START, device=str(device))

    loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    total_steps  = (len(loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, total_steps // 10)

    # P-G-03: 4-group AdamW. Each parameter belongs to exactly one group.
    # Note: in V2, the encoder is RobertaForMaskedLM exposed at `encoder_full`;
    # body and LM head are both in the "encoder" group (they share embedding
    # weights so a uniform LR is appropriate).
    encoder_params:   list = []
    ib_head_params:   list = []
    proj_pred_params: list = []
    log_s_params:     list = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n == "log_s":
            log_s_params.append(p)
        elif n.startswith("mu_head") or n.startswith("log_sigma_head"):
            ib_head_params.append(p)
        elif n.startswith("projector") or n.startswith("predictor"):
            proj_pred_params.append(p)
        elif n.startswith("encoder_full") or n.startswith("attn_pool_student"):
            encoder_params.append(p)
        # encoder_teacher and attn_pool_teacher are requires_grad=False — skip
    if accelerator.is_main_process:
        print(
            f"[param groups] encoder={sum(p.numel() for p in encoder_params):,} "
            f"ib_head={sum(p.numel() for p in ib_head_params):,} "
            f"proj_pred={sum(p.numel() for p in proj_pred_params):,} "
            f"log_s={sum(p.numel() for p in log_s_params):,}",
            flush=True,
        )
    opt = AdamW([
        {"params": encoder_params,   "lr": LR_ENCODER},
        {"params": ib_head_params,   "lr": LR_IB_HEAD},
        {"params": proj_pred_params, "lr": LR_PROJ_PRED},
        {"params": log_s_params,     "lr": LR_LOG_S, "weight_decay": 0.0},
    ])
    # P-G-04: cosine LR schedule with warmup. Replaces V1's linear schedule.
    # All 4 param groups follow the same cosine curve (scaled by their own LR).
    scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    model, opt, loader, scheduler = accelerator.prepare(model, opt, loader, scheduler)

    # DDP only broadcasts trainable params at construction. The teacher attention
    # pool AND the teacher encoder are requires_grad=False (EMA-only), so their
    # init differs across ranks. Broadcast rank-0's teacher params so all ranks
    # start from an identical teacher. No-op on single GPU.
    if accelerator.num_processes > 1:
        unwrapped = accelerator.unwrap_model(model)
        for p in unwrapped.attn_pool_teacher.parameters():
            dist.broadcast(p.data, src=0)
        for p in unwrapped.encoder_teacher.parameters():
            dist.broadcast(p.data, src=0)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(
            f"[plan] total_steps={total_steps} warmup_steps={warmup_steps} "
            f"β_warmup_steps≈{int(total_steps * BETA_KL_WARMUP_FRAC)} "
            f"len(loader)={len(loader)}",
            flush=True,
        )

    global_step    = 0
    step_times     = deque(maxlen=50)     # rolling sec/optimizer-step
    z_proj_buffer  = deque(maxlen=Z_BUFFER_LEN)  # detached fp32 cpu Z_proj_pred for diagnostic collapse metric
    # Temporal anti-collapse buffers (GPU, detached). Gradient flows through current
    # sample only; the queue contributes to the variance/cov computation but no
    # gradient flows back through it.
    z_proj_pred_loss_buf = deque(maxlen=TEMPORAL_BUF_LEN)   # each entry: (B, 768) on device
    mu_loss_buf          = deque(maxlen=TEMPORAL_BUF_LEN)   # each entry: (B, 256) on device
    collapse_metric = 0.0
    eff_rank_z      = 0.0
    eff_rank_mu     = 0.0
    infonce_lb      = 0.0
    # Parallel rolling buffer for the InfoNCE diagnostic: we pair Z_proj (post-
    # bottleneck) with z_partial (pre-bottleneck) — both 768-D — and compute
    # I(Z_proj; z_partial) lower bound. If the IB is actually compressing,
    # this bound should DROP relative to a no-bottleneck baseline.
    z_partial_diag_buffer = deque(maxlen=Z_BUFFER_LEN)
    t_train_start  = time.time()

    for epoch in range(EPOCHS):
        model.train()

        loader_iter = loader
        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(
                loader,
                desc=f"epoch {epoch + 1}/{EPOCHS}",
                leave=False,
                dynamic_ncols=True,
            )
            loader_iter = pbar

        opt_step_start = time.time()

        for full_batch in loader_iter:
            # Build Xm: word/sentence mask each paragraph in each document
            masked_batch = [
                [mask_text(para) for para in doc]
                for doc in full_batch
            ]
            kept_indices_batch = [
                get_paragraph_mask(len(doc))
                for doc in full_batch
            ]

            beta_kl = beta_kl_schedule(global_step, total_steps)

            with accelerator.accumulate(model):
                out = model(full_batch, masked_batch, kept_indices_batch)
                # V2.1: forward returns 11-tuple. Loss code consumes 0–7;
                # 8–9 logging-only; 10 is l_mlm (computed inside the DDP-wrapped
                # forward so gradients all-reduce correctly).
                total, lc, ll, li, lg, lvar, lcov = glocal_ib_loss(
                    *out[:8],
                    beta_kl=beta_kl,
                    free_bits_nats=FREE_BITS_NATS,
                    var_weight=VAR_WEIGHT,
                    cov_weight=COV_WEIGHT,
                    var_gamma=VAR_GAMMA,
                )
                l_mlm = out[10]

                # Temporal anti-collapse on Z_proj_pred (out[1]) and mu (out[5]).
                # Concatenate the current with-gradient sample with detached past
                # samples; variance/cov gradient flows through current sample only,
                # pushing it AWAY from the queue mean / decorrelating its dims.
                # This addresses the failure mode where the IB+projector+predictor
                # chain maps varied encoder outputs to constant final reps.
                l_var_z = out[1].new_zeros(())
                l_cov_z = out[1].new_zeros(())
                l_var_m = out[5].new_zeros(())
                l_cov_m = out[5].new_zeros(())
                if len(z_proj_pred_loss_buf) >= TEMPORAL_MIN_FILL:
                    queue_z = torch.cat(list(z_proj_pred_loss_buf), dim=0)   # (K, 768)
                    z_all   = torch.cat([out[1], queue_z], dim=0)
                    l_var_z = variance_loss(z_all, gamma=VAR_GAMMA)
                    l_cov_z = covariance_loss(z_all)

                    queue_m = torch.cat(list(mu_loss_buf), dim=0)            # (K, 256)
                    mu_all  = torch.cat([out[5], queue_m], dim=0)
                    l_var_m = variance_loss(mu_all, gamma=VAR_GAMMA)
                    l_cov_m = covariance_loss(mu_all)

                # P-C-01: per-paragraph token MLM loss outside UW. l_mlm came
                # from out[10], computed *inside* the DDP-wrapped forward
                # (so its gradients all-reduce; the V2.0 unwrap_model path
                # bypassed the reducer on multi-GPU).
                total = total \
                      + Z_PROJ_VAR_WEIGHT * l_var_z \
                      + Z_PROJ_COV_WEIGHT * l_cov_z \
                      + MU_VAR_WEIGHT     * l_var_m \
                      + MU_COV_WEIGHT     * l_cov_m \
                      + MLM_WEIGHT        * l_mlm

                accelerator.backward(total)

                # Push current detached samples to the temporal buffers AFTER
                # backward (so the next step sees them). Use bf16 to save memory.
                z_proj_pred_loss_buf.append(out[1].detach().to(torch.bfloat16))
                mu_loss_buf.append(out[5].detach().to(torch.bfloat16))

                grad_norm = None
                grad_norm_ib_head   = 0.0
                grad_norm_proj_pred = 0.0
                if accelerator.sync_gradients:
                    # P-G-03: per-group clip contract. Tighter clip on the
                    # randomly initialized heads (IB, projector, predictor) so
                    # they cannot dominate the encoder's gradient direction.
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        encoder_params, max_norm=CLIP_ENCODER)
                    if ib_head_params:
                        grad_norm_ib_head = float(torch.nn.utils.clip_grad_norm_(
                            ib_head_params, max_norm=CLIP_IB_HEAD).item())
                    if proj_pred_params:
                        grad_norm_proj_pred = float(torch.nn.utils.clip_grad_norm_(
                            proj_pred_params, max_norm=CLIP_PROJ_PRED).item())

                opt.step()
                scheduler.step()
                opt.zero_grad()

                # EMA update after optimizer step — encoder + attention pool.
                # P-G-04: τ ramps with the schedule. Set the model's tau in-place
                # so update_teacher_ema picks it up.
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.ema_tau = ema_tau_schedule(global_step, total_steps)
                    unwrapped.update_teacher_ema()

            if accelerator.sync_gradients:
                global_step += 1
                step_dt = time.time() - opt_step_start
                opt_step_start = time.time()
                step_times.append(step_dt)
                avg_step = sum(step_times) / len(step_times)
                eta_sec  = avg_step * max(0, total_steps - global_step)

                # Maintain rolling buffers of post-predictor Z_proj (out[1]) and
                # pre-bottleneck z_partial (out[9]) for the InfoNCE I(Z;X) bound.
                z_proj_buffer.append(out[1].detach().float().cpu().mean(0))
                z_partial_diag_buffer.append(out[9].detach().float().cpu().mean(0))

                if accelerator.is_main_process:
                    log_s         = out[7].detach().float()
                    uw_weights    = torch.exp(-log_s)
                    mu_det        = out[5].detach().float()
                    log_sigma_det = out[6].detach().float()
                    per_dim_kl    = -0.5 * (
                        1 + 2 * log_sigma_det - mu_det.pow(2) - torch.exp(2 * log_sigma_det)
                    )
                    active_kl_dims = (per_dim_kl > FREE_BITS_NATS).float().mean().item()
                    kl_per_dim_max = per_dim_kl.mean(dim=0).max().item()  # heaviest dim
                    kl_gini_coeff  = kl_gini(per_dim_kl)
                    predictor_norm_mean = out[1].detach().float().norm(dim=-1).mean().item()
                    z_proj_norm_mean    = out[8].detach().float().norm(dim=-1).mean().item()

                    if global_step % COLLAPSE_LOG_EVERY == 0:
                        collapse_metric = mean_pairwise_cosine(z_proj_buffer)
                        # P-A-01: effective rank on Z and μ; InfoNCE I(Z_proj; z_partial)
                        # lower bound (both 768-D, paired from the same step).
                        # If the IB is compressing this bound should drop vs a
                        # no-bottleneck baseline.
                        if (len(z_proj_buffer) >= 4
                                and len(z_partial_diag_buffer) == len(z_proj_buffer)):
                            z_buf_tensor  = torch.stack(list(z_proj_buffer))           # (K, 768)
                            zp_buf_tensor = torch.stack(list(z_partial_diag_buffer))   # (K, 768)
                            eff_rank_z = effective_rank(z_buf_tensor)
                            _, infonce_lb = infonce_lower_bound(z_buf_tensor, zp_buf_tensor)
                        else:
                            eff_rank_z = 0.0
                            infonce_lb = 0.0
                        eff_rank_mu = effective_rank(mu_det)

                    if torch.cuda.is_available():
                        gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                        torch.cuda.reset_peak_memory_stats()
                    else:
                        gpu_mem_gb = 0.0

                    grad_norm_value = grad_norm.item() if grad_norm is not None else 0.0

                    if global_step % LOG_EVERY == 0:
                        wandb.log({
                            "total_loss":          total.item(),
                            "l_compress":          lc.item(),
                            "l_local":             ll.item(),
                            "l_inter":             li.item(),
                            "l_global":            lg.item(),
                            "l_variance":          lvar.item(),
                            "l_covariance":        lcov.item(),
                            "l_var_z_temporal":    l_var_z.item(),
                            "l_cov_z_temporal":    l_cov_z.item(),
                            "l_var_mu_temporal":   l_var_m.item(),
                            "l_cov_mu_temporal":   l_cov_m.item(),
                            "l_mlm":               l_mlm.item(),
                            "uw_contrib_local":    (ll * uw_weights[0]).item(),
                            "uw_contrib_inter":    (li * uw_weights[1]).item(),
                            "uw_contrib_global":   (lg * uw_weights[2]).item(),
                            "log_s_local":         log_s[0].item(),
                            "log_s_inter":         log_s[1].item(),
                            "log_s_global":        log_s[2].item(),
                            "uw_weight_local":     uw_weights[0].item(),
                            "uw_weight_inter":     uw_weights[1].item(),
                            "uw_weight_global":    uw_weights[2].item(),
                            "beta_kl_contrib":     (beta_kl * lc).item(),
                            "beta_kl":             beta_kl,
                            "mu_abs_mean":         mu_det.abs().mean().item(),
                            "log_sigma_mean":      log_sigma_det.mean().item(),
                            "kl_per_dim_mean":     per_dim_kl.mean().item(),
                            "kl_per_dim_max":      kl_per_dim_max,
                            "kl_gini":             kl_gini_coeff,
                            "kl_per_dim_hist":     wandb.Histogram(per_dim_kl.mean(dim=0).cpu().numpy()),
                            "active_kl_dims":      active_kl_dims,
                            "predictor_norm_mean": predictor_norm_mean,
                            "z_proj_norm_mean":    z_proj_norm_mean,
                            "collapse_metric_inter_doc_cos": collapse_metric,
                            "effective_rank_z_proj": eff_rank_z,
                            "effective_rank_mu":     eff_rank_mu,
                            "infonce_lb_zproj":      infonce_lb,
                            "grad_norm_encoder":   grad_norm_value,
                            "grad_norm_ib_head":   grad_norm_ib_head,
                            "grad_norm_proj_pred": grad_norm_proj_pred,
                            "lr_encoder":          scheduler.get_last_lr()[0],
                            "lr_ib_head":          scheduler.get_last_lr()[1],
                            "lr_proj_pred":        scheduler.get_last_lr()[2],
                            "lr_log_s":            scheduler.get_last_lr()[3],
                            "ema_tau":             accelerator.unwrap_model(model).ema_tau,
                            "sec_per_step":        avg_step,
                            "examples_per_sec":    (BATCH_SIZE * GRAD_ACCUM * accelerator.num_processes) / max(avg_step, 1e-9),
                            "gpu_mem_gb":          gpu_mem_gb,
                            "eta_min":             eta_sec / 60.0,
                            "elapsed_min":         (time.time() - t_train_start) / 60.0,
                            "epoch":               epoch,
                            "progress":            global_step / max(1, total_steps),
                            "step":                global_step,
                        })

                    if pbar is not None:
                        pbar.set_postfix({
                            "loss": f"{total.item():.3f}",
                            "l_c":  f"{lc.item():.1f}",
                            "l_g":  f"{lg.item():.3f}",
                            "l_vZ": f"{l_var_z.item():.3f}",
                            "l_vμ": f"{l_var_m.item():.3f}",
                            "β":    f"{beta_kl:.2f}",
                            "coll": f"{collapse_metric:.2f}",
                        })

                    if global_step % PRINT_EVERY == 0:
                        print(
                            f"[step {global_step}/{total_steps}] "
                            f"loss={total.item():.3f}  "
                            f"l_c={lc.item():.1f} l_l={ll.item():.4f} "
                            f"l_i={li.item():.4f} l_g={lg.item():.3f} "
                            f"l_v={lvar.item():.3f} l_cov={lcov.item():.3f} "
                            f"l_vZ={l_var_z.item():.3f} l_vμ={l_var_m.item():.3f}  "
                            f"log_s=[{log_s[0].item():.2f},{log_s[1].item():.2f},"
                            f"{log_s[2].item():.2f}]  "
                            f"β={beta_kl:.2f} act_kl={active_kl_dims:.2f} "
                            f"μ|·|={mu_det.abs().mean().item():.3f} "
                            f"coll={collapse_metric:.3f}  "
                            f"grad={grad_norm_value:.2f} mem={gpu_mem_gb:.1f}GB  "
                            f"sec/step={avg_step:.2f} eta={eta_sec/60:.1f}min",
                            flush=True,
                        )

                    if collapse_metric > COLLAPSE_ALARM_THR and global_step % COLLAPSE_LOG_EVERY == 0:
                        print(
                            f"[ALARM] collapse_metric_inter_doc_cos={collapse_metric:.3f} "
                            f"exceeds {COLLAPSE_ALARM_THR} at step {global_step}. "
                            f"Anti-collapse may be insufficient — consider raising "
                            f"VAR_WEIGHT or EMA_TAU.",
                            flush=True,
                        )

        if pbar is not None:
            pbar.close()

        accelerator.wait_for_everyone()
        # save_state is collective — must be called on all ranks
        accelerator.save_state(f"checkpoints/{CONDITION}_epoch{epoch}")
        if accelerator.is_main_process:
            elapsed_min = (time.time() - t_train_start) / 60.0
            print(f"Epoch {epoch} done.  elapsed={elapsed_min:.1f}min", flush=True)
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                f"checkpoints/{CONDITION}_epoch{epoch}.pt",
            )

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
