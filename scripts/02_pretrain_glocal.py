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
from collections import deque
from datetime import timedelta

import torch
import torch.distributed as dist
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator, InitProcessGroupKwargs

# Suppress benign per-paragraph warnings (long tokens are sub-chunked manually,
# use_cache is explicitly disabled alongside gradient checkpointing).
transformers.logging.set_verbosity_error()

sys.path.append(".")
from src.data import load_ecthr, mask_text, get_paragraph_mask
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS              = 5
BATCH_SIZE          = 1       # per GPU; effective = BATCH_SIZE × num_GPUs × GRAD_ACCUM
GRAD_ACCUM          = 8
LR                  = 1e-5
LR_LOG_S            = 1e-2    # ~1000× encoder LR — required for UW to actually converge
MAX_GRAD_NORM       = 1.0
EMA_TAU             = 0.99
BETA_KL_FINAL       = 1.0
BETA_KL_WARMUP_FRAC = 0.25    # ramp β over first 25% of total optimizer steps
FREE_BITS_NATS      = 0.5     # per-dim KL floor — prevents posterior collapse to N(0,1)
LOG_EVERY           = 1       # W&B log frequency (optimizer steps)
PRINT_EVERY         = 50      # stdout summary frequency (optimizer steps)
WANDB_PROJECT       = "glocal-nlp"
CONDITION           = "glocal_ib"
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return [item["text"] for item in batch]


def beta_kl_schedule(step: int, total_steps: int) -> float:
    warmup = max(1, int(total_steps * BETA_KL_WARMUP_FRAC))
    return BETA_KL_FINAL * min(1.0, step / warmup)


def train():
    # 30-min collective timeout: protects against a slow first step (kernel JIT,
    # long-doc encode) tripping the default 10-min NCCL watchdog and killing the run.
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=30))],
    )
    device = accelerator.device

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
            "epochs":               EPOCHS,
            "batch_size":           BATCH_SIZE,
            "grad_accum":           GRAD_ACCUM,
            "world_size":           accelerator.num_processes,
            "lr":                   LR,
            "lr_log_s":             LR_LOG_S,
            "ema_tau":              EMA_TAU,
            "beta_kl_final":        BETA_KL_FINAL,
            "beta_kl_warmup_frac":  BETA_KL_WARMUP_FRAC,
            "free_bits_nats":       FREE_BITS_NATS,
        })
    # Block non-main ranks until main has created checkpoints/ — avoids a race
    # at the first accelerator.save_state() call.
    accelerator.wait_for_everyone()

    dataset = load_ecthr()
    model   = GlocalIBModel(ema_tau=EMA_TAU, device=str(device))

    loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    total_steps  = (len(loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = max(1, total_steps // 10)

    # Two-group AdamW: log_s needs ~1000× the encoder LR to converge, and weight
    # decay would pull it toward 0 — fighting its natural drift toward log(L_i).
    log_s_params = [p for n, p in model.named_parameters() if n == "log_s"]
    other_params = [p for n, p in model.named_parameters() if n != "log_s"]
    opt = AdamW([
        {"params": other_params, "lr": LR},
        {"params": log_s_params, "lr": LR_LOG_S, "weight_decay": 0.0},
    ])
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    model, opt, loader, scheduler = accelerator.prepare(model, opt, loader, scheduler)

    # DDP only broadcasts trainable params at construction. The teacher attention
    # pool has requires_grad=False (EMA-only), so its random init differs across
    # ranks. Explicitly broadcast rank-0's teacher params so all ranks start
    # from an identical teacher — otherwise step-0 L_local/L_global rep targets
    # are rank-dependent until EMA converges. No-op on single GPU.
    if accelerator.num_processes > 1:
        unwrapped = accelerator.unwrap_model(model)
        for p in unwrapped.attn_pool_teacher.parameters():
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

        opt_step_start = time.time()   # measures the full GRAD_ACCUM window per optimizer step

        for full_batch in loader_iter:
            # Build Xm: word/sentence mask each paragraph in each document
            masked_batch = [
                [mask_text(para) for para in doc]
                for doc in full_batch
            ]
            # Paragraph dropout indices (applied at pooling stage, not encoding)
            kept_indices_batch = [
                get_paragraph_mask(len(doc))
                for doc in full_batch
            ]

            beta_kl = beta_kl_schedule(global_step, total_steps)

            with accelerator.accumulate(model):
                out = model(full_batch, masked_batch, kept_indices_batch)
                total, lc, ll, li, lg = glocal_ib_loss(
                    *out, beta_kl=beta_kl, free_bits_nats=FREE_BITS_NATS,
                )
                accelerator.backward(total)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                opt.step()
                scheduler.step()
                opt.zero_grad()

                # EMA update after optimizer step — attention pooling only
                if accelerator.sync_gradients:
                    accelerator.unwrap_model(model).update_teacher_ema()

            if accelerator.sync_gradients:
                global_step += 1
                step_dt = time.time() - opt_step_start
                opt_step_start = time.time()
                step_times.append(step_dt)
                avg_step = sum(step_times) / len(step_times)
                eta_sec  = avg_step * max(0, total_steps - global_step)

                if accelerator.is_main_process:
                    log_s         = out[7].detach().float()
                    uw_weights    = torch.exp(-log_s)
                    mu_det        = out[5].detach().float()
                    log_sigma_det = out[6].detach().float()
                    per_dim_kl    = -0.5 * (
                        1 + 2 * log_sigma_det - mu_det.pow(2) - torch.exp(2 * log_sigma_det)
                    )
                    active_kl_dims = (per_dim_kl > FREE_BITS_NATS).float().mean().item()

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
                            "uw_contrib_compress": (lc * uw_weights[0]).item(),
                            "uw_contrib_local":    (ll * uw_weights[1]).item(),
                            "uw_contrib_inter":    (li * uw_weights[2]).item(),
                            "uw_contrib_global":   (lg * uw_weights[3]).item(),
                            "log_s_compress":      log_s[0].item(),
                            "log_s_local":         log_s[1].item(),
                            "log_s_inter":         log_s[2].item(),
                            "log_s_global":        log_s[3].item(),
                            "uw_weight_compress":  uw_weights[0].item(),
                            "uw_weight_local":     uw_weights[1].item(),
                            "uw_weight_inter":     uw_weights[2].item(),
                            "uw_weight_global":    uw_weights[3].item(),
                            "beta_kl":             beta_kl,
                            "mu_abs_mean":         mu_det.abs().mean().item(),
                            "log_sigma_mean":      log_sigma_det.mean().item(),
                            "kl_per_dim_mean":     per_dim_kl.mean().item(),
                            "active_kl_dims":      active_kl_dims,
                            "grad_norm":           grad_norm_value,
                            "lr":                  scheduler.get_last_lr()[0],
                            "lr_log_s":            scheduler.get_last_lr()[1],
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
                            "β":    f"{beta_kl:.2f}",
                            "act":  f"{active_kl_dims:.2f}",
                        })

                    if global_step % PRINT_EVERY == 0:
                        print(
                            f"[step {global_step}/{total_steps}] "
                            f"loss={total.item():.3f}  "
                            f"l_c={lc.item():.1f} l_l={ll.item():.4f} "
                            f"l_i={li.item():.4f} l_g={lg.item():.3f}  "
                            f"log_s=[{log_s[0].item():.2f},{log_s[1].item():.2f},"
                            f"{log_s[2].item():.2f},{log_s[3].item():.2f}]  "
                            f"β={beta_kl:.2f} act_kl={active_kl_dims:.2f} "
                            f"μ|·|={mu_det.abs().mean().item():.3f}  "
                            f"grad={grad_norm_value:.2f} mem={gpu_mem_gb:.1f}GB  "
                            f"sec/step={avg_step:.2f} eta={eta_sec/60:.1f}min",
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
