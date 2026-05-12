import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # silence unauth-request warning
# Titan Xp has no NVLink — PCIe-only P2P deadlocks the DDP param broadcast at startup.
# Force NCCL onto shared-memory/socket transports; harmless on NVLink hardware.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
# Surface NCCL hangs as readable errors instead of silent 10-min timeouts.
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
EPOCHS        = 5
BATCH_SIZE    = 1       # per GPU; effective = BATCH_SIZE × num_GPUs × GRAD_ACCUM
GRAD_ACCUM    = 8
LR            = 1e-5
MAX_GRAD_NORM = 1.0
EMA_TAU       = 0.99
WANDB_PROJECT = "glocal-nlp"
CONDITION     = "glocal_ib"
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return [item["text"] for item in batch]


def train():
    # 30-min collective timeout: protects against a slow first step (kernel JIT,
    # long-doc encode) tripping the default 10-min NCCL watchdog and killing the run.
    accelerator = Accelerator(
        mixed_precision="fp16",
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
            "condition":   CONDITION,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "grad_accum":  GRAD_ACCUM,
            "world_size":  accelerator.num_processes,
            "lr":          LR,
            "ema_tau":     EMA_TAU,
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
    opt          = AdamW(model.parameters(), lr=LR)
    scheduler    = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    model, opt, loader, scheduler = accelerator.prepare(model, opt, loader, scheduler)

    # DDP only broadcasts trainable params at construction. The teacher attention
    # pool has requires_grad=False (EMA-only), so its random init differs across
    # ranks. Explicitly broadcast rank-0's teacher params so all ranks start
    # from an identical teacher — otherwise step-0 L_local/L_global rep targets
    # are rank-dependent until EMA converges.
    if accelerator.num_processes > 1:
        unwrapped = accelerator.unwrap_model(model)
        for p in unwrapped.attn_pool_teacher.parameters():
            dist.broadcast(p.data, src=0)
        accelerator.wait_for_everyone()

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for full_batch in loader:
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

            with accelerator.accumulate(model):
                out = model(full_batch, masked_batch, kept_indices_batch)
                total, lc, ll, li, lg = glocal_ib_loss(*out)
                accelerator.backward(total)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                opt.step()
                scheduler.step()
                opt.zero_grad()

                # EMA update after optimizer step — attention pooling only
                if accelerator.sync_gradients:
                    accelerator.unwrap_model(model).update_teacher_ema()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 1 == 0 and accelerator.is_main_process:
                    log_s = out[7].detach().float()
                    weights = torch.exp(-log_s)   # UW per-loss multiplier
                    wandb.log({
                        "total_loss":     total.item(),
                        "l_compress":     lc.item(),
                        "l_local":        ll.item(),
                        "l_inter":        li.item(),
                        "l_global":       lg.item(),
                        # UW contributions — detect degeneracy if any stay flat near raw loss
                        "uw_contrib_compress": (lc * weights[0]).item(),
                        "uw_contrib_local":    (ll * weights[1]).item(),
                        "uw_contrib_inter":    (li * weights[2]).item(),
                        "uw_contrib_global":   (lg * weights[3]).item(),
                        "log_s_compress": log_s[0].item(),
                        "log_s_local":    log_s[1].item(),
                        "log_s_inter":    log_s[2].item(),
                        "log_s_global":   log_s[3].item(),
                        "lr":             scheduler.get_last_lr()[0],
                        "epoch":          epoch,
                        "progress":       global_step / max(1, total_steps),
                        "step":           global_step,
                    })

        accelerator.wait_for_everyone()
        # save_state is collective — must be called on all ranks
        accelerator.save_state(f"checkpoints/{CONDITION}_epoch{epoch}")
        if accelerator.is_main_process:
            print(f"Epoch {epoch} done.")
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                f"checkpoints/{CONDITION}_epoch{epoch}.pt",
            )

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
