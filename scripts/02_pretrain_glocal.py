import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
import sys

sys.path.append(".")
from src.data import load_ecthr, mask_text, get_paragraph_mask
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS        = 5
BATCH_SIZE    = 1       # per GPU; effective = BATCH_SIZE × num_GPUs × GRAD_ACCUM
GRAD_ACCUM    = 4
LR            = 1e-5
MAX_GRAD_NORM = 1.0
EMA_TAU       = 0.99
WANDB_PROJECT = "glocal-nlp"
CONDITION     = "glocal_ib"
# ─────────────────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return [item["text"] for item in batch]


def train():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT, name=CONDITION, config={
            "condition":   CONDITION,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "grad_accum":  GRAD_ACCUM,
            "lr":          LR,
            "ema_tau":     EMA_TAU,
        })

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
                if global_step % 10 == 0 and accelerator.is_main_process:
                    log_s = out[7].detach().float()
                    wandb.log({
                        "total_loss":     total.item(),
                        "l_compress":     lc.item(),
                        "l_local":        ll.item(),
                        "l_inter":        li.item(),
                        "l_global":       lg.item(),
                        "log_s_compress": log_s[0].item(),
                        "log_s_local":    log_s[1].item(),
                        "log_s_inter":    log_s[2].item(),
                        "log_s_global":   log_s[3].item(),
                        "lr":             scheduler.get_last_lr()[0],
                        "epoch":          epoch,
                        "step":           global_step,
                    })

        if accelerator.is_main_process:
            print(f"Epoch {epoch} done.")
            accelerator.save_state(f"checkpoints/{CONDITION}_epoch{epoch}")
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                f"checkpoints/{CONDITION}_epoch{epoch}.pt",
            )

    if accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
