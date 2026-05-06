import torch
import wandb
import os
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
import sys

sys.path.append(".")
from src.data import load_ecthr, mask_paragraphs, truncate_paragraphs
from src.model import GlocalIBModel
from src.loss import glocal_ib_loss

# --- CONFIG ---
EPOCHS         = 5
BATCH_SIZE     = 1      
GRAD_ACCUM     = 4
LR             = 1e-5
WARMUP_STEPS   = 200
BETA           = 1e-3  # Fixed IB trade-off
MAX_GRAD_NORM  = 1.0
WANDB_PROJECT  = "glocal-nlp"
CONDITION      = "glocal_ib"
DISABLE_IB     = (CONDITION == "glocal_beta0")


class GlocalDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_text = self.dataset[idx]["text"]
        # 1. Truncate and get indices
        full_paras, full_indices = truncate_paragraphs(raw_text, max_chunks=50)
        # 2. Mask paragraphs (relative to the truncated list)
        masked_paras, kept_indices = mask_paragraphs(full_paras)
        
        return {
            "full_batch_data": (full_paras, full_indices),
            "masked_batch": masked_paras,
            "indices_batch": kept_indices
        }


def collate_fn(batch):
    return {
        "full_batch_data": [item["full_batch_data"] for item in batch],
        "masked_batch": [item["masked_batch"] for item in batch],
        "indices_batch": [item["indices_batch"] for item in batch],
    }


def train():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT, name=f"{CONDITION}_final_fix", config={
            "condition": CONDITION, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "lr": LR, "beta": BETA
        })

    dataset = load_ecthr()
    train_ds = GlocalDataset(dataset["train"])
    
    model   = GlocalIBModel(device=str(device))
    opt     = AdamW(model.parameters(), lr=LR)

    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=2 # FIX: Use workers to avoid data starvation
    )

    # Scheduler
    num_training_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )

    model, opt, loader, scheduler = accelerator.prepare(model, opt, loader, scheduler)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(loader):
            with accelerator.accumulate(model):
                out = model(
                    batch["full_batch_data"], 
                    batch["masked_batch"], 
                    batch["indices_batch"]
                )
                total, lc, ll, li, lg = glocal_ib_loss(*out, beta=BETA, disable_ib=DISABLE_IB)
                accelerator.backward(total)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                opt.step()
                scheduler.step()
                if accelerator.sync_gradients:
                    accelerator.unwrap_model(model).update_teacher()
                opt.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % 10 == 0 and accelerator.is_main_process:
                    log_s = out[8].detach().float()
                    wandb.log({
                        "total_loss": total.item(),
                        "l_compress": lc.item(),
                        "l_local":    ll.item(),
                        "l_inter":    li.item(),
                        "l_global":   lg.item(),
                        "log_s_0_local":    log_s[0].item(),
                        "log_s_1_inter":    log_s[1].item(),
                        "log_s_2_global":   log_s[2].item(),
                        "lr": scheduler.get_last_lr()[0],
                        "epoch": epoch, "step": global_step,
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

if __name__ == "__main__":
    train()
