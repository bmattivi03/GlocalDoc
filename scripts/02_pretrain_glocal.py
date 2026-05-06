import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
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
MAX_GRAD_NORM  = 1.0
WANDB_PROJECT  = "glocal-nlp"
CONDITION      = "glocal_ib"
DISABLE_IB     = (CONDITION == "glocal_beta0")


def collate_fn(batch):
    return [item["text"] for item in batch]


def train():
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT, name=f"{CONDITION}_momentum_fixed", config={
            "condition": CONDITION, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "lr": LR,
        })

    dataset = load_ecthr()
    model   = GlocalIBModel(device=str(device))
    opt     = AdamW(model.parameters(), lr=LR)

    loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    model, opt, loader = accelerator.prepare(model, opt, loader)

    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, full_batch in enumerate(loader):
            full_batch = [truncate_paragraphs(text, max_chunks=50) for text in full_batch]

            masked_data    = [mask_paragraphs(text) for text in full_batch]
            masked_batch   = [m[0] for m in masked_data]
            indices_batch  = [m[1] for m in masked_data]

            with accelerator.accumulate(model):
                out = model(full_batch, masked_batch, indices_batch)
                total, lc, ll, li, lg = glocal_ib_loss(*out, disable_ib=DISABLE_IB)
                accelerator.backward(total)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                
                opt.step()
                # FIX: Essential EMA update for the momentum teacher
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
                        "log_s_0_compress": log_s[0].item(),
                        "log_s_1_local":    log_s[1].item(),
                        "log_s_2_inter":    log_s[2].item(),
                        "log_s_3_global":   log_s[3].item(),
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
