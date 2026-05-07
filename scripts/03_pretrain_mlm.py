import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling
import sys

sys.path.append(".")
from src.data import load_ecthr

# --- CONFIG ---
EPOCHS     = 3
BATCH_SIZE = 4
LR         = 5e-5


def train():
    accelerator = Accelerator(log_with="wandb")
    if accelerator.is_main_process:
        import wandb
        wandb.init(project="glocal-nlp", name="mlm_baseline_fixed")

    tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    model     = RobertaForMaskedLM.from_pretrained("distilroberta-base")

    dataset = load_ecthr()

    def tokenize_function(examples):
        # FIX: Process the data EXACTLY like GlocalIB (paragraph-wise then join)
        # However, MLM traditionally works on flattened text. 
        # To be fair, we must ensure it only sees the SAME 50 paragraphs.
        processed_texts = []
        for doc_paragraphs in examples["text"]:
            # load_ecthr already filters to ≤50 paragraphs — no truncation needed
            processed_texts.append(" ".join(doc_paragraphs))
            
        return tokenizer(processed_texts, truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "labels"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    train_loader = DataLoader(
        tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=data_collator
    )
    optimizer = AdamW(model.parameters(), lr=LR)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()

            if i % 50 == 0 and accelerator.is_main_process:
                import wandb
                wandb.log({"mlm_loss": loss.item(), "epoch": epoch})

    if accelerator.is_main_process:
        model.save_pretrained("checkpoints/mlm_baseline")
        print("MLM baseline saved → checkpoints/mlm_baseline/")


if __name__ == "__main__":
    train()
