import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset
import sys

sys.path.append(".")

# --- CONFIG ---
EPOCHS = 3
BATCH_SIZE = 4
LR = 5e-5

def train():
    accelerator = Accelerator(log_with="wandb")
    if accelerator.is_main_process:
        import wandb
        wandb.init(project="glocal-nlp", name="mlm_baseline_ddp")

    tokenizer = RobertaTokenizerFast.from_pretrained("distilroberta-base")
    model = RobertaForMaskedLM.from_pretrained("distilroberta-base")
    
    dataset = load_dataset("coastalcph/lex_glue", "ecthr_a")
    
    def tokenize_function(examples):
        # Flatten paragraphs into a single text block for standard MLM
        texts = [" ".join(doc) for doc in examples["text"]]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text", "labels"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=data_collator)
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
                wandb.log({"mlm_loss": loss.item(), "epoch": epoch})

    if accelerator.is_main_process:
        model.save_pretrained("checkpoints/mlm_baseline")

if __name__ == "__main__":
    train()
