# GlocalIB — Project TODO

> Tracks implementation progress. Full details for each task are in `PLAN.md`.

---

## Progress Overview

| Task | Description | Status |
|------|-------------|--------|
| 1 | Repository structure + requirements | ✅ Done |
| 2 | `src/data.py` — data pipeline | ✅ Done |
| 3 | `src/loss.py` — loss functions | ✅ Done |
| 4 | `src/model.py` — model architecture | ✅ Done |
| 5 | `notebooks/01_data_exploration.ipynb` | ⬜ Todo |
| 6 | `notebooks/02_pretrain_glocal.ipynb` | ⬜ Todo |
| 7 | `notebooks/03_pretrain_mlm.ipynb` | ⬜ Todo |
| 8 | `slurm/` — SLURM job scripts | ⬜ Todo |
| 9 | `notebooks/04_finetune.ipynb` | ⬜ Todo |
| 10 | `notebooks/05_evaluate.ipynb` | ⬜ Todo |

---

## ✅ Completed

### Task 1 — Repository Structure
- Directories: `src/`, `notebooks/`, `slurm/`, `checkpoints/`, `results/`, `logs/`
- `requirements.txt` updated (torch, transformers, datasets, scikit-learn, wandb, jupyter, nbconvert, accelerate)
- `.gitignore` added (excludes checkpoints, results, wandb artifacts, executed notebooks)
- `CLAUDE.md` and `PLAN.md` added

### Task 2 — `src/data.py`
- `load_ecthr()` — loads ECtHR from HuggingFace, filters docs < 5 paragraphs
- `mask_paragraphs()` — drops 20–40% of paragraphs randomly (student branch input)
- `sample_few_shot()` — multi-label aware few-shot sampler with seed control

### Task 3 — `src/loss.py`
- `alignment_loss()` — 1 minus cosine similarity between student projection and teacher target (both 768-dim)
- `compression_loss()` — closed-form KL(N(mu, sigma²) || N(0,1))
- `glocal_ib_loss()` — total loss = l_align + beta × l_compress; `disable_ib=True` zeros the KL for the β=0 ablation

### Task 4 — `src/model.py`
- `GlocalIBModel` — teacher branch (stop-grad, full doc → 768-dim) + student branch (masked doc → probabilistic head mu/sigma 256-dim → reparameterization → MLP projector 256→512→768) + learnable `log_beta` clamped to min 0.01
- `DocumentClassifier` — takes a pre-trained encoder, chunk-and-pool encoding, `Linear(768, 10)` head, sigmoid output for multi-label classification

---

## ⬜ Todo

### Task 5 — `notebooks/01_data_exploration.ipynb`
- Replace `import.ipynb`
- Cells: dataset loading, paragraph length stats + histogram, label distribution, masking sanity check
- Run on lab machine to verify

### Task 6 — `notebooks/02_pretrain_glocal.ipynb`
- Config cell: `CONDITION = "glocal_ib"` or `"glocal_beta0"`, `BATCH_SIZE=4` (32GB GPU) or `16` (Spark)
- W&B logging: loss, l_align, l_compress, beta, epoch
- Saves checkpoint every epoch to `checkpoints/{CONDITION}_epoch{N}.pt`
- Run twice: once for `glocal_ib`, once for `glocal_beta0`

### Task 7 — `notebooks/03_pretrain_mlm.ipynb`
- HuggingFace Trainer + DataCollatorForLanguageModeling (15% masking)
- Flattens ECtHR paragraphs into single text field per document
- Saves to `checkpoints/mlm/mlm_final`

### Task 8 — `slurm/` job scripts
- `pretrain_glocal.sh` — Spark node, `--gres=gpu:4`, 128GB VRAM
- `pretrain_mlm.sh` — Spark node, `--gres=gpu:4`, 128GB VRAM
- `finetune.sh` — standard 32GB GPU, `--gres=gpu:1`
- ⚠️ Confirm Spark partition name with lab admin before submitting

### Task 9 — `notebooks/04_finetune.ipynb`
- Loads encoder from each of the 3 checkpoints (glocal_ib, glocal_beta0, mlm)
- Runs `train_and_eval()` for N = {10, 50, 100} × 5 seeds
- Saves all results to `results/finetuning_results.json`

### Task 10 — `notebooks/05_evaluate.ipynb`
- Macro-F1 table: mean ± std per condition per N
- Performance vs N curve (log-scale x-axis, error bars)
- Beta trajectory plot fetched from W&B
- Ablation delta table: GlocalIB full vs β=0 at each N

---

## Verification Checklist (run on lab machine)

- [ ] `python -c "from src.data import load_ecthr, mask_paragraphs, sample_few_shot; print('OK')"` passes
- [ ] `python -c "from src.model import GlocalIBModel, DocumentClassifier; print('OK')"` passes
- [ ] `python -c "from src.loss import glocal_ib_loss; print('OK')"` passes
- [ ] GlocalIB forward + backward pass — loss is finite, no NaN
- [ ] Beta starts near 1.0, stays ≥ 0.01 throughout pre-training
- [ ] MLM Trainer completes at least 2 steps without error
- [ ] `finetuning_results.json` has all 3 conditions × 3 label counts × 5 seeds
- [ ] Performance curve PNG generates without error
- [ ] GlocalIB full macro-F1 at N=10 > MLM baseline (primary hypothesis)
- [ ] GlocalIB full macro-F1 at N=10 > GlocalIB β=0 (ablation confirms IB term)

---

## Compute Notes

| Job | Machine | SLURM flags |
|-----|---------|-------------|
| GlocalIB pre-training | Spark 128GB | `--partition=spark --gres=gpu:4` |
| MLM pre-training | Spark 128GB | `--partition=spark --gres=gpu:4` |
| Fine-tuning | Standard 32GB | `--gres=gpu:1` |
| Data exploration / sanity checks | Standard 32GB | `--gres=gpu:1` |
