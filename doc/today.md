# Plan for Today: First Runnable Version

Our goal is to leverage the AI Lab GPU resources by having a stable, runnable pre-training pipeline by the end of the day. We are not aiming for a final result, but for a "smoke-tested" execution on the cluster.

## 1. Environment & Data Verification (Task 5)
Before launching heavy jobs, we must ensure the environment is correctly set up and the dataset is accessible from the cluster nodes.
- **Action:** Implement `notebooks/01_data_exploration.ipynb`.
- **Validation:** Run the notebook top-to-bottom to verify:
    - HuggingFace `datasets` can download `coastalcph/lex_glue`.
    - `src/data.py` correctly filters and masks paragraphs.
    - Statistics (paragraph counts, label distribution) match our expectations.

## 2. GlocalIB Pre-training Scaffolding (Tasks 6 & 8)
This is the core "runnable" component. We need the notebook that orchestrates the training and the SLURM script to submit it.
- **Action:** 
    - Implement `notebooks/02_pretrain_glocal.ipynb` as defined in `PLAN.md`.
    - Implement `slurm/pretrain_glocal.sh` with the correct partition (`spark` for 128GB or a standard one for 32GB).
- **Validation:** 
    - Run a "Short-Circuit" test: Modify the notebook to run for only 2 steps and 1 epoch on a small subset of data.
    - Confirm that `wandb` logs are appearing and a checkpoint `.pt` file is created in `checkpoints/`.

## 3. MLM Baseline Setup (Task 7)
Since we want to compare our results, having the MLM baseline ready is a close second priority.
- **Action:** Implement `notebooks/03_pretrain_mlm.ipynb`.
- **Validation:** Run a short-circuit test (2 steps) via the HuggingFace `Trainer`.

## 4. Proposed Execution Order for Today
1. **Branching:** We are already on `first-training`.
2. **Setup:** Run `pip install -r requirements.txt` and verify imports.
3. **Task 5 (Exploration):** Create and run the exploration notebook.
4. **Task 6 (GlocalIB):** Create the pre-training notebook.
5. **Task 8 (SLURM):** Create the submission scripts.
6. **Task 11 (Smoke Test):** Submit a 5-minute test job to the cluster to verify GPU utilization and logging.

## Why this scope?
This plan avoids the complexity of fine-tuning (Task 9) and evaluation (Task 10) for now, focusing entirely on the **Pre-training phase**, which is the most compute-intensive part and where we need the AI Lab resources most. Once we have a model training on the cluster, we can develop the fine-tuning logic while the pre-training runs in the background.
