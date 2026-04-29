#!/bin/bash
#SBATCH --job-name=mlm_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/mlm_%j.out
#SBATCH --error=logs/mlm_%j.err

# --- HARDWARE SELECTION ---
# If using Spark:
##SBATCH --partition=spark
##SBATCH --gres=gpu:4

# If using RTX 4090:
#SBATCH --gres=gpu:1

# Load environment
source ~/.bashrc
# conda activate glocal_nlp

cd "$SLURM_SUBMIT_DIR"

# Execute MLM notebook
jupyter nbconvert --to notebook --execute notebooks/03_pretrain_mlm.ipynb \
    --output notebooks/03_pretrain_mlm_executed.ipynb \
    --ExecutePreprocessor.timeout=86400
