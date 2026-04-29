#!/bin/bash
#SBATCH --job-name=glocal_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/glocal_%j.out
#SBATCH --error=logs/glocal_%j.err

# --- HARDWARE SELECTION ---
# If using Spark, uncomment the partition line:
##SBATCH --partition=spark
##SBATCH --gres=gpu:4

# If using RTX 4090, uncomment this (if partition is known, e.g., 'gpu'):
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load environment
source ~/.bashrc
# conda activate glocal_nlp

# Enter project directory (update this path!)
cd "$SLURM_SUBMIT_DIR"

# Execute notebook
# Note: Ensure BATCH_SIZE in the notebook matches your GPU choice.
# 4090 (24GB) -> BATCH_SIZE=4
# Spark (128GB) -> BATCH_SIZE=16
jupyter nbconvert --to notebook --execute notebooks/02_pretrain_glocal.ipynb \
    --output notebooks/02_pretrain_glocal_executed.ipynb \
    --ExecutePreprocessor.timeout=86400
