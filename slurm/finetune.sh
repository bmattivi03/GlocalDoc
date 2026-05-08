#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc               # ← update before submitting
mkdir -p logs

python scripts/04_finetune.py
