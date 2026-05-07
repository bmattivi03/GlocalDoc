#!/bin/bash
#SBATCH --job-name=h_mlm_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=spark
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/h_mlm_%j.out
#SBATCH --error=logs/h_mlm_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc               # ← update before submitting

accelerate launch --num_processes=4 scripts/03_pretrain_mlm.py
