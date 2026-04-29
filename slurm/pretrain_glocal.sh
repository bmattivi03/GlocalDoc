#!/bin/bash
#SBATCH --job-name=glocal_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=spark           # ← confirm partition name with lab admin
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/glocal_%j.out
#SBATCH --error=logs/glocal_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc               # ← update before submitting

# Edit CONDITION in train_glocal.py before each run
accelerate launch --num_processes=4 train_glocal.py
