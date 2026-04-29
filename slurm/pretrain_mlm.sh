#!/bin/bash
#SBATCH --job-name=mlm_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=spark
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/mlm_%j.out
#SBATCH --error=logs/mlm_%j.err

source ~/.bashrc
conda activate glocal_nlp
cd /path/to/GlocalDoc               # ← update before submitting

accelerate launch --num_processes=4 train_mlm.py
