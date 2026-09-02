#!/bin/bash
#SBATCH --job-name=train_2D_BK
#SBATCH --partition=iirmas-gpu
#SBATCH --qos=iirmas-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1           # 1 CPU core per task
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err

source ~/.bashrc
PY="/home/u2025010169/.conda/envs/pytorch1.10.0/bin/python"
mkdir -p logs
ex='ex4'
layer_info='32_4'
$PY -u PNO_BB.py > logs/trian_${ex}.log 2>&1