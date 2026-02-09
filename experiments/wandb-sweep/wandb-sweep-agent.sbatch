#!/bin/bash
#SBATCH -p bsse-gpu
#SBATCH --gres=gpu:GP104:1
#SBATCH -c 15
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --nice=10000
#SBATCH --job-name=wandb-sweep
#SBATCH --output=experiments/logs/slurm-%A_%a.out

set -euo pipefail

SWEEP_ID="$1"

# 1 run (agent) per Slurm task
wandb agent --count 1 "$SWEEP_ID"
