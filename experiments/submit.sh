#!/bin/bash

#SBATCH -p bsse-gpu
#SBATCH --gres=gpu:GP104:1
#SBATCH -c 15
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --nice=10000
#SBATCH --job-name=mdd-heterogeneity
#SBATCH --output=experiments/logs/slurm-%j.out

# ./slurm/submit.sh --learning-rate "$lr" --batch-size "$bs" --hidden-dim "$hd" --epochs "$epochs" --loss-function "$lf" -X "$X" -y "$y"

echo ""
echo "==============================================================================="
echo ""
echo "LR=$2, BS=$4, HD=$6, Epochs=$8, Loss=${10}, X=${12}, y=${14}"
echo ""
echo "==============================================================================="
echo ""

uv run python ./training/train.py \
    -lr "$2"    \
    -b  "$4"    \
    -hd "$6"    \
    -e  "$8"    \
    -l  "${10}" \
    -u          \
    -v          \
    -X "${12}"  \
    -y "${14}"  \
    --run-name "lr${2}_b${4}_hd${6}_l${10}"