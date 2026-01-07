#!/bin/bash

#SBATCH -p bsse-gpu
#SBATCH --gres=gpu:GP104:1
#SBATCH -c 15
#SBATCH --mem=50G
#SBATCH --time=7-00:00:00
#SBATCH --nice=10000
#SBATCH --job-name=mdd-heterogeneity
#SBATCH --output=experiments/logs/slurm-%j.out

# ./slurm/submit.sh --learning-rate "$lr" --batch-size "$bs" --n-layers "$nl" 
#                   --epochs "$epochs" -X "$X" -y "$y" --cossim-lambda "$cossim_lambda"
#                   --mpc-lambda "$mpc_lambda" --mse-lambda "$mse_lambda" --pnll-lambda "$pnll_lambda"

echo ""
echo "==============================================================================="
echo ""
echo "LR=$2, BS=$4, NL=$6, Epochs=$8, Loss=${10}, CosSim=${12}, MPC=${14}, MSE=${16}, PNLL=${18}, X=${20}, y=${22}"
echo ""
echo "==============================================================================="
echo ""

uv run python ./training/train.py \
    -lr "$2"    \
    -b  "$4"    \
    -nl "$6"    \
    -e  "$8"    \
    -es         \
    -v          \
    -X "${20}"  \
    -y "${22}"  \
    --cossim-lambda "${12}" \
    --mpc-lambda "${14}"      \
    --mse-lambda "${16}"      \
    --pnll-lambda "${18}"     \
    --run-name "lr${2}_b${4}_nl${6}_cs${12}_mpc${14}_mse${16}_pnll${18}"