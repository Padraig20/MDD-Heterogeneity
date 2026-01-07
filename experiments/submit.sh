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
echo "LR=$2, BS=$4, NL=$6, Epochs=$8, X=${10}, y=${12}, CosSim=${14}, MPC=${16}, MSE=${18}, PNLL=${20}"
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
    -X "${10}"  \
    -y "${12}"  \
    --cossim-lambda "${14}" \
    --mpc-lambda "${16}"      \
    --mse-lambda "${18}"      \
    --pnll-lambda "${20}"     \
    --run-name "lr${2}_b${4}_nl${6}_cs${14}_mpc${16}_mse${18}_pnll${20}"