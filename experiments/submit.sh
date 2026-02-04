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
#                   --norm-inputs "$norm_inputs" --norm_targets "$norm_targets" --norm-layer "$norm_layer"

echo ""
echo "==============================================================================="
echo ""
echo "LR=$2, BS=$4, NL=$6, Epochs=$8, X=${10}, y=${12}, CosSim=${14}, MPC=${16}, MSE=${18}, PNLL=${20}", NormInputs=${22}, NormTargets=${24}, "NormLayer=${26}"
echo ""
echo "==============================================================================="
echo ""

NORM_INPUTS_ARG=""
NORM_TARGETS_ARG=""
NORM_LAYER_ARG=""

if [[ "${22}" == "true" ]]; then
    NORM_INPUTS_ARG="--norm-inputs"
fi

if [[ "${24}" == "true" ]]; then
    NORM_TARGETS_ARG="--norm-targets"
fi

if [[ "${26}" == "true" ]]; then
    NORM_LAYER_ARG="--norm-layer"
fi

uv run python ./training/train.py \
    -lr "$2" \
    -b  "$4" \
    -nl "$6" \
    -e  "$8" \
    -es \
    -v \
    -X "${10}" \
    -y "${12}" \
    --seed 777 \
    --cossim-lambda "${14}" \
    --mpc-lambda "${16}" \
    --mse-lambda "${18}" \
    --pnll-lambda "${20}" \
    $NORM_INPUTS_ARG \
    $NORM_TARGETS_ARG \
    $NORM_LAYER_ARG \
    --run-name "lr${2}_b${4}_nl${6}_cs${14}_mpc${16}_mse${18}_pnll${20}_nc${22}_nt${24}_nl${26}"