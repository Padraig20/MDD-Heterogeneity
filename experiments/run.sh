#!/bin/bash

# ./slurm/submit.sh --learning-rate "$lr" --batch-size "$bs" --n-layers "$nl" 
#                   --epochs "$epochs" -X "$X" -y "$y" --cossim-lambda "$cossim_lambda"
#                   --mpc-lambda "$mpc_lambda" --mse-lambda "$mse_lambda" --pnll-lambda "$pnll_lambda"

learning_rates=(5e-6)
batch_sizes=(256)
n_layers=(1 3 5)

cossim_lambdas=(0 0.2 1.0)
mpc_lambdas=(0 0.2 1.0)
mse_lambdas=(0 0.2 1.0)
pnll_lambdas=(0 0.2 1.0)

epochs=50
X="./X"
y="./y.csv"

for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for nl in "${n_layers[@]}"; do
            for cossim_lambda in "${cossim_lambdas[@]}"; do
                for mpc_lambda in "${mpc_lambdas[@]}"; do
                    for mse_lambda in "${mse_lambdas[@]}"; do
                        for pnll_lambda in "${pnll_lambdas[@]}"; do
                            echo "Submitting job with lr=$lr, bs=$bs, nl=$nl, epochs=$epochs, cossim_lambda=$cossim_lambda, mpc_lambda=$mpc_lambda, mse_lambda=$mse_lambda, pnll_lambda=$pnll_lambda"
                            sbatch ./experiments/submit.sh --learning-rate "$lr" --batch-size "$bs" --n-layers "$nl" --epochs "$epochs" -X "$X" -y "$y" --cossim-lambda "$cossim_lambda" --mpc-lambda "$mpc_lambda" --mse-lambda "$mse_lambda" --pnll-lambda "$pnll_lambda"
                        done
                    done
                done
            done
        done
    done
done
