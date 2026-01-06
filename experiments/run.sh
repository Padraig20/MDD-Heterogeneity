#!/bin/bash

learning_rates=(5e-4 5e-5 5e-6)
batch_sizes=(128 256)
loss_functions=("seq2cells" "composite")
hidden_dims=(264 512 1024)

epochs=50
X="./X"
y="./y.csv"

for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for lf in "${loss_functions[@]}"; do
            for hd in "${hidden_dims[@]}"; do
                echo "Submitting job with lr=$lr, bs=$bs, lf=$lf, hd=$hd, epochs=$epochs"
                sbatch ./experiments/submit.sh --learning-rate "$lr" --batch-size "$bs" --hidden-dim "$hd" --epochs "$epochs" --loss-function "$lf" -X "$X" -y "$y"
            done
        done
    done
done

