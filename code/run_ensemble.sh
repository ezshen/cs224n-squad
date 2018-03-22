#!/bin/bash
source_file=$1
target_file=$2
rerun=$3

declare -a experiments=(
                "bidaf_charcnn_train1"
                "bidaf_charcnn_train2"
                "bidaf_charcnn_train3"
                "bidaf_charcnn_train4"
                "bidaf_charcnn_train5"
                "bidaf_charcnn_train6"
                "bidaf_charcnn_train_singlelayer1"
                "bidaf_charcnn_train_singlelayer2"
                "bidaf_charcnn_train_singlelayer3"
                "bidaf_charcnn_train_xavier"
                )

eargs=""
for exp in "${experiments[@]}"; do
    eval_path="experiments/$exp/predictions_span.json"
    eargs="$eargs $eval_path"
    if [ ! $rerun ]
    then
        echo "running official_eval for $exp..."
        python code/main.py --experiment_name=$exp --mode=official_eval --json_in_path=data/$source_file --ckpt_load_dir=experiments/$exp/best_checkpoint
    fi
done
wait

# Ensemble
python ensemble.py $eargs --json_in_path=data/$source_file --output_file=$target_file