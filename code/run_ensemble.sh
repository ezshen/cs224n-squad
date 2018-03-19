#!/bin/bash
source_path=$1
target_path=$2

declare -a experiments=(
                "bidaf_charcnn_train"
                "bidaf_charcnn_train1"
                "bidaf_charcnn_train2"
                # "bidaf_charcnn_train3"
                # "bidaf_charcnn_train4"
                # "bidaf_charcnn_train5"
                # "bidaf_charcnn_train6"
                "bidaf_charcnn_train_singlelayer1"
                )

eargs=""
for exp in "${experiments[@]}"; do
    eval_path="../experiments/$exp/predictions_span.json"
    eargs="$eargs $eval_path"
    echo "running official_eval for $exp..."
    python main.py --experiment_name=$exp --mode=official_eval --json_in_path=$source_path --ckpt_load_dir=../experiments/$exp/best_checkpoint
done
wait

# Ensemble
python ensemble.py $eargs --json_in_path=$source_path --output_file=$target_path
./run_ensemble.sh ~/cs224n-squad/tiny-dev.json ~/cs224n-squad/ensemble_predictions.json