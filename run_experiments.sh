#!/usr/bin/bash
# This script runs multiple instances of qa.py with different alpha values.

# Define common parameters
llm_ranker="True"
pre_trained="False"
dataset="webqsp"
calibrate_with_rog="True"

# Run the processes with different alpha values
for alpha in 0.5 0.4 0.3; do 
    echo "Starting qa.py with alpha=${alpha}"
    nohup python3 qa.py --alpha ${alpha} --llm_ranker ${llm_ranker} --pre_trained ${pre_trained} --dataset_name ${dataset} --calibrate_rog ${calibrate_with_rog} > output_${alpha}.log 2>&1
done

echo "All processes started."
