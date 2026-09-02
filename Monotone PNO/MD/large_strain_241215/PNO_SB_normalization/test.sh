#!/bin/bash
source ~/.bashrc
mkdir -p logs

solution_type='one_phase'
# test_dataset='index_cg-md-250212_0.1_0.5'
# test_dataset='index_cg-md-250212_0.1_0.3_0.01'
# test_dataset='test'
for layer_info in '64_4_256_4'; do
    for test_dataset in 'train' 'valid'; do
        nohup python3 -u PNO_SB.py --test --solution_type $solution_type --test_dataset $test_dataset --layer_info $layer_info > logs/test_${solution_type}_${test_dataset}_${layer_info}.log 2>&1 &
    done
done