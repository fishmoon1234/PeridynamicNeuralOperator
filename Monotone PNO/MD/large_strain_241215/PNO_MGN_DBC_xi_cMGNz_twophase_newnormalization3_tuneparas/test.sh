#!/bin/bash
source ~/.bashrc
mkdir -p logs

solution_type='one_phase'
# test_dataset='index_cg-md-250212_0.1_0.5'
# test_dataset='index_cg-md-250212_0.1_0.3_0.01'
# test_dataset='index_cg-md-250212_0.1_0.15'
# test_dataset='index_cg-md-250212_0.1_0.15_0.01'
# test_dataset='test'
batch_size=2
lrs=0.001
# for layer_info in 16_16_256_4 16_16_128_4 64_8_256_4; do
#     for test_dataset in 'index_cg-md-250212_0.1_0.15_0.01'; do
#         nohup python3 -u PNO_BB.py --test --solution_type $solution_type --test_dataset $test_dataset --batch_size ${batch_size} --lrs ${lrs} --layer_info ${layer_info} > logs/test_${solution_type}_${test_dataset}_${batch_size}_${lrs}_${layer_info}.log 2>&1 &
#     sleep 2
#     done
# done

for layer_info in 16_16_256_4; do
    for test_dataset in 'valid'; do
        nohup python3 -u PNO_BB.py --test --solution_type $solution_type --test_dataset $test_dataset --batch_size ${batch_size} --lrs ${lrs} --layer_info ${layer_info} > logs/test_${solution_type}_${test_dataset}_${batch_size}_${lrs}_${layer_info}.log 2>&1 &
        sleep 2
    done
done