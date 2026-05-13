#!/bin/bash
source ~/.bashrc
mkdir -p logs

# layer_info='20_4'
# lr=0.99

layer_info='128_5'
lr=0.995

act_xi='Sigmoid'
for ex_data in ex35; do
    for initial_type in linear_sol; do
        nohup python3 -u PNO_BB_test_u.py --layer_info ${layer_info} --initial_type ${initial_type} --ex_data ${ex_data} --lr ${lr} --act_xi ${act_xi} > logs/test_u_${layer_info}_${initial_type}_${ex_data}.log 2>&1 &
    done
done