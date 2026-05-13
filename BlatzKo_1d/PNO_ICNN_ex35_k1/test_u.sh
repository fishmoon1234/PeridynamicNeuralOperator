#!/bin/bash
source ~/.bashrc
mkdir -p logs


ex_data='ex35_ood'
for initial_type in linear_sol; do
for layer_info in 128_5 20_4; do
    nohup python3 -u PNO_BB_test_u.py --layer_info ${layer_info} --initial_type ${initial_type} --ex_data ${ex_data} > logs/test_u_${layer_info}_${initial_type}_${ex_data}_Softplus_seed_42.log 2>&1 &
done
done