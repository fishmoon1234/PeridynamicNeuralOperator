
#!/bin/bash
source ~/.bashrc
mkdir -p logs

lr=0.99
# initial_type='linear_sol'
ex_data='ex35'
act_xi='ReLU'
for initial_type in linear_sol; do
for layer_info in 128_5 20_4; do
    nohup python3 -u PNO_BB_test_u.py --layer_info ${layer_info} --lr ${lr} --initial_type ${initial_type} --ex_data ${ex_data} --act_xi ${act_xi} > logs/test_u_${layer_info}_${lr}_${initial_type}_${ex_data}_${act_xi}_L1loss.log 2>&1 &
done
done