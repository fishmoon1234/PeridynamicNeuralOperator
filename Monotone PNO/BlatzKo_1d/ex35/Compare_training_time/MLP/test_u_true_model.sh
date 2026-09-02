
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# layer_info='20_4'
# act_xi='Softplus'
# lr=0.99

layer_info='128_5'
act_xi='ReLU'
lr=0.99
initial_type='linear_sol'

for seed in 42; do
    nohup python3 -u PNO_BB_test_u_true_model.py --act_xi ${act_xi} --lr ${lr} --layer_info ${layer_info} --seed ${seed} --initial_type ${initial_type} > logs/test_u_${layer_info}_${act_xi}_${lr}_${seed}_${initial_type}_true_model_truncation_0.5.log 2>&1 &
    sleep 1
done