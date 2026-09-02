
#!/bin/bash
source ~/.bashrc
mkdir -p logs

layer_info='20_4'
act_xi='Softplus'
lr=0.99

# layer_info='128_5'
# act_xi='ReLU'
# lr=0.99

ex_data='ex35'

for seed in 42; do
    nohup python3 -u PNO_BB_test_u.py --act_xi ${act_xi} --lr ${lr} --layer_info ${layer_info} --seed ${seed} --ex_data ${ex_data} > logs/test_u_${layer_info}_${act_xi}_${lr}_${seed}_${ex_data}.log 2>&1 &
    sleep 1
done