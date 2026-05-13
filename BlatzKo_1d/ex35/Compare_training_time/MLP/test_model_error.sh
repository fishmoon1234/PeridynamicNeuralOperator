
#!/bin/bash
source ~/.bashrc
mkdir -p logs

layer_info='20_4'
act_xi='Softplus'

# layer_info='128_5'
# act_xi='ReLU'

for seed in 42; do
    nohup python3 -u plot_MGN_MLP_ICNN.py --act_xi ${act_xi} --layer_info ${layer_info} > logs/test_model_error_${layer_info}_${act_xi}.log 2>&1 &
    sleep 1
done