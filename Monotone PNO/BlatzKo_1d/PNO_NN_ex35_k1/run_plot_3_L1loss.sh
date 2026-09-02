
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# for initial_type in linear_sol zero; do
# for layer_info in 20_4 128_5; do
#     nohup python3 -u plot_NN_MGN_ICNN_L1loss.py --layer_info ${layer_info} --initial_type ${initial_type} > logs/plot_${layer_info}_${initial_type}_L1loss.log 2>&1 &
# done
# done


nohup python3 -u plot_u_NN_MGN_ICNN_L1loss.py > logs/plot_u_NN_MGN_ICNN_L1loss.log 2>&1 &