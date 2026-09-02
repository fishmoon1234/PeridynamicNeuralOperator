
#!/bin/bash
source ~/.bashrc
mkdir -p logs

for act_xi in Sigmoid ReLU; do
for layer_info in 20_4 128_5; do
    nohup python3 -u plot_NN_MGN_ICNN.py --layer_info ${layer_info} > logs/plot_${layer_info}.log 2>&1 &
done
done