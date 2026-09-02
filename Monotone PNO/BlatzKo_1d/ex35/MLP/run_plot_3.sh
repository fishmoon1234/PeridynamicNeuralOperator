
#!/bin/bash
source ~/.bashrc
mkdir -p logs


for act_xi in ReLU; do
for layer_info in 128_5; do
    nohup python3 -u plot_NN_MGN_ICNN.py --layer_info ${layer_info} --act_xi ${act_xi} > logs/plot_${layer_info}_${act_xi}_L1loss.log 2>&1 &
done
done