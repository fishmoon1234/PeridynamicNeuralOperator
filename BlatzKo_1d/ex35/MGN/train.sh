
#!/bin/bash
source ~/.bashrc
mkdir -p logs
layer_info='128_5'
lr=0.995

seed=42
act_xi='Sigmoid'
nohup python3 -u PNO_BB.py --layer_info ${layer_info} --act_xi ${act_xi} --seed ${seed} --lr ${lr} > logs/train_${layer_info}_${act_xi}_${lr}_${seed}.log 2>&1 &