
#!/bin/bash
source ~/.bashrc
mkdir -p logs

layer_info='20_4'
act_xi='Softplus'
for seed in 42; do
    for lr in 0.995; do
        nohup python3 -u PNO_BB.py --lr ${lr} --layer_info ${layer_info} --act_xi ${act_xi} --seed ${seed} > logs/train_${layer_info}_${act_xi}_${lr}_${seed}.log 2>&1 &
        sleep 3
    done
done