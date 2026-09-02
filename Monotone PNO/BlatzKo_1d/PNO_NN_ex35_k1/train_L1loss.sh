
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# act_xi='Sigmoid'
for layer_info in '128_5'; do
    for act_xi in 'Softplus'; do
        for lr in 0.995; do
            for seed in 43; do
                nohup python3 -u PNO_BB_L1loss.py --act_xi ${act_xi} --lr ${lr} --layer_info ${layer_info} --seed ${seed} > logs/train_L1loss_${layer_info}_${act_xi}_${lr}_${seed}.log 2>&1 &
                sleep 2
            done
        done
    done
done