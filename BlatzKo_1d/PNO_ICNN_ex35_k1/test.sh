
#!/bin/bash
source ~/.bashrc
mkdir -p logs


for layer_info in '128_5'; do
    for act_xi in 'Softplus'; do
        for seed in 42 43 45; do
            for lr in 0.995; do
                nohup python3 -u PNO_BB.py --test --lr ${lr} --layer_info ${layer_info} --act_xi ${act_xi} --seed ${seed} > logs/test${layer_info}_${act_xi}_${lr}_${seed}.log 2>&1 &
                sleep 2
            done
        done
    done
done