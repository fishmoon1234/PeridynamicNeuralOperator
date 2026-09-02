
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# act_xi='Sigmoid'
for layer_info in '128_5' '20_4'; do
    for act_xi in 'Softplus' 'ReLU' 'Sigmoid'; do
        for lr in 0.99 0.995; do
            for seed in 43; do
                nohup python3 -u PNO_BB_L1loss.py --test --act_xi ${act_xi} --lr ${lr} --layer_info ${layer_info} --seed ${seed} > logs/test_L1loss_${layer_info}_${act_xi}_${lr}_${seed}.log 2>&1 &
                sleep 2
            done
        done
    done
done