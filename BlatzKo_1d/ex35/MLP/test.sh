
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# act_xi='Sigmoid'
for layer_info in '128_5' '20_4'; do
    for act_xi in 'ReLU'; do
        for lr in 0.99; do
            for seed in 42; do
                nohup python3 -u PNO_BB.py --test --act_xi ${act_xi} --lr ${lr} --layer_info ${layer_info} --seed ${seed} > logs/test_${layer_info}_${act_xi}_${lr}_${seed}.log 2>&1 &
                sleep 2
            done
        done
    done
done