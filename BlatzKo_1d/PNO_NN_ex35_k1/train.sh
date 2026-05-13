
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# act_xi='Sigmoid'
for layer_info in '20_4' '128_5'; do
for act_xi in 'GeLU'; do
for lr in 0.99; do
    nohup python3 -u PNO_BB.py --act_xi ${act_xi} --lr ${lr} --layer_info ${layer_info} > logs/train_${layer_info}_${act_xi}_${lr}.log 2>&1 &
done
done
done