
#!/bin/bash
source ~/.bashrc
mkdir -p logs

batch_size=10
for beta in 10 50; do
    for layer_info in 16_16_16_16; do
        for lrs in 0.001; do
            nohup python3 -u PNO_BB.py --lrs ${lrs} --layer_info ${layer_info} --beta ${beta} --batch_size ${batch_size} > logs/train_${lrs}_${layer_info}_${beta}_${batch_size}.log 2>&1 &
        done
        sleep 5
    done
done