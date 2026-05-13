
#!/bin/bash
source ~/.bashrc
mkdir -p logs
# layer_info='300_2_300_2'
# layer_info='128_4_220_4'
for beta in 10; do
    for layer_info in 64_4_256_4; do
        nohup python3 -u PNO_BB.py --layer_info ${layer_info} --beta ${beta} > logs/output_${layer_info}_${beta}.log 2>&1 &
        sleep 2
    done
done