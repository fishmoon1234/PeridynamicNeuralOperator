
#!/bin/bash
source ~/.bashrc
mkdir -p logs
# layer_info='300_2_300_2'
# layer_info='128_4_220_ 4'
for layer_info in '128_4_220_4' '64_4_256_4'; do
    nohup python3 -u PNO_SB.py --layer_info ${layer_info} > logs/output_${layer_info}.log 2>&1 &
done