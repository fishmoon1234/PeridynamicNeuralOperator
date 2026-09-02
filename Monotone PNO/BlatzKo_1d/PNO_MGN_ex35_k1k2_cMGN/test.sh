
#!/bin/bash
source ~/.bashrc
mkdir -p logs
# layer_info='300_2_300_2'
layer_info='128_4_128_4'
nohup python3 -u PNO_BB.py --test --layer_info ${layer_info} > logs/test_${layer_info}.log 2>&1 &