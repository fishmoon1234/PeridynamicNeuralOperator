
#!/bin/bash
source ~/.bashrc
mkdir -p logs
# layer_info='300_2_300_2'
layer_info='128_4_128_4'
test_type='ID'
nohup python3 -u PNO_SB.py --test --layer_info ${layer_info} --test_type ${test_type} > logs/test_${layer_info}_${test_type}.log 2>&1 &