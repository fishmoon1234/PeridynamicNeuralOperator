
#!/bin/bash
source ~/.bashrc
mkdir -p logs
ex_data='ex35'

# layer_info='32_4'
for layer_info in 128_5; do
nohup python3 -u PNO_BB_test_u.py --layer_info ${layer_info} --ex_data ${ex_data} > logs/test_u_${layer_info}_${ex_data}_retest.log 2>&1 &
done