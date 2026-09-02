
#!/bin/bash
source ~/.bashrc
mkdir -p logs

layer_info='128_4_256_4'
nohup python3 -u PNO_BB.py --layer_info $layer_info > logs/train_$layer_info.log 2>&1 &