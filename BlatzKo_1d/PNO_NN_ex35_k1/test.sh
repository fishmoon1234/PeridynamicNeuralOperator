
#!/bin/bash
source ~/.bashrc
mkdir -p logs

layer_info='14_2'
nohup python3 -u PNO_BB.py --test --layer_info $layer_info > logs/test_$layer_info.log 2>&1 &