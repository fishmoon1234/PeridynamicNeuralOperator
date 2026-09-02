
#!/bin/bash
source ~/.bashrc
mkdir -p logs

nohup python3 -u PNO_BB.py > logs/train.log 2>&1 &