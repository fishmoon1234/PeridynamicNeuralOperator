
#!/bin/bash
source ~/.bashrc
mkdir -p logs

nohup python3 -u PNO_BB.py --test > logs/test.log 2>&1 &