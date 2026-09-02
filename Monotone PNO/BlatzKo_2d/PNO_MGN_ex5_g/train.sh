
#!/bin/bash
source ~/.bashrc
mkdir -p logs
layer_info='64_4'
gamma='0.5'
wd='1e-4'
lr='1e-3'
nohup python3 -u PNO_BB.py --layer_info ${layer_info} --gamma ${gamma} --wd ${wd} --lr ${lr} > logs/train_${layer_info}_gamma${gamma}_wd${wd}_lr${lr}.log 2>&1 &