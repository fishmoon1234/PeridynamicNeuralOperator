#!/bin/bash
source ~/.bashrc
mkdir -p logs

solution_type='one_phase'

nohup python3 -u PNO_BB_test_disk.py --test > logs/test_disk.log 2>&1 &