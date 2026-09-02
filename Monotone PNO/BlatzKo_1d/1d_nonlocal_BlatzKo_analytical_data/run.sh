
#!/bin/bash
source ~/.bashrc
mkdir -p logs

nohup python3 -u BlatzKo_1d_analytical_u_ex35_ood.py > logs/output_ex35_ood.log 2>&1 &