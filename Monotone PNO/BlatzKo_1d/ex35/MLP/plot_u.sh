
#!/bin/bash
source ~/.bashrc
mkdir -p logs

# initial_type='zero'
initial_type='linear_sol'

nohup python3 -u plot_u_MGN_MLP_ICNN.py --initial_type ${initial_type} > logs/plot_u_MGN_MLP_ICNN_${initial_type}.log 2>&1 &