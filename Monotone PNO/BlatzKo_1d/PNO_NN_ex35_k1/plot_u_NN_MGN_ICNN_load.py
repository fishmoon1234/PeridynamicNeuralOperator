import torch
import numpy as np
import os
from utilities_INO_PD import plot_u_NN_MGN_ICNN

# Set up paths
current_dir = os.path.dirname(os.path.realpath(__file__))

# Configuration
layer_info = '128_5'
initial_type = 'linear_sol'
plot_index = 2
h_str = 'h8'  # h=2^(-8)

# Load saved data
base_dir = 'Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.995, 0.998]_gap_1_ReLU_L1loss_seed_43'
base_dir = os.path.join(current_dir, base_dir)
load_path = os.path.join(base_dir, f'solved_u_plot_index_{plot_index}_{h_str}.npz')

print(f"Loading data from: {load_path}")

# Check if file exists
if not os.path.exists(load_path):
    print(f"Error: File {load_path} does not exist!")
    print("Please run plot_u_NN_MGN_ICNN_L1loss_save.py first to generate the data.")
    exit(1)

# Load the data
data = np.load(load_path)

# Extract data
data_X = data['data_X']  # Shape: (S, 1)
u_true = data['u_true']  # Shape: (S, 1)
uh_NN = data['uh_NN']    # Shape: (S, 1)
uh_MGN = data['uh_MGN']  # Shape: (S, 1)
uh_ICNN = data['uh_ICNN']  # Shape: (S, 1)

# Convert to 1D arrays as required by plot function
x = data_X.squeeze()  # Shape: (S,)
u_true_1d = u_true.squeeze()  # Shape: (S,)
uh_NN_1d = uh_NN.squeeze()    # Shape: (S,)
uh_MGN_1d = uh_MGN.squeeze()  # Shape: (S,)
uh_ICNN_1d = uh_ICNN.squeeze()  # Shape: (S,)

# Print loaded information
print(f"\nLoaded data information:")
# Handle string data from npz
loaded_layer_info = str(data['layer_info']) if isinstance(data['layer_info'], np.ndarray) else data['layer_info']
loaded_initial_type = str(data['initial_type']) if isinstance(data['initial_type'], np.ndarray) else data['initial_type']
print(f"  Layer info: {loaded_layer_info}")
print(f"  Initial type: {loaded_initial_type}")
print(f"  Plot index: {int(data['plot_index'])}")
print(f"  h value: {float(data['h_i'])}")
print(f"  Data shape - x: {x.shape}, u_true: {u_true_1d.shape}")
print(f"  Data shape - uh_NN: {uh_NN_1d.shape}, uh_MGN: {uh_MGN_1d.shape}, uh_ICNN: {uh_ICNN_1d.shape}")

# Create plot name
plot_name = f'{loaded_layer_info}_{loaded_initial_type}'

# Plot using the loaded data
print(f"\nGenerating plot...")
plot_u_NN_MGN_ICNN(x, u_true_1d, uh_NN_1d, uh_MGN_1d, uh_ICNN_1d, 
                    plot_index, current_dir, plot_name)

print(f"Plot saved successfully!")

