import numpy as np
import os
import matplotlib.pyplot as plt
from utilities_INO_PD import plot_u_NN_MGN_ICNN

current_dir = os.path.dirname(os.path.realpath(__file__))
N = 4
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)

ex = 'ex35'
initial_type = 'linear_sol'

# Directory where solutions are saved
save_dir = os.path.join(current_dir, 'saved_solutions', ex, initial_type)

# Specify which samples to plot
plot_index = [1]

for i in range(3, N):
    gap = int(h[i]/h0)
    
    print(f"Plotting samples: {plot_index} for gap {gap}")
    
    # Load and plot each saved solution
    for j in plot_index:
        save_file_base = os.path.join(save_dir, f'sample_{j}_gap_{gap}')
        save_file_NN = f'{save_file_base}_NN.npy'
        save_file_MGN = f'{save_file_base}_MGN.npy'
        save_file_ICNN = f'{save_file_base}_ICNN.npy'
        save_file_true = f'{save_file_base}_true.npy'
        save_file_X = f'{save_file_base}_X.npy'
        
        # Check if all files exist
        if not all(os.path.exists(f) for f in [save_file_NN, save_file_MGN, save_file_ICNN, save_file_true, save_file_X]):
            print(f"Warning: Missing files for sample {j}, gap {gap}. Skipping...")
            continue
        
        # Load saved solutions
        print(f"Loading saved solutions for sample {j}, gap {gap}...")
        uh_NN = np.load(save_file_NN)
        uh_MGN = np.load(save_file_MGN)
        uh_ICNN = np.load(save_file_ICNN)
        u_true = np.load(save_file_true)
        data_X = np.load(save_file_X)
        
        # Convert to torch tensors for plotting function (if needed)
        import torch
        uh_NN = torch.from_numpy(uh_NN).double()
        uh_MGN = torch.from_numpy(uh_MGN).double()
        uh_ICNN = torch.from_numpy(uh_ICNN).double()
        u_true = torch.from_numpy(u_true).double()
        data_X = torch.from_numpy(data_X).double()
        
        # Plot
        print(f"Plotting sample {j}...")
        plot_u_NN_MGN_ICNN(data_X, u_true, uh_NN, uh_MGN, uh_ICNN, j, current_dir, '%s_%s' % (ex, initial_type))
        print(f"Plot saved for sample {j}")

print("Done!")
