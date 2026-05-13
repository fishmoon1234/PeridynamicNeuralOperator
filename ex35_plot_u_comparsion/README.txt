================================================================================
ex35_plot_u_comparsion - Solution Comparison Tool
================================================================================

This tool compares solutions from three neural network models (MLP, MGN, ICNN)
for solving PDE problems.

================================================================================
MAIN SCRIPTS
================================================================================

There are three main scripts in MLP/ directory:

1. plot_u_MGN_MLP_ICNN.py
   - Direct plotting script for u solutions
   - Loads saved solutions from plot_u_MGN_MLP_ICNN_save.py
   - No model loading or computation needed, just plotting
   - Run: python3 plot_u_MGN_MLP_ICNN.py 
   
2. plot_u_MGN_MLP_ICNN_save.py
   - Compares solutions (u) from three models
   - Solves PDE problems and saves solutions
   - Generates comparison plots of u solutions
   - Run: python3 plot_u_MGN_MLP_ICNN_save.py

3. plot_MGN_MLP_ICNN.py
   - Visualizes the learned function g(λ) from three models
   - Compares g(λ) predictions vs true g(λ)
   - Run: python3 plot_MGN_MLP_ICNN.py

================================================================================
QUICK START
================================================================================

1. Navigate to MLP directory:
   cd MLP

2. Run the script:
   python3 plot_u_MGN_MLP_ICNN_save.py

That's it! The script will:
- Load three pre-trained models
- Solve PDE problems
- Save solutions for fast reloading (on subsequent runs)
- Generate comparison plots

================================================================================
REQUIREMENTS
================================================================================

Python packages: torch, numpy, scipy, matplotlib, torch_geometric
Hardware: GPU recommended (CUDA), but CPU works too

================================================================================
DIRECTORY STRUCTURE
================================================================================

ex35_plot_u_comparsion/
├── README.txt                          # This file
├── DATA/                               # Input data (.mat file)
├── MLP/
│   ├── plot_u_MGN_MLP_ICNN_save.py    # Compare u solutions (solves & saves)
│   ├── plot_u_MGN_MLP_ICNN.py         # Plot u solutions (loads saved data)
│   ├── plot_MGN_MLP_ICNN.py           # Plot g(λ) function (direct plotting)
│   ├── Results/                        # Pre-trained model checkpoints
│   └── saved_solutions/                # Saved solutions (auto-generated)
├── MGN/Results/                        # MGN model checkpoints
└── ICNN/Results/                       # ICNN model checkpoints

================================================================================
OUTPUT
================================================================================

1. Saved solutions: MLP/saved_solutions/ex35/linear_sol/
   - Solutions are saved as .npy files for fast reloading
   - Delete these files if you want to recompute

2. Plot images:
   - MLP/ex35_linear_sol_u_NN_MGN_ICNN_{j}.png
     * Comparison plots showing true u solution vs three model solutions
   - MLP/ex35_g_NN_MGN_ICNN_128_5.png
     * Comparison plots showing true g(λ) vs learned g(λ) from three models

================================================================================
IMPORTANT NOTES
================================================================================

- First run: Computes solutions (may take several minutes)
- Subsequent runs: Loads saved solutions (very fast)
- Ensure model checkpoints exist in Results/ directories
- Ensure data file exists in DATA/ directory

================================================================================
