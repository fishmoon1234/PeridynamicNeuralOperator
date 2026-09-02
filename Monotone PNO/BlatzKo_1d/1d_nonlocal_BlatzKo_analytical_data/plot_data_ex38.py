import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from timeit import default_timer
import os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import bisect
from utilities_INO_PD import *

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = f'{current_dir}/BlatzKo_data_1d/'
# ex = 'ex15'
ex = 'ex38'
DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
# DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)

# ndata = 400
ntrain = 300
nvalid = 50
ntest = 50

# Nx = 129 
# h0 = 2**(-7)
Nx = 257  
h0 = 2**(-8)
delta = 0.25
dx = 2**(-8)    # change
gap = int(dx/h0)
m_fact = int(delta/dx)
s = int((Nx-1)/gap)+1
S = s+2*m_fact
print(f'>> Training Mesh resolution: {s}x{s}')

reader = MatReader(DATA)
data_x = reader.read_field('coords')[:,::gap].reshape(S,)
data_x = data_x[m_fact:s+m_fact]
data_u = reader.read_field('displacement')[:,::gap].reshape(-1, S)
data_f = reader.read_field('bodyforce')[:,::gap].reshape(-1, S)
data_u = data_u[:, m_fact:s+m_fact]
data_f = data_f[:, m_fact:s+m_fact]


# old version
# plt.rcParams.update({'font.size': 15})
# fig, axs = plt.subplots(2, 4, figsize=(14, 6))
# n = 0
# titles = [[r'Ex-I: $u^{(1)}$', r'Ex-I: $u^{(2)}$', r'Ex-II: $u^{(1)}$', r'Ex-II: $u^{(2)}$'], 
#           [r'Ex-I: $b^{(1)}$', r'Ex-I: $b^{(2)}$', r'Ex-II: $b^{(1)}$', r'Ex-II: $b^{(2)}$']]
# for col in range(4):
#     axs[0, col].plot(data_x, data_u[col+n,:], color='orange')
#     axs[0, col].set_title(titles[0][col])
    
#     axs[1, col].plot(data_x, data_f[col+n,:], color='forestgreen')
#     axs[1, col].set_title(titles[1][col])
    
# # fig.suptitle(f'data: n={n}:{n+3}', fontsize=16)
# plt.tight_layout()
# plt.savefig('%s/%s_data.png' % (current_dir, ex), format='png')
# plt.close()


# plt.rcParams.update({'font.size': 15})
# fig, axs = plt.subplots(1, 4, figsize=(14, 3))
# n = 0
# titles = [r'(Ex-I) $u$', r'(Ex-I) $b$', r'(Ex-II) $u$', r'(Ex-II) $b$']
# for col in range(4):
#     if col == 0:  # Ex-I: u1
#         axs[col].plot(data_x, data_u[0+n,:], color='orange')
#     elif col == 1:  # Ex-I: b1
#         axs[col].plot(data_x, data_f[0+n,:], color='forestgreen')
#     elif col == 2:  # Ex-II: u1
#         axs[col].plot(data_x, data_u[2+n,:], color='orange')
#     elif col == 3:  # Ex-II: b1
#         axs[col].plot(data_x, data_f[2+n,:], color='forestgreen')
    
#     axs[col].set_title(titles[col])
    
# plt.tight_layout()
# plt.savefig('%s/%s_data.png' % (current_dir, ex), format='png', dpi=300)
# plt.close()



# Visualization parameters
fontsize = 22
linewidth = 3.0
markersize = 8

# Configure global plot styling
plt.rcParams.update({
    'font.size': fontsize,
    'font.family': 'serif',  # Use serif font for better readability in publications
    'axes.linewidth': 1.2,   # Border line width
    'axes.edgecolor': 'gray', # Border color
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize+2,  # Title slightly larger than labels
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'grid.linewidth': 0.8,   # Grid line thickness
    'grid.linestyle': '-',   # Solid grid lines
    'grid.alpha': 0.5,       # Grid transparency (50%)
    'grid.color': 'gray',    # Grid color
    'legend.fontsize': fontsize-2,  # Slightly smaller legend text
    'legend.frameon': True,  # Show legend border
    'legend.framealpha': 0.95,  # 95% opaque legend background
    'legend.fancybox': True,  # Rounded legend corners
    'legend.shadow': True    # Add shadow effect to legend
})

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Select first sample for visualization
n = 0

# Define discontinuity point
x_discontinuity = 0.49

# Left subplot: displacement (u)
# Plot line first (behind points)
ax1.plot(data_x, data_u[n, :], color='orange', linewidth=1.5, alpha=0.6)
# Plot scatter points on top
ax1.scatter(data_x, data_u[n, :], color='orange', s=markersize*4, alpha=0.8, edgecolors='darkorange', linewidth=0.5)
# Add discontinuity line
ax1.axvline(x=x_discontinuity, linestyle='--', color='gray', alpha=0.8, linewidth=2)
# Add discontinuity label below x-axis
ax1.text(x_discontinuity, ax1.get_ylim()[0] - (ax1.get_ylim()[1] - ax1.get_ylim()[0])*0.16, 'x=0.49', 
         fontsize=fontsize-4, color='gray', fontweight='bold', ha='center')
# ax1.set_xlabel(r'$x$', fontweight='bold')
# ax1.set_ylabel(r'$u(x)$', fontweight='bold')
ax1.set_title(r'Displacement $u(x)$')
ax1.grid(True)

# Right subplot: body force (b)
# Plot line first (behind points)
ax2.plot(data_x, data_f[n, :], color='forestgreen', linewidth=1.5, alpha=0.6)
# Plot scatter points on top
ax2.scatter(data_x, data_f[n, :], color='forestgreen', s=markersize*4, alpha=0.8, edgecolors='darkgreen', linewidth=0.5)
# Add discontinuity line
ax2.axvline(x=x_discontinuity, linestyle='--', color='gray', alpha=0.8, linewidth=2)
# Add discontinuity label below x-axis
ax2.text(x_discontinuity, ax2.get_ylim()[0] - (ax2.get_ylim()[1] - ax2.get_ylim()[0])*0.16, 'x=0.49', 
         fontsize=fontsize-4, color='gray', fontweight='bold', ha='center')
# ax2.set_xlabel(r'$x$', fontweight='bold')
# ax2.set_ylabel(r'$b(x)$', fontweight='bold')
ax2.set_title(r'Body Force $b(x)$')
ax2.grid(True)

plt.tight_layout()
plt.savefig('%s/%s_data.png' % (current_dir, ex), format='png', dpi=300)

plt.close()
