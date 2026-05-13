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
ex = 'ex32'
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
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
dx = 2**(-6)    # change
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


plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1, 4, figsize=(14, 3))
n = 0
titles = [r'(Ex-I) $u$', r'(Ex-I) $b$', r'(Ex-II) $u$', r'(Ex-II) $b$']
for col in range(4):
    if col == 0:  # Ex-I: u1
        axs[col].plot(data_x, data_u[0+n,:], color='orange')
    elif col == 1:  # Ex-I: b1
        axs[col].plot(data_x, data_f[0+n,:], color='forestgreen')
    elif col == 2:  # Ex-II: u1
        axs[col].plot(data_x, data_u[2+n,:], color='orange')
    elif col == 3:  # Ex-II: b1
        axs[col].plot(data_x, data_f[2+n,:], color='forestgreen')
    
    axs[col].set_title(titles[col])
    
plt.tight_layout()
plt.savefig('%s/%s_data.png' % (current_dir, ex), format='png', dpi=300)
plt.close()

