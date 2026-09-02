
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import sys
import math
from utilities_INO_PD import *
import sklearn.metrics
import scipy
import scipy.integrate as integrate
import time
from timeit import default_timer
import torch

np.random.seed(123)
    
current_dir = os.path.dirname(os.path.realpath(__file__))
# DATA_PATH = '%s/../2d_BK_gen_data/analytical_u/DATA/' % current_dir
DATA_PATH = '%s/DATA/' % current_dir
ex = 'ex2'
DATA_NAME = 'BK_2d_%s_ndata_2_Nx_33' % (ex)
Nx = 33
Nx_all = 33+16
h0 = 2**(-5)
delta = 0.25
ex = 'ex4'
DATA_NAME = 'BK_2d_%s_ndata_100_Nx_65' % (ex)
Nx = 65
Nx_all = 65+32
h0 = 2**(-6)
delta = 0.25

DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
fig_dir = os.path.join(current_dir, "figure")
os.makedirs(fig_dir, exist_ok=True)


dx = h0 
gap = int(dx/h0)
m_fact = int(delta/dx)
s0 = int((Nx-1)/gap)+1
s = s0+2*m_fact
n = s**2
print(f'>> Training Mesh resolution: {s}x{s}')

reader = MatReader(DATA)

data_x = reader.read_field('coords').reshape(Nx_all, Nx_all, 2)
data_u = reader.read_field('displacement')
data_f = reader.read_field('bodyforce')
data_x = data_x[::gap,::gap,:].reshape(s, s, 2)
data_u = data_u[:,::gap,::gap,:].reshape(-1, s, s, 2)
data_f = data_f[:,::gap,::gap,:].reshape(-1, s, s, 2)


for n in range(5):
    data = [
        data_u[n,:,:,0], data_u[n,:,:,1], data_f[n,:,:,0], data_f[n,:,:,1]
    ]
    title = [
        r'$u^{0}$', r'$u^{1}$', r'$b^{0}$', r'$b^{1}$'
    ]
    plt.rcParams.update({'font.size': 15}) 
    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for col in range(4):
        im = axs[col].imshow(data[col])
        axs[col].set_title(title[col])
        fig.colorbar(im, ax=axs[col])
    plt.tight_layout()
    plt.savefig('%s/figure/%s_data_%s.png' % (current_dir, ex, n), format='png')
    plt.close()

    