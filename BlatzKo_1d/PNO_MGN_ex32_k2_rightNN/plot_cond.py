import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
from egnn_gcl import *
from timeit import default_timer
import os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import bisect
from scipy.special import gamma

torch.set_default_dtype(torch.float64)

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

N = 5
h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h0 = 2**(-8)
ntrain=300
delta = 0.25
Nx = 257 

current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
ex = 'ex32'
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)

# noise_std = 0.003
noise_std = 0.0

# 'Lcurve-RKHS'
# cond = np.array([1498107, 154291, 10584, 451])
# eig_min = np.array([4.88, 14.59, 45.43, 257.67])

# cond = np.array([1498107, 154291, 10584, 451])
# eig_min = np.array([4.88, 14.59, 45.43, 257.67])

cond = np.zeros((N,))
eig_min = np.zeros((N,))

for i in range(N):
    dx = h[i]    # change
    gap = int(dx/h0)
    m_fact = int(delta/dx)
    
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_x[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[:ntrain,::gap].reshape(-1, S)
    data_f = reader.read_field('bodyforce')[:ntrain,::gap].reshape(-1, S)
    
    # dx = data_x[1]-data_x[0]
    # delta = m_fact_train * dx
    ksi_range = torch.range(-m_fact, m_fact).int()
    # n_ksi = 2*m_fact+1
    ksi_range = ksi_range[ksi_range != 0]
    n_ksi = 2*m_fact
    data_ksi = ksi_range*dx
    data_eta = torch.zeros((ntrain, s, n_ksi))
    for j in range(s):
        data_eta[:,j,:] = (data_u[:,m_fact+j+ksi_range].reshape(-1,1,n_ksi)-data_u[:,m_fact+j].reshape(-1,1,1)).squeeze()
        
        
    ksi_plus_eta = data_ksi+data_eta
    ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
    ksi_norm = torch.abs(data_ksi)
    extension = ksi_plus_eta_norm - ksi_norm
    lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
    # lambdaa = 1.0 + extension / (ksi_norm)
    
    weights = (lambdaa-lambdaa**(-3))*ksi_plus_eta/ksi_plus_eta_norm
    A = weights.reshape(-1, n_ksi)*dx
        
    # A = data_eta.reshape(-1, n_ksi)*dx
    
    cond[i] = np.linalg.cond(np.dot(A.T, A))


from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter
fontsize = 15
plt.rcParams.update({
    'font.size': fontsize,
    'axes.titlesize': fontsize,
    'axes.labelsize': fontsize,
    'xtick.labelsize': fontsize-1,
    'ytick.labelsize': fontsize-1
})

h = np.asarray(h).ravel()
cond = np.asarray(cond).ravel()
idx = np.argsort(h)
h_sorted = h[idx]
cond_sorted = cond[idx]

fig, ax = plt.subplots(figsize=(6.8, 5.2))

ax.plot(
    h_sorted, cond_sorted,
    's--', linewidth=2.2,
    markerfacecolor='white',
    markeredgewidth=1.8,
    markersize=6.5,
    color='#d97706',  # dark orange
    label=r'$\mathrm{cond}(A^\top A)$'
)

ax.plot(h_sorted, (h_sorted/h_sorted[-1])**(-2)*cond_sorted[-1], '-', lw=2.0, color='0.3', label='slope = -2')

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)

ax.invert_xaxis()

ax.xaxis.set_major_locator(LogLocator(base=2))
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2,10))))
ax.yaxis.set_minor_formatter(NullFormatter())

ax.grid(True, which='major', ls='--', lw=0.8, color='0.7', alpha=0.9)
ax.grid(True, which='minor', ls=':', lw=0.6, color='0.85', alpha=0.9)

# remove top/right spines; ticks pointing inward
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
ax.tick_params(which='both', direction='in', length=5, width=1)
ax.tick_params(which='minor', length=3)

# axis labels and title
ax.set_xlabel(r'Mesh size $h$')
ax.set_ylabel(r'$\mathrm{cond}(A^\top A)$')
ax.set_title(r'Learn $k(\xi)$: $\mathrm{cond}(A^\top A)$')

# ---- Optional: slope reference lines (log-log y proportional to x^m) ----
# Anchor at the last point and draw two reference slope lines; enable/comment out as needed
def add_slope_ref(ax, xref, yref, slope, frac=0.6, label=None, color='0.3'):
    # Draw over [xref*2^-L, xref] so it sits on the right side
    # frac controls the fraction of the x-axis (in log scale) covered by the line
    x_right = xref
    x_left = xref / (2**(np.clip(frac, 0.1, 0.9) * 2.0))  # 2 is an empirical segment-count
    xs = np.geomspace(x_left, x_right, 50)
    ys = yref * (xs / xref)**(slope)
    ax.plot(xs, ys, '-', lw=1.4, color=color)
    if label:
        # annotate at the midpoint
        xm = np.sqrt(x_left * x_right)
        ym = yref * (xm / xref)**(slope)
        ax.text(xm, ym, label, fontsize=fontsize-2, color=color,
                ha='left', va='bottom')

# examples: slope -3 and -4
# add_slope_ref(ax, h_sorted[-1], cond_sorted[-1], slope=-4, label='slope = -4')
# add_slope_ref(ax, h_sorted[-1], cond_sorted[-1], slope=-2, label='slope = -2')

# legend: enable if reference lines are drawn
ax.legend(frameon=False)

plt.tight_layout()
name = f'{ex}_cond_k.png'
plt.savefig(os.path.join(current_dir, name), dpi=300, bbox_inches='tight', transparent=False)
plt.close()
