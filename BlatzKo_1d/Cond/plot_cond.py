import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
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

current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, 'BlatzKo_data_1d')
os.makedirs(path, exist_ok=True)

# example_type = 'nondiff'
example_type = 'BK'

ndata = 50
xmin, xmax = 0, 1
# Nx = 129
Nx = 1025
h0 = (xmax-xmin)/(Nx-1)
xh = np.linspace(xmin, xmax, Nx)
# m_fact = 3
# delta = m_fact*h
delta = 0.25
m_fact = int(delta/h0)
Nx_all = Nx+2*m_fact
xh_all = np.linspace(xmin-delta, xmax+delta, Nx_all)

# g_fun = lambda x: np.ones_like(x)
# g_fun = lambda x: x**2
g_fun = lambda x: x-x**(-3)
# kernel_fun = lambda x: g_fun(x)
h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8), 2**(-9), 2**(-10)])
N = len(h)


nfeq_all = np.array([10, 20, 40, 60, 80, 100])
# maxium_all = np.array([0.001, 0.001, 0.001, 0.001, 0.0001])
# maxium_all = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
maxium_all = np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001])
n_nfeq = len(nfeq_all)
cond = np.zeros((n_nfeq, N))
ATA_inv_A = np.zeros((n_nfeq, N))
for i_nfeq in range(n_nfeq):
    nfeq = nfeq_all[i_nfeq]

    feq = np.linspace(0, nfeq-1, nfeq).reshape(-1,1)
    u_fun = lambda a, b, x: np.sum(a*np.exp(-feq/nfeq)*np.sin(feq*np.pi*x)+ b*np.exp(-feq/nfeq)*np.cos(feq*np.pi*x), axis=0)
    # u_fun = lambda A, a, b, x: A*np.sin(a*x+b)

    data_u = np.zeros((ndata, Nx_all))
    for n in range(ndata):
        maxium = maxium_all[i_nfeq]
        a = np.random.uniform(-maxium, maxium, nfeq).reshape(-1,1)
        b = np.random.uniform(-maxium, maxium, nfeq).reshape(-1,1)
        u = u_fun(a, b, xh_all.reshape(1,-1))
        # maxium = nfeq*np.pi
        # A = np.random.uniform(0.005, 0.02)
        # a = np.round(np.random.uniform(0, maxium), 2)
        # b = np.round(np.random.uniform(0, maxium), 2)
        # u = u_fun(A, a, b, xh_all)
    
        data_u[n,:] = u
    
    data_u = torch.from_numpy(data_u)
    for i in range(N):
        dx = h[i]    # change
        gap = int(dx/h0)
        m_fact = int(delta/dx)
        data_u_i = data_u[:, ::gap]
        
        s = int((Nx-1)/gap)+1
        S = s+2*m_fact    
        ksi_range = torch.range(-m_fact, m_fact).int()
        # n_ksi = 2*m_fact+1
        ksi_range = ksi_range[ksi_range != 0]
        n_ksi = 2*m_fact
        data_ksi = ksi_range*dx
        data_eta = torch.zeros((ndata, s, n_ksi))
        for j in range(s):
            data_eta[:,j,:] = (data_u_i[:,m_fact+j+ksi_range].reshape(-1,1,n_ksi)-data_u_i[:,m_fact+j].reshape(-1,1,1)).squeeze()
        
        if example_type == 'nondiff':
            A = data_eta.reshape(-1, n_ksi)*dx
        elif example_type == 'BK':
            ksi_plus_eta = data_ksi+data_eta
            ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
            ksi_norm = torch.abs(data_ksi)
            extension = ksi_plus_eta_norm - ksi_norm
            lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
            # lambdaa = 1.0 + extension / (ksi_norm)
            
            weights = g_fun(lambdaa)*ksi_plus_eta/(ksi_plus_eta_norm+1e-9)
            A = weights.reshape(-1, n_ksi)*dx
        
        cond[i_nfeq, i] = np.linalg.cond(np.dot(A.T, A))   
        ATA_inv_A[i_nfeq, i] = np.linalg.norm(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
        
        
def plot_A(cond, name, h, y_label):        

    from matplotlib.ticker import LogLocator, LogFormatter, NullFormatter
    fontsize = 22
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
        'legend.fontsize': fontsize-4,  # Slightly smaller legend text
        'legend.frameon': True,  # Show legend border
        'legend.framealpha': 0.9,  # 95% opaque legend background
        'legend.fancybox': True # Rounded legend corners
        # 'legend.shadow': True    # Add shadow effect to legend
    })

    h = np.asarray(h).ravel()
    # Sort h and corresponding indices
    idx = np.argsort(h)
    h_sorted = h[idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors for different nfeq values
    # colors = ['#FF6B35', '#004E98', '#8E44AD', '#2ECC71', '#F39C12', '#95A5A6']
    import seaborn as sns
    colors = sns.color_palette('muted', n_colors=n_nfeq)

    # Plot each nfeq line with different colors
    for i_nfeq in range(n_nfeq):
        nfeq = nfeq_all[i_nfeq]
        cond_sorted = cond[i_nfeq, idx]  # Sort cond for this nfeq according to h sorting
        
        ax.plot(
            h_sorted, cond_sorted,
            '*--', linewidth=2.2,
            markerfacecolor='white',
            markeredgewidth=1.8,
            markersize=6.5,
            color=colors[i_nfeq % len(colors)],
            label=f'nfeq = {nfeq}'
        )

    # Add reference slope line using the last nfeq data
    cond_sorted_ref = cond[-1, idx]  # Use the last nfeq for reference slope
    ax.plot(h_sorted, (h_sorted/h_sorted[-1])**(-2)*cond_sorted_ref[-1], '-', lw=2.0, color='0.3', label='slope = -2')
    # ax.plot(h_sorted, (h_sorted/h_sorted[-1])**(-4)*cond_sorted_ref[-1], '-', lw=1.4, color='0.25', label='slope = -4')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)

    ax.invert_xaxis()

    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=7))
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2,10))))
    # ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True)


    ax.set_xlabel(r'Mesh size $h$')
    ax.set_ylabel(y_label)
    # ax.set_title(r'Learn $k(\xi)$: $\mathrm{cond}(A^\top A)$')

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
    # add_slope_ref(ax, h_sorted[-1], cond_sorted[-1], slope=-3, label='slope = -3')

    ax.legend(loc='best')

    plt.tight_layout()
    # name = f'{example_type}_cond_k_ndata{ndata}_{maxium_all[-1]}.png'
    plt.savefig(os.path.join(current_dir, name), dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    
    
plot_A(cond, f'{example_type}_cond_ndata{ndata}_{maxium_all[-1]}.png', h, r'$\mathrm{cond}(A^\top A)$')
plot_A(ATA_inv_A, f'{example_type}_ATA_inv_A_ndata{ndata}_{maxium_all[-1]}.png', h, r'$\|(A^\top A)^{-1}A\|_2$')
