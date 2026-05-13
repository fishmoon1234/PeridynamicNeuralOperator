#quadrature weights
import torch

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.integrate as integrate
import matplotlib.ticker as ticker

h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# err = np.array([0.0111, 0.0024, 0.00145, 0.00121, 0.000872])
# err = np.array([0.0726488, 0.045723725, 0.04161059, 0.0386333, 0.031260498])
# err = np.array([0.037390035, 0.008931619, 0.0033416722, 0.0023599407, 0.0034566324])
err = np.array([5.20, 0.11, 0.0187, 0.014, 0.016])*1e-3

current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, 'Results')
os.makedirs(path, exist_ok=True)  
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,6))
plt.plot(h, err, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
plt.plot(h, h/h[0]*err[0], 'k', label='slope=1', linewidth=2)
plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
# plt.ylabel('Quadrature error', fontsize=fontsize)
plt.ylabel(r'Relative $L^2$ error', fontsize=fontsize)
# plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0], numticks=10))
# plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='auto', numticks=10))
# plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
# plt.ylim(1e-6,1e-4)
# show the figure
# plt.title(r'Without constant basis ($\alpha=%.1f$)' % (alpha), fontsize=14)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.show()

plt.tight_layout()
# name = 'ex3_alpha%s_porder%d_delta%f.png' % (alpha, p_order, delta)
name = 'ex1_error.png'
plt.savefig (os.path.join (path, name))
