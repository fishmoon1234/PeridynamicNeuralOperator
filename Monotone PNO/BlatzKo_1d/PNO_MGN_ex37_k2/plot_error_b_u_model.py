
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.integrate as integrate
import matplotlib.ticker as ticker

N = 4
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
# h = np.array([2**(-6), 2**(-7), 2**(-8)])
h0 = 2**(-5)
current_dir = os.path.dirname(os.path.realpath(__file__))
# ex = 'ex20_sample_1'

ex = 'ex37'

layer_info = '256_4'
err_u = [0.26795220375061035, 0.025547953322529793, 0.012035490013659, 0.005903386510908604]
rel_L2_rho_err_k = [0.12786141, 0.02463812, 0.00650929, 0.00206963]
loss_test = [1.67054339e-04, 4.61231622e-06, 3.74735090e-07, 7.73088667e-08]
err_b = [0.06428343843513633, 0.010894672522794751, 0.0034658950305793755, 0.0014525578448255864] # mean of rel L2

# colors = ['darkorange', 'yellowgreen','tomato', 'mediumslateblue', 'palevioletred', 'gray']
# colors = ['forestgreen', 'palevioletred', 'orange']
colors = ['#76cd26', 'tomato', 'mediumslateblue']
fontsize = 20
linewidth = 2.8
markersize = 8  # marker size
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, err_b, 'x--', markerfacecolor='none', color=colors[0], linewidth=linewidth, markersize=markersize, markeredgewidth=2, label=r'Error of $b$')
plt.plot(h, rel_L2_rho_err_k,  'o-.', markerfacecolor='none',color=colors[1], linewidth=linewidth, markersize=markersize, markeredgewidth=2, label=r'$L^2(\rho)$ Error of $k(\xi)$')
plt.plot(h, err_u,  '^:', markerfacecolor='none',color=colors[2], linewidth=linewidth, markersize=markersize, markeredgewidth=2, label=r'Error of $u$')
plt.plot(h, (h/h[-1])*err_b[-1], 'k', label='Theory: $O(\Delta x)$', linewidth=linewidth)
# plt.gca().invert_xaxis()
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('$\Delta x$', fontsize=fontsize+5)
plt.ylabel(r'Error', fontsize=fontsize+5)
plt.title(r'Learn $k(\xi)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=fontsize-2)
plt.tight_layout()
name = '%s_error_all_k_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name), dpi=300)
plt.close()
