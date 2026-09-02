
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

ex = 'ex32'

layer_info = '128_4'
# err_u = [1.2061941623687744, 0.026816776022315025, 0.0008740746998228133, 0.0005806388799101114]
err_u = [1.2061941623687744, 0.026816776022315025, 0.0008740746998228133, 0.000140527] # the last result is of 128,6 
rel_L2_rho_err_gk = [0.28460884, 0.09668645, 0.02480265, 0.00737275]
loss_train = [3.380122143435639e-05, 2.1241019791589795e-07, 1.0587497368099391e-08, 3.012223790548184e-09]
# err_b = [0.08261484887722065, 0.005102617761710572, 0.0011007040621090207, 0.0005117527372138215] # mean of rel L2
err_b = [0.08261484887722065, 0.005102617761710572, 0.0011007040621090207, 0.0001602504292415311] # mean of rel L2

# colors = ['darkorange', 'yellowgreen','tomato', 'mediumslateblue', 'palevioletred', 'gray']
# colors = ['forestgreen', 'palevioletred', 'orange']
colors = ['#76cd26', 'tomato', 'mediumslateblue']
fontsize = 20
linewidth = 2.8
markersize = 8  # marker size
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, err_b, 'x--', markerfacecolor='none', color=colors[0], linewidth=linewidth, label=r'Error of $b$', markersize=markersize, markeredgewidth=2)
plt.plot(h, rel_L2_rho_err_gk,  'o-.', markerfacecolor='none',color=colors[1], linewidth=linewidth, label=r'$L^2(\rho)$ Error of $g(\lambda)k(\xi)$', markersize=markersize, markeredgewidth=2 )
plt.plot(h, err_u,  '^:', markerfacecolor='none',color=colors[2], linewidth=linewidth, label=r'Error of $u$', markersize=markersize, markeredgewidth=2)
plt.plot(h, (h/h[-1])**2*err_b[-1], 'k', label='Theory: $O(\Delta x^2)$', linewidth=linewidth)
plt.plot(h, (h/h[-1])**2*rel_L2_rho_err_gk[-1], 'k', linewidth=linewidth)
# plt.gca().invert_xaxis()
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('$\Delta x$', fontsize=fontsize+5)
plt.ylabel(r'Error', fontsize=fontsize+5)
plt.title(r'Learn $g(\lambda)k(\xi)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=fontsize-2, loc='upper left')
plt.tight_layout()
name = '%s_error_all_gk_%s_wrong.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name), dpi=300)
plt.close()
