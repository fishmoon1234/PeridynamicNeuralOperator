
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
rel_L2_rho_err_gk = [0.12786141, 0.02463812, 0.00650929, 0.00206963]
loss_test = [1.67054339e-04, 4.61231622e-06, 3.74735090e-07, 7.73088667e-08]



fontsize = 22
linewidth = 2.5
plt.rcParams.update({'font.size': fontsize}) 
fig, ax1 = plt.subplots(figsize = (6,5))
color_err = 'dodgerblue'

# ax1.plot(h, rel_L2_err_k2, 's--', markerfacecolor='none',color=color_err, linewidth=2)
ax1.plot(h, rel_L2_rho_err_gk,  's-', markerfacecolor='none',color=color_err, linewidth=linewidth, label=r'$L^2(\rho)$ error of $k(\xi)$')
ax1.plot(h, err_u,  'x:', markerfacecolor='none',color=color_err, linewidth=linewidth, label=r'error of $u$')
# ax1.plot(h, h/h[-1]*rel_L2_rho_err_gk[-1], 'k', label='slope=1', linewidth=2)
ax1.plot(h, h/h[-1]*err_u[-1], 'k', label='slope=1', linewidth=linewidth)
# ax1.plot(h, (h/h[-1])**2*rel_L2_err_k2[-1], 'k--', label='slope=2', linewidth=2)
# ax1.plot(h, (h/h[-1])**2*L2_err_k2[-1], 'k--', label='slope=2', linewidth=2)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Mesh size')
# ax1.set_ylabel(r'rel $L^2(\rho)$ error of $k(\xi)$', color=color_err)
ax1.tick_params(axis='y', colors=color_err)
ax1.grid(True, which="both", ls="--", color='gray')
# ax1.legend()

color_loss = 'tomato'
ax2 = ax1.twinx() 
ax2.plot(h, loss_test, 'o--', color=color_loss, linewidth=linewidth, label=r'error of $b$')
ax2.set_yscale('log')
# ax2.set_ylabel('Loss value', color=color_loss)
ax2.tick_params(axis='y', colors=color_loss)
# ax2.legend()

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# fig.legend()
plt.title(r'Learn $k(\xi)$')           
plt.tight_layout()
name = '%s_error_all_k_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name))
plt.close()
