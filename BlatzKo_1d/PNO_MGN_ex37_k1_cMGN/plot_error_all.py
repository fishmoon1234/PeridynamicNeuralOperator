
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

layer_info = '128_4'
err_u = [1.1096857786178589, 0.031025610864162445, 0.0032880937214940786,  0.0008368344861082733] # mean of rel L2
rel_L2_rho_err_k = [0.18337411, 0.0475185,  0.01400646, 0.00331474]
loss_test = [4.96672919e-05, 1.16292750e-06, 6.61243439e-08, 3.71491292e-09]
err_b = [0.08145881867216578,0.01356877385213972, 0.004039848483908166, 0.0016465185964602213] # mean of rel L2


fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax1 = plt.subplots(figsize = (6,5))
color_err = 'dodgerblue'

# ax1.plot(h, rel_L2_err_k2, 's--', markerfacecolor='none',color=color_err, linewidth=2)
ax1.plot(h, rel_L2_rho_err_k,  's--', markerfacecolor='none',color=color_err, linewidth=2, label=r'$L^2(\rho)$ error of $g(\lambda)$')
ax1.plot(h, err_u,  'x:', markerfacecolor='none',color=color_err, linewidth=2, label=r'error of $u$')
# ax1.plot(h, h/h[-1]*rel_L2_rho_err_gk[-1], 'k', label='slope=1', linewidth=2)
ax1.plot(h, (h/h[-1])**2*rel_L2_rho_err_k[-1], 'k', label='slope=2', linewidth=2)
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
ax2.plot(h, loss_test, 'x--', color=color_loss, linewidth=2, label=r'error of $b$')
ax2.set_yscale('log')
# ax2.set_ylabel('Loss value', color=color_loss)
ax2.tick_params(axis='y', colors=color_loss)
# ax2.legend()

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# fig.legend()
plt.title(r'Learn $g(\lambda)$')           
plt.tight_layout()
name = '%s_error_all_k_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name))
plt.close()
