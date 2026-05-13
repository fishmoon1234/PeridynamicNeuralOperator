
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.integrate as integrate
import matplotlib.ticker as ticker

N = 4
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
# h = np.array([2**(-6), 2**(-7), 2**(-8)])
h0 = 2**(-5)
current_dir = os.path.dirname(os.path.realpath(__file__))


ex = 'ex37'

layer_info = '128_4_128_4'
err_u = [0.26139742136001587, 0.0277915857732296, 0.011809890158474445, 0.006046296562999487]
rel_L2_rho_err_gk = [0.16381894, 0.05608077, 0.01097551, 0.0061331 ]
loss_test = [9.32719256e-04, 1.31291794e-05, 1.73529347e-06, 4.86029946e-07]

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax1 = plt.subplots(figsize = (6,5))
color_err = 'dodgerblue'

# ax1.plot(h, rel_L2_err_k2, 's--', markerfacecolor='none',color=color_err, linewidth=2)
ax1.plot(h, rel_L2_rho_err_gk,  's--', markerfacecolor='none',color=color_err, linewidth=2, label=r'$L^2(\rho)$ error of $g(\lambda)k(\xi)$')
ax1.plot(h, err_u,  'x:', markerfacecolor='none',color=color_err, linewidth=2, label=r'error of $u$')
# ax1.plot(h, (h/h[-1])*rel_L2_rho_err_gk[-1], 'k', label='slope=1', linewidth=2)
# ax1.plot(h, (h/h[-1])**2*rel_L2_err_k2[-1], 'k--', label='slope=2', linewidth=2)
ax1.plot(h, (h/h[-1])*err_u[-1], 'k', label='slope=1', linewidth=2)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Mesh size')
# ax1.set_ylabel(r'rel $L^2(\rho)$ error of $k(\xi)$', color=color_err)
ax1.tick_params(axis='y', colors=color_err)
ax1.grid(True, which="both", ls="--", color='gray')
# ax1.legend()

color_loss = 'tomato'
ax2 = ax1.twinx() 
ax2.plot(h, loss_test, 'o--', color=color_loss, linewidth=2, label = r'error of $b$')
ax2.set_yscale('log')
# ax2.set_ylabel('Loss value', color=color_loss)
ax2.tick_params(axis='y', colors=color_loss)

lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title(r'Learn $g(\lambda)k(\xi)$')           
plt.tight_layout()
name = '%s_error_all_gk_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name))
plt.close()
