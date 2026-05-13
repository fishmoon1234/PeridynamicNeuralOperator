
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

layer_info = '128_4_128_4'
err_u = [0.26139742136001587, 0.0277915857732296, 0.011809890158474445, 0.006046296562999487]
rel_L2_rho_err_gk = [0.16381894, 0.05608077, 0.01097551, 0.0061331 ]
# loss_test = [9.32719256e-04, 1.31291794e-05, 1.73529347e-06, 4.86029946e-07]
# err_b = [0.08261484887722065, 0.005102617761710572, 0.0011007040621090207, 0.0005117527372138215] # mean of rel L2
# layer_info = '128_4_256_4'
err_b = [0.06670836634933948, 0.006764802094548941, 0.0031765437498688697, 0.0016086156782694162]

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
plt.plot(h, (h/h[-1])*err_b[-1], 'k', label='Theory: $O(\Delta x)$', linewidth=linewidth)
# plt.plot(h, (h/h[-1])**2*rel_L2_rho_err_gk[-1], 'k', label='slope=2', linewidth=2)
# plt.gca().invert_xaxis()
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('$\Delta x$', fontsize=fontsize+5)
plt.ylabel(r'Error', fontsize=fontsize+5)
plt.title(r'Learn $g(\lambda)k(\xi)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=fontsize-2)
plt.tight_layout()
name = '%s_error_all_gk_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name), dpi=300)
plt.close()
