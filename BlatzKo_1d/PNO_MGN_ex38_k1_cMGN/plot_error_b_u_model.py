
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
# rel_L2_rho_err_k = [0.18337411, 0.0475185,  0.01400646, 0.00331474]
rel_L2_rho_err_g =  [0.13033879, 0.03696738, 0.01116593, 0.00337568]
# loss_test = [4.96672919e-05, 1.16292750e-06, 6.61243439e-08, 3.71491292e-09]
err_b = [0.08145881867216578,0.01356877385213972, 0.004039848483908166, 0.0016465185964602213] # mean of rel L2

# colors = ['darkorange', 'yellowgreen','tomato', 'mediumslateblue', 'palevioletred', 'gray']
# colors = ['forestgreen', 'palevioletred', 'orange']
colors = ['#76cd26', 'tomato', 'mediumslateblue']
fontsize = 20
linewidth = 2.8
markersize = 8  # marker size
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, err_b, 'x--', markerfacecolor='none', color=colors[0], linewidth=linewidth, label=r'error of $b$', markersize=markersize, markeredgewidth=2)
plt.plot(h, rel_L2_rho_err_g,  'o-.', markerfacecolor='none',color=colors[1], linewidth=linewidth, label=r'$L^2(\rho)$ error of $g(\lambda)$', markersize=markersize, markeredgewidth=2 )
plt.plot(h, err_u,  '^:', markerfacecolor='none',color=colors[2], linewidth=linewidth, label=r'error of $u$', markersize=markersize, markeredgewidth=2)
plt.plot(h, h/h[-1]*err_b[-1], 'k', label='Theory: $O(\Delta x)$', linewidth=linewidth)
# plt.gca().invert_xaxis()
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('$\Delta x$', fontsize=fontsize+5)
plt.ylabel(r'Error', fontsize=fontsize+5)
plt.title(r'Learn $g(\lambda)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=fontsize-2)
plt.tight_layout()
name = '%s_error_all_g_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name))
plt.close()
