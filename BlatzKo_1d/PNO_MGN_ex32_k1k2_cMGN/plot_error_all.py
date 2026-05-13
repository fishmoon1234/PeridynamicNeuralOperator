
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
# initial_type = 'zero'
# Err_u_1_MGN =[1.0352, 0.02637, 0.002356, 0.01250]
# initial_type = 'linear_solution'
# Err_u_2_MGN =[0.26795220375061035, 0.025547953322529793, 0.012035490013659, 0.005903386510908604]

layer_info = '128_4'
err_u = [1.2061941623687744, 0.026816776022315025, 0.0008740746998228133, 0.0005806388799101114]
rel_L2_rho_err_gk = [0.28460884, 0.09668645, 0.02480265, 0.00737275]
loss_train = [3.41872040e-05, 2.14708822e-07, 2.03999162e-08, 9.32441758e-09]



fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax1 = plt.subplots(figsize = (6,5))
color_err = 'dodgerblue'

# ax1.plot(h, rel_L2_err_k2, 's--', markerfacecolor='none',color=color_err, linewidth=2)
ax1.plot(h, rel_L2_rho_err_gk,  's--', markerfacecolor='none',color=color_err, linewidth=2, label=r'Error of $g(\lambda)k(\xi)$')
ax1.plot(h, err_u,  'o:', markerfacecolor='none',color=color_err, linewidth=2, label=r'Error of $u$')
ax1.plot(h, (h/h[-1])**2*rel_L2_rho_err_gk[-1], 'k', label='slope=2', linewidth=2)
# ax1.plot(h, (h/h[-1])**2*rel_L2_err_k2[-1], 'k--', label='slope=2', linewidth=2)
ax1.plot(h, (h/h[-1])**2*err_u[-1], 'k--', label='slope=2', linewidth=2)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Mesh size')
# ax1.set_ylabel(r'rel $L^2(\rho)$ error of $k(\xi)$', color=color_err)
ax1.tick_params(axis='y', colors=color_err)
ax1.grid(True, which="both", ls="--", color='gray')
ax1.legend()

color_loss = 'tomato'
ax2 = ax1.twinx() 
ax2.plot(h, loss_train, 'x--', color=color_loss, linewidth=2)
ax2.set_yscale('log')
ax2.set_ylabel('Loss value', color=color_loss)
ax2.tick_params(axis='y', colors=color_loss)

plt.title(r'Learn $g(\lambda)k(\xi)$')           
plt.tight_layout()
name = '%s_error_all_gk_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name), dpi=300)
plt.close()
