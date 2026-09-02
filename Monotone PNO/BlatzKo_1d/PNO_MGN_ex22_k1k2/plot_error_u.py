
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
h0 = 2**(-6)
current_dir = os.path.dirname(os.path.realpath(__file__))
# ex = 'ex20_sample_1'

ex = 'ex22'
initial_type = 'zero'
Err_u_1 =[0.6907, 0.01568, 0.0025479, 0.0101]
initial_type = 'linear_solution'
Err_u_2 =[0.6907, 0.01568, 0.0025479, 0.001683]

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, Err_u_1, 's--', markerfacecolor='none',color='darkorange', linewidth=2, label=r"zero")
plt.plot(h, Err_u_2, 'o--', markerfacecolor='none',color='dodgerblue', linewidth=2, label=r"linear solution")
plt.plot(h, h/h[-1]*Err_u_2[-1], 'k', label='slope=1', linewidth=2)
plt.plot(h, (h/h[-1])**2*Err_u_2[-1], 'k--', label='slope=2', linewidth=2)
# plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
plt.ylabel(r'Rel $L^2$ error', fontsize=fontsize)
plt.title(r'learn $k(\lambda, \xi)$: error of $u$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=12)
plt.show()
plt.tight_layout()
name = '%s_k_error_u_train_300.png' % (ex)
plt.savefig (os.path.join (current_dir, name))
plt.close()
