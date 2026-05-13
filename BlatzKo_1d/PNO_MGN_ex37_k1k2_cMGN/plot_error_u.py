
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


ex = 'ex32'
initial_type = 'zero'
Err_u_1_MGN =[1.1646, 2.638e-2, 4.2945e-4, 1.657e-4]
initial_type = 'linear_solution'
Err_u_2_MGN =[1.1646, 2.638e-2, 4.2945e-4, 1.657e-4]


initial_type = 'zero'
Err_u_1_NN =[1.2091, 0.026602471247315407, 0.00249976827763021, 0.005982648581266403]
initial_type = 'linear_solution'
Err_u_2_NN =[1.2091052532196045, 0.02660244144499302, 0.0024997706059366465, 0.0009508949005976319]


colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, Err_u_1_MGN, 's--', markerfacecolor='none',color=colors[0], linewidth=2, label=r"MGN,zero")
plt.plot(h, Err_u_2_MGN, 's--', markerfacecolor='none',color=colors[1], linewidth=2, label=r"MGN, linear")
# plt.plot(h, Err_u_1_NN, 'o--', markerfacecolor='none',color=colors[2], linewidth=2, label=r"NN, zero")
# plt.plot(h, Err_u_2_NN, 's--', markerfacecolor='none',color=colors[3], linewidth=2, label=r"NN, linear")
plt.plot(h, h/h[-1]*Err_u_2_MGN[-1], 'k', label='slope=1', linewidth=2)
plt.plot(h, (h/h[-1])**2*Err_u_2_MGN[-1], 'k--', label='slope=2', linewidth=2)
# plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
plt.ylabel(r'Rel $L^2$ error', fontsize=fontsize)
plt.title(r'Learn $g(\lambda)k(\xi)$: error of $u$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=12)
plt.show()
plt.tight_layout()
name = '%s_gk_error_u.png' % (ex)
plt.savefig (os.path.join (current_dir, name))
plt.close()
