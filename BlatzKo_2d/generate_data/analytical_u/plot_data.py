
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import sys
import math
from utilities_INO_PD import *
import sklearn.metrics
import scipy
import scipy.integrate as integrate
import time

np.random.seed(123)
# make new directory
# path = 'BlatzKo_Dir_data_PNO'

current_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(current_dir, 'DATA')

s = 33+16
s_i = 33
ex = 'ex12'
# reader = MatReader(os.path.join(data_path,'BK_ex11_ntheta_101_ndata_300_Nx_33.mat'))
reader = MatReader(os.path.join(data_path,'BK_2d_%s_ndata_300_Nx_33.mat' % ex))
x = reader.read_field('coords')
x = x.reshape(s,s,2)
u = reader.read_field('displacement')
b = reader.read_field('bodyforce')
m = 8
x = x[m:m+s_i,m:m+s_i,:]
u = u[:,m:m+s_i,m:m+s_i,:]
b = b[:,m:m+s_i,m:m+s_i,:]
ndata = u.size(0)

print(torch.max(u))
print(torch.max(b))

u_L2 = torch.norm(u.reshape(ndata,-1), dim=[1])
b_L2 = torch.norm(b.reshape(ndata,-1), dim=[1])

# nplot = 200
plt.rcParams.update({'font.size': 15}) 
fig, ax = plt.subplots(figsize = (6,5))
fontsize = 15
ax.plot(b_L2, color='darkorange', linewidth=1, label=r'$L^2$ norm of $b$')
ax.plot(u_L2, color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u$')
plt.xlabel('data index')
plt.legend()
plt.title('%s'% ex)
plt.tight_layout()
plt.savefig('%s/figure/%s_norm_u_b.png' % (current_dir, ex), format='png')
plt.close(fig)

# n = 10   
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# im0 = axes[0, 0].imshow(u[n,:,:,0])
# axes[0, 0].set_title(r'$u_1$')
# fig.colorbar(im0, ax=axes[0,0])
# im1 = axes[0, 1].imshow(u[n,:,:,1])
# axes[0, 1].set_title(r'$u_2$')
# fig.colorbar(im1, ax=axes[0, 1]) 
# im2 = axes[1,0].imshow(b[n,:,:,0])
# axes[1,0].set_title(r'$b_1$')
# fig.colorbar(im2, ax=axes[1,0]) 
# im3 = axes[1,1].imshow(b[n,:,:,1])
# axes[1,1].set_title(r'$b_2$')
# fig.colorbar(im3, ax=axes[1,1]) 
# plt.savefig('%s/data.png' % data_path, format='png')
# plt.show()
# plt.tight_layout()
# plt.show()

# n=10
# plt.rcParams.update({'font.size': 15}) 
# fig, ax = plt.subplots(figsize = (20,5))
# plt.subplot(1,4,1)
# plt.pcolor(x[:,:,0], x[:,:,1], u[n,:,:,0])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$u_1$')
# plt.colorbar()
# plt.subplot(1,4,2)
# plt.pcolor(x[:,:,0], x[:,:,1], u[n,:,:,1])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$u_2$')
# plt.colorbar()
# plt.subplot(1,4,3)
# plt.pcolor(x[:,:,0], x[:,:,1], b[n,:,:,0])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$b_1$')
# plt.colorbar()
# plt.subplot(1,4,4)
# plt.pcolor(x[:,:,0], x[:,:,1], b[n,:,:,1])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$b_2$')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# plt.savefig('%s/data.png' % data_path, format='png')
   
