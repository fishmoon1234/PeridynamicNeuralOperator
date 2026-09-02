
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
# DATA = os.path.join(current_dir, 'graphene.mat')
# DATA = os.path.join(current_dir, 'graphene_0.5.mat')
# DATA_NAME = 'lenjo-md-241204'  # graphene, lenjo,
DATA_NAME = 'cg-md-250212_5_70'
DATA = os.path.join(current_dir, '%s.mat' % DATA_NAME)

s = 21
n = s**2
reader = MatReader(DATA)
# x = reader.read_field('coords').reshape(s,s,2)
# u = reader.read_field('disps').reshape(-1,s,s,2)
# b = reader.read_field('forces').reshape(-1,s,s,2)

x = reader.read_field('coords').reshape(n,2)
u = reader.read_field('disps').reshape(-1,n,2)
b = reader.read_field('forces').reshape(-1,n,2)

ndata = u.size(0)

# u_max = torch.max(torch.abs(u))
# f_max = torch.max(torch.abs(f))

# u_L2 = torch.norm(u, dim=[1,2])
# b_L2 = torch.norm(b, dim=[1,2])
u1_L2 = torch.norm(u[:,:,0], dim=[1])
u2_L2 = torch.norm(u[:,:,1], dim=[1])
b1_L2 = torch.norm(b[:,:,0], dim=[1])
b2_L2 = torch.norm(b[:,:,1], dim=[1])

x = x.reshape(s,s,2)
u = u.reshape(-1,s,s,2)
b = b.reshape(-1,s,s,2)
utt = torch.zeros((ndata,s,s,2))

num_groups = 70
group_size = 5
for i in range(num_groups):
    data = u[i * group_size:(i + 1) * group_size]
    diff_data = data[2:,:, :, :] - 2 * data[1:-1, :,:, :] + data[:-2, :, :, :]
    utt[i * group_size+1:(i + 1) * group_size-1] = diff_data
    
    
    
# print(utt)
# print(torch.min(u))
# print(torch.max(b))
utt1_L2 =  torch.norm(utt[:,:,:,0], dim=[1,2])
utt2_L2 =  torch.norm(utt[:,:,:,1], dim=[1,2])



# print(torch.min(u1_L2))
# print(torch.min(u2_L2))
# print(torch.min(b1_L2))
# print(torch.min(b2_L2))
# # filtered_values = u_L2[u_L2 < 1e-1]
# # indices = torch.where(u_L2 < 1e-1)[0]
# # corr_b = b_L2[indices]
# # print(torch.max(corr_b))

for j in range(7):
    plt.rcParams.update({'font.size': 15}) 
    fig, ax = plt.subplots(figsize = (6,5))
    fontsize = 15
    n_plot_start = j*50
    n_plot_end = (j+1)*50
    index = range(n_plot_start,n_plot_end)
    ax.plot(index, utt1_L2[index], color='darkorange', linewidth=1, label=r'$L^2$ norm of $utt1$')
    ax.plot(index, u1_L2[index], color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u1$')
    ax.plot(index, utt2_L2[index], linestyle='--', color='darkorange', linewidth=1, label=r'$L^2$ norm of $utt2$')
    ax.plot(index, u2_L2[index], linestyle='--', color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u2$')
    # ax.plot(index, b1_L2[index], color='k', linewidth=1, label=r'$L^2$ norm of $b_1$')
    # ax.plot(index, b2_L2[index], linestyle='--', color='k', linewidth=1, label=r'$L^2$ norm of $b_2$')
    plt.xlabel('data index')
    plt.legend()
    plt.title('%s'% DATA_NAME)
    plt.tight_layout()
    plt.savefig('%s/figure/%s_utt_u_%s.png' % (current_dir, DATA_NAME, n_plot_start), format='png')
    plt.close(fig)


# x = x.reshape(s,s,2)
# u = u.reshape(-1,s,s,2)
# b = b.reshape(-1,s,s,2)
# n= 1
# plt.rcParams.update({'font.size': 15}) 
# fig, ax = plt.subplots(figsize = (16,10))
# plt.subplot(2,3,1)
# plt.pcolor(x[:,:,0], x[:,:,1], u[n,:,:,0])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$u_1$')
# plt.colorbar()
# plt.subplot(2,3,2)
# plt.pcolor(x[:,:,0], x[:,:,1], b[n,:,:,0])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$b_1$')
# plt.colorbar()
# plt.subplot(2,3,3)
# plt.pcolor(x[:,:,0], x[:,:,1], utt[n,:,:,0])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$utt_1$')
# plt.colorbar()
# plt.subplot(2,3,4)
# plt.pcolor(x[:,:,0], x[:,:,1], u[n,:,:,1])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$u_2$')
# plt.colorbar()
# plt.subplot(2,3,5)
# plt.pcolor(x[:,:,0], x[:,:,1], b[n,:,:,1])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$b_2$')
# plt.colorbar()
# plt.subplot(2,3,6)
# plt.pcolor(x[:,:,0], x[:,:,1], utt[n,:,:,1])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# plt.title('$utt_2$')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# plt.savefig('%s/%s_utt_data_%s.png' % (current_dir, DATA_NAME, n), format='png')
    
   
