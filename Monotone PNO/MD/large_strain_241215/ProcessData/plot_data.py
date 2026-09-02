
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

def is_periodic(tensor, tolerance=1e-5):
    if tensor.shape[0] != tensor.shape[1]:
        raise ValueError("The input tensor must be a square matrix.")
    
    n = tensor.shape[0]
    
    if not np.allclose(tensor[0, :], tensor[-1, :], atol=tolerance):
        return False
    
    if not np.allclose(tensor[:, 0], tensor[:, -1], atol=tolerance):
        return False
    
    return True

def calculate_max_boundary_difference(tensor):
    row_diff = np.abs(tensor[0, :] - tensor[-1, :])
    col_diff = np.abs(tensor[:, 0] - tensor[:, -1])
    
    max_row_diff = torch.max(row_diff)
    max_col_diff = torch.max(col_diff)
    
    return max(max_row_diff, max_col_diff)


np.random.seed(123)

current_dir = os.path.dirname(os.path.realpath(__file__))
# DATA = os.path.join(current_dir, 'graphene.mat')
# DATA = os.path.join(current_dir, 'graphene_0.5.mat')
# DATA_NAME = 'md-241215_5_60'  # graphene, lenjo, md-241215
DATA_NAME = 'cg-md-250212'
# DATA_NAME = 'cg-md-250212_0.5'
# DATA_NAME = 'cg-md-250212_5_70'
# DATA_NAME = 'test-250318'
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

INDEX_NAME = 'cg-md-250212_index_b1_0.02'
DATA_index = os.path.join(current_dir, "../ProcessData/%s.mat" % INDEX_NAME)
reader = MatReader(DATA_index)
index = reader.read_field('index').flatten().int()
u = u[index]
b = b[index]

ndata = u.size(0)


####### check if peridoc  ##########
# is_per = np.zeros((ndata,))
# non_periodic_indices = []
# non_periodic_boundary_difference = []
# for i in range(ndata):
#     ui = u[i,:,0].reshape(21, 21)
#     if not is_periodic(ui, 5e-2):
#         non_periodic_indices.append(i)
#         err = calculate_max_boundary_difference(ui).item()
#         non_periodic_boundary_difference.append(err)
#         print(i)
#         plt.imshow(ui)
#         plt.colorbar()
#         plt.title('%s: %s, err: %.4f'% (DATA_NAME, i, err))
#         plt.tight_layout()
#         plt.savefig('%s/figure/%s_data_%s.png' % (current_dir, DATA_NAME,i), format='png')
#         plt.close()


u_L2 = torch.norm(u, dim=[1,2])
b_L2 = torch.norm(b, dim=[1,2])
u1_L2 = torch.norm(u[:,:,0], dim=[1])
u2_L2 = torch.norm(u[:,:,1], dim=[1])
b1_L2 = torch.norm(b[:,:,0], dim=[1])
b2_L2 = torch.norm(b[:,:,1], dim=[1])


print(torch.min(u_L2))
print(torch.min(b_L2))
print(torch.min(u1_L2))
print(torch.min(u2_L2))
print(torch.min(b1_L2))
print(torch.min(b2_L2))


# plt.rcParams.update({'font.size': 15}) 
# fig, ax = plt.subplots(figsize = (6,5))
# fontsize = 15
# ax.plot(b1_L2[:100], color='darkorange', linewidth=1, label=r'$L^2$ norm of $b1$')
# ax.plot(u1_L2[:100], color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u1$')
# ax.plot(b2_L2[:100], linestyle='--', color='darkorange', linewidth=1, label=r'$L^2$ norm of $b2$')
# ax.plot(u2_L2[:100], linestyle='--', color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u2$')
# plt.xlabel('data index')
# plt.legend()
# plt.title('%s'% DATA_NAME)
# plt.tight_layout()
# plt.savefig('%s/figure/%s_norm_u1_b1.png' % (current_dir, DATA_NAME), format='png')
# plt.close(fig)


# start = 250
# nplot = 10
# plot_index = range(start,start+nplot)
# plt.rcParams.update({'font.size': 15}) 
# fig, ax = plt.subplots(figsize = (6,5))
# fontsize = 15
# ax.plot(plot_index, b_L2[plot_index], color='darkorange', linewidth=1, label=r'$L^2$ norm of $b$')
# ax.plot(plot_index, u_L2[plot_index], color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u$')
# plt.xlabel('data index')
# plt.legend()
# plt.title('%s'% DATA_NAME)
# plt.tight_layout()
# plt.savefig('%s/figure/%s_norm_u_b_%s_%s.png' % (current_dir, DATA_NAME, start, nplot), format='png')
# plt.close(fig)


start = 0
nplot = index.size(0)
plot_index = range(start,start+nplot)
plt.rcParams.update({'font.size': 15}) 
fig, ax = plt.subplots(figsize = (20, 5))
fontsize = 15
# ax.plot(plot_index, b_L2[plot_index], color='darkorange', linewidth=1, label=r'$L^2$ norm of $b$')
ax.plot(plot_index, u_L2[plot_index], color='forestgreen', linewidth=1, label=r'$L^2$ norm of $u$')
plt.xlabel('data index')
plt.legend()
plt.title('%s'% INDEX_NAME)
plt.tight_layout()
plt.savefig('%s/figure/%s_norm_u_%s_%s.png' % (current_dir, INDEX_NAME, start, nplot), format='png')
plt.close(fig)


# start = 0
# n_plot = 2
# u = u.reshape(ndata, s, s, 2)
# b = b.reshape(ndata, s, s, 2)
# for n in range(start, start+n_plot): 
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
#     im0 = axes[0, 0].imshow(u[n,:,:,0], interpolation='spline16')
#     axes[0, 0].set_title(r'$u_1$')
#     fig.colorbar(im0, ax=axes[0,0])
#     im1 = axes[0, 1].imshow(u[n,:,:,1], interpolation='spline16')
#     axes[0, 1].set_title(r'$u_2$')
#     fig.colorbar(im1, ax=axes[0, 1]) 
#     im2 = axes[1,0].imshow(b[n,:,:,0], interpolation='spline16')
#     axes[1,0].set_title(r'$b_1$')
#     fig.colorbar(im2, ax=axes[1,0]) 
#     im3 = axes[1,1].imshow(b[n,:,:,1], interpolation='spline16')
#     axes[1,1].set_title(r'$b_2$')
#     fig.colorbar(im3, ax=axes[1,1]) 
#     plt.show()
#     plt.tight_layout()
#     plt.savefig('%s/figure/%s_data_%s.png' % (current_dir, DATA_NAME, n), format='png')
    
# # start = 81
# # n_plot = 1
# u = u.reshape(ndata, s, s, 2)
# b = b.reshape(ndata, s, s, 2)
# plot_index = np.arange(0, 1400, 20)
# plot_index = [160]
# fontsize = 30
# for n in plot_index: 
#     plt.rcParams.update({'font.size': 20})
#     fig, axes = plt.subplots(1, 4, figsize=(20, 4))
#     im0 = axes[0].imshow(u[n,:,:,0], interpolation='spline16')
#     axes[0].set_title(r'$u_1$', fontsize =fontsize)
#     fig.colorbar(im0, ax=axes[0])
#     im1 = axes[1].imshow(u[n,:,:,1], interpolation='spline16')
#     axes[1].set_title(r'$u_2$', fontsize =fontsize)
#     fig.colorbar(im1, ax=axes[1]) 
#     im2 = axes[2].imshow(b[n,:,:,0], interpolation='spline16')
#     axes[2].set_title(r'$b_1$', fontsize =fontsize)
#     fig.colorbar(im2, ax=axes[2]) 
#     im3 = axes[3].imshow(b[n,:,:,1], interpolation='spline16')
#     axes[3].set_title(r'$b_2$', fontsize =fontsize)
#     fig.colorbar(im3, ax=axes[3]) 
#     plt.show()
#     plt.tight_layout()
#     plt.savefig('%s/figure/%s_data_%s.png' % (current_dir, DATA_NAME, n), format='png')
    