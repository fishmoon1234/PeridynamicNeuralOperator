import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
from egnn_gcl import *
from timeit import default_timer
import os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import bisect

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


N = 4
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
ntrain = 300

lrs = [0.01]
lr = [0.99, 0.998]

# model and training parameters
batch_size = 10
batch_size2 = batch_size
layer_info = '128_6_256_4'
phi_1_layer = [1, 128, 128, 128, 128, 128, 128, 1]
phi_2_layer = [1, 256, 256, 256, 256, 1]
act_xi = 'ReLU'
# act_xi = 'GELU'
# act_xi = 'Tanh'
# act_xi = 'Softplus'
if act_xi == 'ReLU':
    act_fun_xi = torch.nn.ReLU
elif act_xi == 'GELU': 
    act_fun_xi = torch.nn.GELU
elif act_xi == 'Tanh': 
    act_fun_xi = torch.nn.Tanh
elif act_xi == 'Softplus': 
    act_fun_xi = torch.nn.Softplus


current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex32'
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
# g_fun = lambda x: np.ones_like(x)
# g_fun = lambda x: x*(1-x**2)
# g_fun = lambda x: x*np.exp(-50*x**2)
# g_fun = lambda x: np.abs(x)*np.sin(3*np.pi*np.abs(x))
# g_fun = lambda x: x*(np.abs(x)-delta)
g_fun = lambda x: np.exp(-50*x**2)*(delta-np.abs(x))
# g_fun = lambda x: np.abs(x)*np.cos(3*np.pi*np.abs(x))
k1_fun = lambda x: x - x**(-3)


Lambda_plot, Rho_lambda, k1_all = [], [], []
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']

for i in range(N):
    gap = int(h[i]/h0)
    lrs = [0.001]
    lr = [0.99, 0.998]
    normlize_type = 'k'
    
    layer_index = 5
    # if i >=2:
    #     layer_index = 7
    #     layer_info = '128_8_128_5'
    #     phi_1_layer = [1, 128, 128, 128, 128, 128, 128, 128, 128, 1]
    #     phi_2_layer = [1, 128, 128, 128, 128, 128, 1]
    model = E_GCL_GKN(phi_1_layer, layer_index, phi_2_layer, act_fun_xi).to(device)

    base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr%s_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
    model_path = '%s/model_%s.ckpt' % (base_dir, layer_index)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    lambda_min, lambda_max = 0.3, 5
    
    Nlambda = 100
    lambdaa = torch.linspace(lambda_min, lambda_max, Nlambda)
    # dlambda = lambda_rho[1]- lambda_rho[0]
    
    Lambdaa = lambdaa.reshape(-1,1)
    Lambdaa_cuda = Lambdaa.to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    
    k1_true = k1_fun(lambdaa)
    k1_NN = (model.phi_MGN(Lambdaa_cuda)- model.phi_MGN(Lambdaa_1_cuda)).reshape(-1,1)
    k1_NN = k1_NN.cpu().detach().flatten()
    
    error_k1 = torch.norm(k1_NN-k1_true)/torch.norm(k1_true)
    print(error_k1)
    
    k1_all.append(k1_NN)
        


###################################
# plot g(lambda)
################################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
plt.plot(lambdaa, k1_true, color='k', linewidth=1.5, label='true') 
for i in range(N):
    plt.plot(lambdaa, k1_all[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
# for i in [0,2,4,6]:
#     plt.plot(lambdaa, k1_all[i],  color=colors[int(i/2)], linewidth=1.5, label=r'$h=%s$' % (h[i]))
plt.xlabel(r'$\lambda$', fontsize=fontsize)
plt.ylabel(r'$g(\lambda)$', fontsize=fontsize)
# plt.yscale('log')
plt.title(r'NN learned $g(\lambda)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.tight_layout()
name = '%s_g_%s_ood_%s_%s_MGN.png' % (ex, layer_info, lambda_min, lambda_max)
plt.savefig (os.path.join (current_dir, name))
plt.close()

