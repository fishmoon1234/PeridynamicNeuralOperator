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
from scipy.special import gamma

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

N = 5
h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h0 = 2**(-8)
ntrain=300
delta = 0.25
Nx = 257 

current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
ex = 'ex30'
# DATA_NAME = 'BK_%s_ndata_500_Nx_129_delta_0.25_h_0.0078125' % (ex)
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)

# noise_std = 0.003
noise_std = 0.0
# model and training parameters
batch_size = 10
batch_size2 = batch_size
# layer_info = '64_5_64_5'
# phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
# phi_2_layer = [1, 64, 64, 64, 64, 64, 1]
layer_info = '128_5_256_4'
phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
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

model = E_GCL_GKN(phi_1_layer, phi_2_layer, act_fun_xi).to(device)


delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
# g_fun = lambda x: np.ones_like(x)
# g_fun = lambda x: x*(1-x**2)
g_fun = lambda x: x*np.exp(-50*x**2)

# 'Lcurve-RKHS'
# cond = np.array([1498107, 154291, 10584, 451])
# eig_min = np.array([4.88, 14.59, 45.43, 257.67])

# cond = np.array([1498107, 154291, 10584, 451])
# eig_min = np.array([4.88, 14.59, 45.43, 257.67])

cond = np.zeros((N,))
eig_min = np.zeros((N,))
err_k1 = np.zeros((N,))
L2_err_k = np.zeros((N,))
rel_L2_err_k = np.zeros((N,))
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))

for i in range(N):
    i = 4
    gap = int(h[i]/h0)
    # base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
    base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    m_fact = int(delta/h[i])
    
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    # ntrain = 100
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_x[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[:ntrain,::gap].reshape(-1, S)
    data_f = reader.read_field('bodyforce')[:ntrain,::gap].reshape(-1, S)
    
    ksi_range = torch.range(-m_fact, m_fact).int()
    # n_ksi = 2*m_fact+1
    ksi_range = ksi_range[ksi_range != 0]
    n_ksi = 2*m_fact
    data_ksi = ksi_range*h[i]
    data_eta = torch.zeros((ntrain, s, n_ksi))
    for ii in range(s):
        data_eta[:,ii,:] = (data_u[:,m_fact+ii+ksi_range].reshape(-1,1,n_ksi)-data_u[:,m_fact+ii].reshape(-1,1,1)).squeeze()
        
    # A = data_eta.reshape(-1, n_ksi)
    # print(np.linalg.cond(np.dot(A.T, A)))
    ksi_plus_eta = data_ksi+data_eta
    ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
    ksi_norm = torch.abs(data_ksi)
    extension = ksi_plus_eta_norm - ksi_norm
    # lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
    lambdaa = 1.0 + extension / (ksi_norm)
    
    
    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    print('min lambda: %s, max lambda: %s:' % (lambda_min_data, lambda_max_data))
    
    # lambda_min_data, lambda_max_data = 0.5064, 1.4935  # h=2**(-8)
    lambda_min, lambda_max = lambda_min_data[i]*1.1, lambda_max_data[i]*0.9
    
    # plot 
    N=100
    # xi_norm = torch.linspace(2**(-3),delta, N)
    xi_norm = torch.linspace(h[i],delta, N)
    dxi = xi_norm[1]-xi_norm[0]
    xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
    # lambdaa = torch.linspace(lambda_min_data,lambda_max_data, N)
    lambdaa = torch.linspace(lambda_min, lambda_max, N)
    lambdaa_cuda = lambdaa.unsqueeze(1).to('cuda')
    lambdaa_1_cuda = torch.ones_like(lambdaa_cuda)
    k1_NN = (model.phi_MGN(lambdaa_cuda)- model.phi_MGN(lambdaa_1_cuda))
    k2_NN = model.phi_2(xi_norm_cuda)
    k_NN =  k1_NN*k2_NN
    k_NN = k_NN.cpu().detach().numpy()
    # k_true =(lambdaa-lambdaa**(-3))*2*c/xi_norm*g_fun(xi_norm)
    k1_true = (lambdaa-lambdaa**(-3))
    k2_true = 2*c*g_fun(xi_norm)/xi_norm
    k_true = k1_true* k2_true
    
    dlambdaa = lambdaa[1]-lambdaa[0]
    weights = torch.ones((100,))
    weights[[0,-1]] = torch.tensor([1/2,1/2])
    coe_true = torch.sum(k1_true*weights)*dlambdaa
    k1_NN = torch.squeeze(k1_NN.cpu().detach())
    k2_NN = torch.squeeze(k2_NN.cpu().detach())
    coe_nn = torch.sum(k1_NN*weights)*dlambdaa
    
    k1_NN_normlized = k1_NN/coe_nn*coe_true
    k2_NN_normlized = k2_NN*coe_nn/coe_true
    # err_k1[i] = np.linalg.norm(k1_true-k1_NN_normlized.flatten())/np.linalg.norm(k1_true)
    
    
    # [Xi_norm, Lambdaa] = torch.meshgrid(xi_norm, lambdaa)
    # Xi_norm_cuda = Xi_norm.reshape(-1,1).to('cuda')   
    # Lambdaa_cuda = Lambdaa.reshape(-1,1).to('cuda')
    # Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    # dw = ((model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)) *model.phi_2(Xi_norm_cuda)).reshape(N,N)
    # dw = dw.cpu().detach().numpy()
    # dw_exact = (Lambdaa-Lambdaa**(-3))*2*c/Xi_norm*g_fun(Xi_norm)
    # L2_err_k[i] = torch.sqrt(torch.sum(dxi*dlambdaa*(dw_exact-dw)**2))
    # rel_L2_err_k[i] = np.linalg.norm(dw_exact-dw)/np.linalg.norm(dw_exact)


# print("L2 error for k2: %s" % L2_err_k2)   
# print("relative L2 error for k2: %s" % rel_L2_err_k2)

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(lambdaa, k1_true, color='k', linewidth=1.5, label='true') 
plt.plot(lambdaa, k1_NN_normlized, color='darkorange', linewidth=1.5, label='learned') 
plt.xlabel(r'$\lambda$', fontsize=fontsize)
plt.ylabel(r'$k_1(\lambda)$', fontsize=fontsize)
plt.title(r'learned $k_1(\lambda)$')
plt.legend()
plt.tight_layout()
# plt.savefig('%s/%s_k_h_%s_%s.png' % (base_dir, ex, h[i], act_xi), format='png')
plt.savefig('%s/%s_k1_ntrain_%s.png' % (current_dir, ex, ntrain), format='png')
plt.close()

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(xi_norm, k2_true, color='k', linewidth=1.5, label='true')  
plt.plot(xi_norm, k2_NN_normlized, color='darkorange', linewidth=1.5, label='learned')   
plt.xlabel(r'$|\xi|$', fontsize=fontsize)
plt.ylabel(r'$k_2(|\xi|)$', fontsize=fontsize)
plt.title(r'learned $k_2(\xi)$')
plt.legend()
plt.tight_layout()
# plt.savefig('%s/%s_k_h_%s_%s.png' % (base_dir, ex, h[i], act_xi), format='png')
plt.savefig('%s/%s_k2_ntrain_%s.png' % (current_dir, ex, ntrain), format='png')
plt.close()

   
