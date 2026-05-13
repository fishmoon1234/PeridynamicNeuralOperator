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

N=4
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
ntrain = 300


# model and training parameters
batch_size = 10
batch_size2 = batch_size
layer_info = '128_4_128_4'
phi_1_layer = [1, 128, 128, 128, 128, 1]
phi_2_layer = [1, 128, 128, 128, 128, 1]
# layer_info = '128_6_256_4'
# phi_1_layer = [1, 128, 128, 128, 128, 128, 128, 1]
# phi_2_layer = [1, 256, 256, 256, 256, 1]
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
    

gammas = [0.8]
scheduler_step = 100


current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex37'
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
# k_fun = lambda x: np.ones_like(x)
# k_fun = lambda x: x*(1-x**2)
# k_fun = lambda x: x*np.exp(-50*x**2)
# k_fun = lambda x: np.abs(x)*np.cos(3*np.pi*np.abs(x))
# k_fun = lambda x: x*(np.abs(x)-delta)
# k_fun = lambda x: x*np.exp(-50*x**2)*(delta-np.abs(x))
# k_fun = lambda x: 2*c*np.cos(np.pi*np.abs(x))
k_fun = lambda x: 2*c*np.cos(np.pi*np.abs(x))  
g_fun = lambda x: x-x**(-3)



rel_L2_err_k1 = np.zeros((N,))
rel_L2_rho_err_k1 = np.zeros((N,))
rel_L2_err_k2 = np.zeros((N,))
rel_L2_rho_err_k2 = np.zeros((N,))
L2_err_k = np.zeros((N,))
rel_L2_err_k = np.zeros((N,))
L2_rho_err_k = np.zeros((N,))
rel_L2_rho_err_k = np.zeros((N,))
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))
loss_train = np.zeros((N,))
loss_test = np.zeros((N,))
k1_all, k2_all = [], []
Rho_xi, Xi, Xi_plot = [], [], []
Rho_lambda, Lambda_plot = [], []
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']

for i in range(3, 4):
    gap = int(h[i]/h0)
    # lrs = [lrs_all[i]]
    # lr = [lr1_all[i], lr2_all[i]]
    lrs = [0.005]
    lr = [0.995, 0.998]
    normlize_type = 'k'
    
    layer_info = '128_4_128_4'
    phi_2_layer = [1, 128, 128, 128, 128, 1]
    embed_dim = 128
    num_layers = 4
    start_layer = 4
    # layer_add_times = 2

    # model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, alpha).to(device)
    model = E_GCL_GKN(num_layers, embed_dim, phi_2_layer, torch.nn.ReLU(), nn.Sigmoid()).to(device)

    # base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr%s_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
    base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_Sigmoid_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
    model_path = '%s/model_%s.ckpt' % (base_dir, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'train'))
    loss_test[i] = loss[-1,1]
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'test'))
    loss_test[i] = loss[-1,1]

    m_fact = int(delta/h[i])
    
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    # ntrain = 100
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_x[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[:ntrain,::gap].reshape(-1, S)
    data_f = reader.read_field('bodyforce')[:ntrain,::gap].reshape(-1, S)
    
    
    ksi_range = torch.arange(-m_fact, m_fact+1).int()
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
    print('min lambda: %s, max lambda: %s:' % (lambda_min_data[i], lambda_max_data[i]))
    
    
    ####  compute rho(xi)  #####
    dxi = data_ksi[1]-data_ksi[0]
    rho_xi = torch.abs(g_fun(lambdaa)*ksi_plus_eta/ksi_plus_eta_norm)
    rho_xi = torch.sum(rho_xi, axis=(0,1))/ntrain/s
    # rho = (torch.flip(rho0[:m_fact], dims=[0])+rho0[m_fact:])/2
    Rho_xi.append(rho_xi)
    Xi_plot.append(data_ksi)
    
    ####  compute rho(lambda)  #####
    total_nlambda = lambdaa.reshape(-1,).size(0)
    Nlambda = 50
    lambda_rho = torch.linspace(lambda_min_data[i], lambda_max_data[i], Nlambda)
    dlambda = lambda_rho[1]- lambda_rho[0]

    ksi_norm4rho = torch.tile(ksi_norm,(lambdaa.size(0),lambdaa.size(1),1))
    weights = torch.zeros((Nlambda,1))
    for k in range(Nlambda-1):
        indices = torch.nonzero((lambdaa >= lambda_rho[k]) & (lambdaa <= lambda_rho[k+1]), as_tuple=False)
        if indices.numel() != 0:
            index = (indices[:,0],indices[:,1],indices[:,2])
            weights[k] = torch.sum(torch.abs((k_fun(ksi_norm4rho[index]))*ksi_plus_eta[index]/ksi_plus_eta_norm[index]))/total_nlambda
        
    rho_lambda = weights.flatten()
    # rho = weights*dlambda_rho
    Rho_lambda.append(rho_lambda)
    Lambda_plot.append(lambda_rho)
    
    # lambda_min, lambda_max = lambda_min_data[i]*1.1, lambda_max_data[i]*0.9
    # lambda_min, lambda_max = 0.7, 1.3
    # lambda_min, lambda_max = 0.6, 1.4
    
    # plot 
    # Nxi = 100
    # xi_norm = torch.linspace(2**(-3),delta, N)
    # xi_norm = torch.linspace(h[i], delta, Nxi)
    # dxi = xi_norm[1] - xi_norm[0]
    xi_norm = torch.abs(data_ksi)
    xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
    lambda_rho_cuda = lambda_rho.unsqueeze(1).to('cuda')
    lambda_rho_1_cuda = torch.ones_like(lambda_rho_cuda)
    k1_NN = (model.phi_MGN(lambda_rho_cuda)- model.phi_MGN(lambda_rho_1_cuda)).cpu().detach()
    k2_NN = model.phi_2(xi_norm_cuda).cpu().detach()
    k1_true = g_fun(lambda_rho)
    k2_true = k_fun(xi_norm)
    
    
    if normlize_type == 'g':
        ### normlize g(lambda)
        weights = torch.ones((Nlambda,))
        weights[[0,-1]] = torch.tensor([1/2, 1/2])
        coe_true = torch.sum(k1_true*weights)*dlambda
        coe_nn = torch.sum(torch.squeeze(k1_NN)*weights)*dlambda
        coe_g = coe_true/coe_nn
        coe_k = 1/coe_g
    elif normlize_type == 'k':
        ### normlize k(xi)
        weights = torch.ones((n_ksi,))
        weights[[0, -1]] = torch.tensor([1/2, 1/2])
        coe_true = torch.sum(k2_true*weights)*dxi
        coe_nn = torch.sum(torch.squeeze(k2_NN)*weights)*dxi
        coe_k = coe_true/coe_nn
        coe_g = 1/coe_k
    

    k1_NN_normlized = k1_NN*coe_g
    rel_L2_err_k1[i] = np.linalg.norm(k1_true-k1_NN_normlized.flatten())/np.linalg.norm(k1_true)
    # rel_L2_rho_err_k1[i] = np.linalg.norm(rho_lambda*(k1_true-k1_NN_normlized.flatten()))/np.linalg.norm(rho_lambda*k1_true)
    rel_L2_rho_err_k1[i] = torch.sqrt(torch.sum(rho_lambda*(k1_true-k1_NN_normlized.flatten())**2))/torch.sqrt(torch.sum(rho_lambda*(k1_true)**2))
    
    
    k2_NN_normlized = k2_NN*coe_k
    rel_L2_err_k2[i] = np.linalg.norm((k2_true-k2_NN_normlized.flatten()))/np.linalg.norm(k2_true)
    # rel_L2_rho_err_k2[i] = np.linalg.norm(rho_xi*(k2_true-k2_NN_normlized.flatten()))/np.linalg.norm(rho_xi*k2_true)
    rel_L2_rho_err_k2[i] = torch.sqrt(torch.sum(rho_xi*(k2_true-k2_NN_normlized.flatten())**2))/torch.sqrt(torch.sum(rho_xi*(k2_true)**2))
    
    k1_all.append(k1_NN_normlized.cpu().detach().numpy())
    k2_all.append(k2_NN_normlized.cpu().detach().numpy())
    
    ################  compute error for gk  ####################
    [Xi_norm, Lambda_rho] = torch.meshgrid(xi_norm, lambda_rho)
    Xi_norm_cuda = Xi_norm.reshape(-1,1).to('cuda')   
    Lambda_rho_cuda = Lambda_rho.reshape(-1,1).to('cuda')
    Lambda_rho_1_cuda = torch.ones_like(Lambda_rho_cuda)
    gk = ((model.phi_MGN(Lambda_rho_cuda)-model.phi_MGN(Lambda_rho_1_cuda)) *model.phi_2(Xi_norm_cuda)).reshape(n_ksi, Nlambda).cpu().detach().numpy()
    gk_true = g_fun(Lambda_rho)*k_fun(Xi_norm)
    L2_err_k[i] = torch.sqrt(torch.sum(dxi*dlambda*(gk_true-gk)**2))
    rel_L2_err_k[i] = np.linalg.norm(gk_true-gk)/np.linalg.norm(gk_true)
    
    [RhoXi, RhoLambda] = torch.meshgrid(rho_xi, rho_lambda.flatten())
    L2_rho_err_k[i] = torch.sqrt(torch.sum(dxi*dlambda*(gk_true-gk)**2*RhoXi*RhoLambda))
    # rel_L2_rho_err_k[i] = np.linalg.norm((gk_true-gk)*RhoXi*RhoLambda)/np.linalg.norm((gk_true)*RhoXi*RhoLambda)
    rel_L2_rho_err_k[i] = L2_rho_err_k[i]/torch.sqrt(torch.sum(dxi*dlambda*(gk_true)**2*RhoXi*RhoLambda))

    
    
print("relative L2 rho error for k: %s" % rel_L2_rho_err_k)
print('Loss test: %s' % loss_test)

###################################
# plot error and train loss on the same figure 
################################### 
# fontsize = 15
# plt.rcParams.update({'font.size': fontsize}) 
# fig, ax1 = plt.subplots(figsize = (6,5))
# color_err = 'dodgerblue'
# plt.plot(h, rel_L2_rho_err_k1, 's--', markerfacecolor='none',color=colors[0], linewidth=2, label=r"rel $L^2(\rho)$ error of $g$")
# plt.plot(h, rel_L2_rho_err_k2, 's--', markerfacecolor='none',color=colors[1], linewidth=2, label=r"rel $L^2(\rho)$ error of $k$")
# # ax1.plot(h, rel_L2_err_k, 's--', markerfacecolor='none',color=color_err, linewidth=2, label=r"rel $L^2$ error of $gk$")
# ax1.plot(h, rel_L2_rho_err_k, 's--', markerfacecolor='none',color=colors[2], linewidth=2, label=r"rel $L^2(\rho)$ error of $gk$")
# plt.plot(h, (h/h[-1])*rel_L2_rho_err_k[-1], 'k', label='slope=1', linewidth=2)
# # ax1.plot(h, (h/h[-1])**2*rel_L2_rho_err_k[-1], 'k--', label='slope=2', linewidth=2)
# ax1.set_xscale('log', base=2)
# ax1.set_yscale('log')
# ax1.set_xlabel('Mesh size')
# # ax1.set_ylabel(r'rel $L^2$ error of $g(\lambda)k(\xi)$', color=color_err)
# ax1.tick_params(axis='y', colors=color_err)
# ax1.grid(True, which="both", ls="--", color='gray')
# ax1.legend(loc='upper left', fontsize=12)

# color_loss = 'tomato'
# ax2 = ax1.twinx() 
# ax2.plot(h, loss_train, 'x--', color=color_loss, linewidth=2)
# ax2.set_yscale('log')
# ax2.set_ylabel('Loss value', color=color_loss)
# ax2.tick_params(axis='y', colors=color_loss)

# plt.title(r'MGN: Learn $g(\lambda)k(\xi)$') 
# # plt.title(r'Learn $g(\lambda)k(\xi)$: $\lambda\in[%s,%s]$' % (lambda_min, lambda_max))            
# plt.tight_layout()
# name = '%s_error_gk_%s_%s_norm_%s.png' % (ex, layer_info, lr[0], normlize_type)
# plt.savefig (os.path.join (current_dir, name))
plt.close()


###################################
# plot g(lambda) and k(xi)
################################### 
fontsize = 25
linewidth = 2.5
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
plt.plot(Lambda_plot[-1], k1_true, color='k', linewidth=linewidth, label='True') 
# for i in range(N):
#     plt.plot(Lambda_plot[i], k1_all[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
plt.plot(Lambda_plot[0], k1_all[0], color='deepskyblue', linewidth=linewidth, label='Learned')
plt.xlabel(r'$\lambda$', fontsize=fontsize)
plt.ylabel(r'$g(\lambda)$', fontsize=fontsize)
# plt.yscale('log')
# plt.title(r'Learned $g(\lambda)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.tight_layout()
name = '%s_g_%s_one.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name), format='png', dpi=300)
plt.close()


plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
plt.plot(Xi_plot[-1], k2_true, color='k', linewidth=linewidth, label='True') 
# for i in range(N):
#     plt.plot(Xi_plot[i], k2_all[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
plt.plot(Xi_plot[0], k2_all[0], color='mediumslateblue', linewidth=linewidth, label='Learned')
plt.xlabel(r'$\xi$', fontsize=fontsize)
plt.ylabel(r'$k(\xi)$', fontsize=fontsize)
# plt.yscale('log')
# plt.title(r'Learned $k(\xi)$')
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.tight_layout()
name = '%s_k_%s_one.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name), format='png', dpi=300)
plt.close()
