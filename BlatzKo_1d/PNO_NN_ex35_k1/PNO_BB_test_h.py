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

N = 5
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# h = np.array([2**(-5), 2**(-6), 2**(-7)])
h = np.array([2**(-4),2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 4
h0 = 2**(-8)
ntrain = 300

# model and training parameters
batch_size = 10
batch_size2 = batch_size
# layer_info = '64_5'
# phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
# phi_2_layer = [1, 10, 10, 1]
layer_info = '128_5'
phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
phi_2_layer = [1, 10, 10, 1]
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

# model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, alpha).to(device)
model = E_GCL_GKN(phi_1_layer, phi_2_layer, act_fun_xi).to(device)
current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex22'
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
# g_fun = lambda x: np.ones_like(x)
# g_fun = lambda x: x*(1-x**2)
g_fun = lambda x: x*np.exp(-50*x**2)
# g_fun = lambda x: x*(np.abs(x)-delta)


lambda_min_all_data, lambda_max_all_data = 0.5, 1.5
Nlambda = 50
Lambda_rho = torch.linspace(lambda_min_all_data, lambda_max_all_data, Nlambda)
dlambda_rho = Lambda_rho[1]- Lambda_rho[0]

L2_err_k1 = np.zeros((N,))
rel_L2_err_k1 = np.zeros((N,))
L2_rho_err_k1 = np.zeros((N,))
rel_L2_rho_err_k1 = np.zeros((N,))
Rho, Xi = [], []
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))
lambda_ave, Data_ksi, Weights = [], [], []

ATA_cond = np.zeros((N,))

colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))

for i in range(N):
    gap = int(h[i]/h0)
    # base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
    base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, 300, batch_size, act_xi, gap)
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
    
    ksi_range = torch.arange(-m_fact, m_fact + 1, 1).int()
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
    
    Data_ksi.append(data_ksi)
    lambda_ave.append(torch.mean(lambdaa, axis=[0,1]))
    
    # weights = torch.abs((lambdaa-lambdaa**(-3))*ksi_plus_eta/ksi_plus_eta_norm)
    # weights = torch.abs((g_fun(ksi_norm)/ksi_norm)*ksi_plus_eta/ksi_plus_eta_norm)
    ksi_norm4rho = torch.tile(ksi_norm,(lambdaa.size(0),lambdaa.size(1),1))
    weights = torch.zeros((Nlambda,1))
    for k in range(Nlambda-1):
        indices = torch.nonzero((lambdaa >= Lambda_rho[k]) & (lambdaa <= Lambda_rho[k+1]), as_tuple=False)
        if indices.numel() != 0:
            index = (indices[:,0],indices[:,1],indices[:,2])
            weights[k] = torch.mean(torch.abs((g_fun(ksi_norm4rho[index])/ksi_norm4rho[index])*ksi_plus_eta[index]/ksi_plus_eta_norm[index]))
        
    
    # rho = torch.sum(weights, axis=(0,1))/ntrain/s
    # rho = (torch.flip(rho[:m_fact], dims=[0])+rho[m_fact:])/2
    rho = weights
    # rho = weights*dlambda_rho
    Rho.append(rho)
    # Xi.append(data_ksi[m_fact:])
    # Weights.append(weights)
    
    # A = weights*dlambda_rho
    # ATA = np.dot(A.T, A)
    # # ATA_inv = np.linalg.pinv(ATA)  
    # # eig_min, _ = np.linalg.eig(ATA)
    # A_L2 = np.linalg.norm(np.linalg.pinv(A), ord=2)
    # A_cond = np.linalg.cond(A)
    ATA_cond[i] = np.linalg.cond(ATA)
    # # ATAinvAT_L2 = np.linalg.norm(np.dot(ATA_inv, A.T), ord=2)
    
    
    Lambdaa = Lambda_rho.reshape(-1,1)
    Lambdaa_cuda = Lambdaa.to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    k1_true = 2*c*(Lambda_rho-Lambda_rho**(-3)).reshape(-1,1)
    k1_NN = (model.phi_MGN(Lambdaa_cuda)- model.phi_MGN(Lambdaa_1_cuda)).reshape(-1,1)
    k1_NN = k1_NN.cpu().detach()
    # weights = weights.reshape(-1,n_ksi)
    L2_rho_err_k1[i] = torch.sqrt(torch.sum(weights*dlambda_rho*(k1_NN-k1_true)**2))
    rel_L2_rho_err_k1[i] = L2_rho_err_k1[i]/torch.sqrt(torch.sum(weights*dlambda_rho*k1_true**2))


    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    print('min lambda: %s, max lambda: %s:' % (lambda_min_data, lambda_max_data))
    
    # lambda_min_data, lambda_max_data = 0.5064, 1.4935  # h=2**(-8)
    lambda_min, lambda_max = lambda_min_data[i]*1.1, lambda_max_data[i]*0.9
    
    # plot 
    # Nxi=100
    # # xi_norm = torch.linspace(h[i],delta, N)
    # xi_norm = torch.linspace(h[i],delta, Nxi)
    # dxi = xi_norm[1]-xi_norm[0]
    # xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
    Lambdaa = torch.linspace(lambda_min, lambda_max, 100)
    dlambda = Lambdaa[1]-Lambdaa[0]
    Lambdaa_cuda = Lambdaa.unsqueeze(1).to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    k1_true = 2*c*(Lambdaa-Lambdaa**(-3))
    k1_NN = (model.phi_MGN(Lambdaa_cuda)- model.phi_MGN(Lambdaa_1_cuda))
    k1_NN = k1_NN.cpu().detach()
    # k2_NN = model.phi_2(xi_norm_cuda)
    # k2_NN = k2_NN.cpu().detach()
    # k2_true = 2*c/xi_norm*g_fun(xi_norm)
    
    L2_err_k1[i] = torch.sqrt(torch.sum(dlambda*(k1_NN.flatten()-k1_true)**2))
    rel_L2_err_k1[i] = L2_err_k1[i]/torch.sqrt(torch.sum(dlambda*k1_true**2))
    
    
    plt.plot(Lambdaa, k1_NN, color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))

ntrain = 300
plt.plot(Lambdaa, k1_true, color='k', linewidth=1.5, label='true')    
plt.xlabel(r'$\lambda$', fontsize=fontsize)
plt.ylabel(r'$k_1(\lambda)$', fontsize=fontsize)
plt.title(r'learned $k_1(\lambda)$, ntrain=%s'% ntrain)
plt.legend()
plt.tight_layout()
# plt.savefig('%s/%s_k_h_%s_%s.png' % (base_dir, ex, h[i], act_xi), format='png')
plt.savefig('%s/%s_k1_ntrain_%s.png' % (current_dir, ex, ntrain), format='png')
plt.close()
print('---')
   
print("L2 error for k1: %s" % L2_err_k1)   
print("relative L2 error for k1: %s" % rel_L2_err_k1)


fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, L2_err_k1, 's--', markerfacecolor='none',color='darkorange', linewidth=2, label=r"$L^2$ error")
plt.plot(h, rel_L2_err_k1, 's--', markerfacecolor='none',color='yellowgreen', linewidth=2, label=r"rel $L^2$ error")
plt.plot(h, L2_rho_err_k1, 's--', markerfacecolor='none',color='deepskyblue', linewidth=2, label=r"$L^2(\rho)$ error")
plt.plot(h, rel_L2_rho_err_k1, 's--', markerfacecolor='none',color='mediumslateblue', linewidth=2, label=r"rel $L^2(\rho)$ error")
plt.plot(h, h/h[1]*rel_L2_err_k1[1], 'k', label='slope=1', linewidth=2)
plt.plot(h, (h/h[1])**2*rel_L2_err_k1[1], 'k--', label='slope=2', linewidth=2)
plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
# plt.ylabel(r'Relative $L^2$ error of $k_2(\xi)$', fontsize=fontsize)
# plt.title(r'$\lambda\in$[%s,%s]' % (lambda_min, lambda_max), fontsize=14)
plt.title(r'error of $k_1(\lambda)$, ntrain=%s'% ntrain)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend(fontsize=12)
plt.show()
plt.tight_layout()
name = '%s_error_k1_ntrain_%s_%s_h.png' % (ex, ntrain, act_xi)
plt.savefig (os.path.join (current_dir, name))
plt.close()


####################### 
# plot condition number of ATA
####################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, ATA_cond, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
# plt.plot(h, (h/h[-1])**(-4)*cond[-1], 'k', label='slope=1', linewidth=2)
# plt.plot(h, (h/h[-1])**(-3)*cond[-1], 'b', label='slope=2', linewidth=2)
# plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
# plt.ylabel(r'Relative $L^2$ error of $k(|\xi|)$', fontsize=fontsize)
plt.title(r'learn $k_1(\lambda)$: condition number of $A^TA$')
plt.grid(True, which="both", ls="--", color='gray')
# plt.legend()
plt.tight_layout()
name = '%s_cond_k1_ntrain_%s.png' % (ex, ntrain)
plt.savefig (os.path.join (current_dir, name))
plt.close()


####################### 
# plot rho
####################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
for i in range(N):
    plt.plot(Lambda_rho, Rho[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
plt.xlabel(r'$\lambda$', fontsize=fontsize)
plt.ylabel(r'$\rho$', fontsize=fontsize)
# plt.yscale('log')
plt.title(r'measure $\rho(\lambda)$, ntrain=%s'% ntrain)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.tight_layout()
name = '%s_rho_lambda_ntrain_%s.png' % (ex, ntrain)
plt.savefig (os.path.join (current_dir, name))
plt.close()

####################### 
# plot range of lambda for different mesh size
####################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, lambda_min_data, 's--', markerfacecolor='none',color='darkorange', linewidth=2, label=r"minimum of $\lambda$")
plt.plot(h, lambda_max_data, 's--', markerfacecolor='none',color='yellowgreen', linewidth=2, label=r"maximum of $\lambda$")
plt.xlabel(r'$h$', fontsize=fontsize)
plt.ylabel(r'$\lambda$', fontsize=fontsize)
plt.gca().invert_xaxis() 
# plt.yscale('log')
plt.title(r'range of $\lambda$, ntrain=%s'% ntrain)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.tight_layout()
name = '%s_lambda_ntrain_%s.png' % (ex, ntrain)
plt.savefig (os.path.join (current_dir, name))
plt.close()

####################### 
# plot lambda(xi)
####################### 
# fontsize = 15
# plt.rcParams.update({'font.size': fontsize}) 
# fig, ax = plt.subplots(1, 4, figsize = (20,5))
# colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
# for i in range(N):
#     ax[0].plot(Xi[i], Rho[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
# ax[0].set_xlabel(r'$\xi$', fontsize=fontsize)
# ax[0].set_ylabel(r'$\rho$', fontsize=fontsize)
# ax[0].set_yscale('log')
# ax[0].set_title(r'measure $\rho(\xi)$, ntrain=%s'% ntrain)
# ax[0].grid(True, which="both", ls="--", color='gray')
# ax[0].legend(fontsize=12)

# for i in range(N):
#     k1 = lambda_ave[i]-lambda_ave[i]**(-3)
#     ax[1].plot(Data_ksi[i], k1, color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
#     # ax[0].plot(data_ksi, torch.mean(k1, axis=[0,1]),color='darkorange', linewidth=2)
# ax[1].set_xlabel(r'$\xi$', fontsize=fontsize)
# ax[1].set_ylabel(r'$k_1(\lambda(\xi))$', fontsize=fontsize)
# ax[1].set_title(r'$k_1(\lambda(\xi))$, ntrain=%s'% ntrain)
# # plt.yscale('log')
# ax[1].grid(True, which="both", ls="--", color='gray')
# ax[1].legend(fontsize=12)

# for i in range(N):
#     ax[2].plot(Data_ksi[i], lambda_ave[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
# ax[2].set_xlabel(r'$\xi$', fontsize=fontsize)
# ax[2].set_ylabel(r'$\lambda(\xi)$', fontsize=fontsize)
# ax[2].set_title(r'$\lambda(\xi)$, ntrain=%s'% ntrain)
# ax[2].grid(True, which="both", ls="--", color='gray')
# ax[2].legend(fontsize=12)

# lambdaa = np.linspace(0.5,1.5,100)
# k1 = lambdaa-lambdaa**(-3)
# ax[3].plot(lambdaa, k1, color='darkorange', linewidth=2)
# ax[3].set_xlabel(r'$\lambda$', fontsize=fontsize)
# ax[3].set_ylabel(r'$k_1(\lambda)$', fontsize=fontsize)
# # plt.yscale('log')
# ax[3].set_title(r'$k_1(\lambda)$, ntrain=%s'% ntrain)
# ax[3].grid(True, which="both", ls="--", color='gray')
# # plt.legend()
# plt.tight_layout()

# name = '%s_rho_ntrain_%s.png' % (ex, ntrain)
# plt.savefig (os.path.join (current_dir, name))
# plt.close()

####################### 
# plot loss
#######################
Loss_train = np.zeros((N,))
loss_name = ["train", "valid", "test"]
for j in range(3):
    fontsize = 15
    plt.rcParams.update({'font.size': fontsize}) 
    fig, ax = plt.subplots(figsize = (6,6))
    start = 10
    # colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'gray']
    for i in range(N):
        gap = int(h[i]/h0)
        base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
        base_dir = os.path.join(current_dir, base_dir) 
        loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , loss_name[j]))
        # valid_loss = np.loadtxt('%s/loss_valid.txt' % (base_dir))
        # test_loss = np.loadtxt('%s/loss_test.txt' % (base_dir))
        
        
        plt.plot(loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
        # plt.plot(valid_loss[start:,0], valid_loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
        # plt.plot(test_loss[start:,0], test_loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
        # plt.plot(valid_loss[start:,0], valid_loss[start:,1], color='mediumorchid', linewidth=1.5, label='slope=1')
        # Loss[i] = loss[-1,1]
    
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss %s' % (loss_name[j]), fontsize=fontsize)
    # plt.ylabel(r'Training loss', fontsize=fontsize)
    # plt.ylabel(r'Valid loss', fontsize=fontsize)
    # plt.ylabel(r'Test loss', fontsize=fontsize)
    plt.yscale('log')
    # plt.title('noise=%s'% noise_std)
    plt.grid(True, which="both", ls="--", color='gray')
    plt.legend()
    plt.show()
    plt.tight_layout()
    name = '%s_loss_%s_ntrain_%s_k2.png' % (ex, loss_name[j], ntrain)
    plt.savefig (os.path.join (current_dir, name))
