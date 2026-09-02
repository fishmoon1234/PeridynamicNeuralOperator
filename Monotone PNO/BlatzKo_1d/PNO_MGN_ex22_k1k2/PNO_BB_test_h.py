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

N=5
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h0 = 2**(-8)
ntrain = 300


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

err_k1 = np.zeros((N,))
L2_err_k = np.zeros((N,))
rel_L2_err_k = np.zeros((N,))
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))

for i in range(N):
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
    k_true =(lambdaa-lambdaa**(-3))*2*c/xi_norm*g_fun(xi_norm)
    k1_true = (lambdaa-lambdaa**(-3))
    
    dlambdaa = lambdaa[1]-lambdaa[0]
    weights = torch.ones((100,))
    weights[[0,-1]] = torch.tensor([1/2,1/2])
    coe_true = torch.sum(k1_true*weights)*dlambdaa
    k1_NN = torch.squeeze(k1_NN.cpu().detach())
    coe_nn = torch.sum(k1_NN*weights)*dlambdaa
    
    dw1_normlized = k1_NN/coe_nn*coe_true
    err_k1[i] = np.linalg.norm(k1_true-dw1_normlized.flatten())/np.linalg.norm(k1_true)
    
    
    [Xi_norm, Lambdaa] = torch.meshgrid(xi_norm, lambdaa)
    Xi_norm_cuda = Xi_norm.reshape(-1,1).to('cuda')   
    Lambdaa_cuda = Lambdaa.reshape(-1,1).to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    dw = ((model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)) *model.phi_2(Xi_norm_cuda)).reshape(N,N)
    dw = dw.cpu().detach().numpy()
    dw_exact = (Lambdaa-Lambdaa**(-3))*2*c/Xi_norm*g_fun(Xi_norm)
    L2_err_k[i] = torch.sqrt(torch.sum(dxi*dlambdaa*(dw_exact-dw)**2))
    rel_L2_err_k[i] = np.linalg.norm(dw_exact-dw)/np.linalg.norm(dw_exact)
    
    
    
print("relative error for k1: %s" % err_k1)
print("relative error for k: %s" % L2_err_k)

# fontsize = 15
# plt.rcParams.update({'font.size': fontsize}) 
# fig, ax = plt.subplots(figsize = (6,5))
# plt.plot(h, err_k1, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
# plt.plot(h, h/h[1]*err_k1[1], 'k', label='slope=1', linewidth=2)
# plt.plot(h, (h/h[1])**2*err_k1[1], 'k--', label='slope=2', linewidth=2)
# plt.gca().invert_xaxis() 
# plt.xscale('log', base=2)
# plt.yscale('log')
# plt.xlabel('Mesh size', fontsize=fontsize)
# plt.ylabel(r'Relative $L^2$ error of normalized $k1(\lambda)$', fontsize=fontsize)
# # show the figure
# plt.title(r'error of $k_1(\lambda)$, $\lambda\in$[%s,%s]' % (lambda_min, lambda_max), fontsize=14)
# plt.grid(True, which="both", ls="--", color='gray')
# plt.legend()
# plt.show()
# plt.tight_layout()
# name = '%s_error_k1_ntrain_%s_lambda_%s_%s_h_%s.png' % (ex, ntrain, lambda_min, lambda_max, act_xi)
# plt.savefig (os.path.join (current_dir, name))
# plt.close()

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
# plt.plot(h, err_k, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
# plt.plot(h, h/h[0]*err_k[0], 'k', label='slope=1', linewidth=2)
plt.plot(h, L2_err_k, 's--', markerfacecolor='none',color='darkorange', linewidth=2, label=r"$L^2$ error")
plt.plot(h, rel_L2_err_k, 's--', markerfacecolor='none',color='yellowgreen', linewidth=2, label=r"rel $L^2$ error")
plt.plot(h, h/h[1]*L2_err_k[1], 'k', label='slope=1', linewidth=2)
plt.plot(h, (h/h[1])**2*L2_err_k[1], 'k--', label='slope=2', linewidth=2)
plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
# plt.ylabel(r'Relative $L^2$ error of $k(\lambda,\xi)$', fontsize=fontsize)
# show the figure
plt.title(r'error of $k(\lambda, \xi)$, $\lambda\in$[%s,%s]' % (lambda_min, lambda_max), fontsize=15)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.show()
plt.tight_layout()
name = '%s_error_k_ntrain_%s_lambda_%s_%s_h_%s.png' % (ex, ntrain, lambda_min, lambda_max, act_xi)
plt.savefig (os.path.join (current_dir, name))
plt.close()


####################### 
# plot loss
#######################
Loss_train = np.zeros((N,))
loss_name = ["train", "valid", "test"]
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
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