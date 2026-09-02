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
    act_fun_xi = torch.nn.ReLU()
elif act_xi == 'GELU': 
    act_fun_xi = torch.nn.GELU()
elif act_xi == 'Tanh': 
    act_fun_xi = torch.nn.Tanh()
elif act_xi == 'Softplus': 
    act_fun_xi = torch.nn.Softplus()

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
loss_train = np.zeros((N,))

for i in range(N):
    gap = int(h[i]/h0)
    # base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
    base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'train'))
    loss_train[i] = loss[-1,1]

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
    # lambda_min, lambda_max = lambda_min_data[i]*1.1, lambda_max_data[i]*0.9
    lambda_min, lambda_max = 0.7, 1.3
    
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
    
    err_gk = torch.norm(k_NN-k_true)/torch.norm(k_true)
    print(f'Error of gk: {err_gk:.4e}')
    
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
    
    

print("relative error for k: %s" % rel_L2_err_k)

###################################
# plot error and train loss on the same figure 
################################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax1 = plt.subplots(figsize = (6,5))
# color_err = 'dodgerblue'
color_err = 'tomato'
# plt.plot(h, L2_err_k2, 's--', markerfacecolor='none',color='darkorange', linewidth=2, label=r"$L^2$ error")
ax1.plot(h, rel_L2_err_k, 's--', markerfacecolor='none',color=color_err, linewidth=2)
# plt.plot(h, h/h[1]*L2_err_k2[1], 'k', label='slope=1', linewidth=2)
ax1.plot(h, (h/h[-1])**2*rel_L2_err_k[-1], 'k', label='slope=2', linewidth=2)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Mesh size')
ax1.set_ylabel(r'rel $L^2$ error')
ax1.tick_params(axis='y')
ax1.grid(True, which="both", ls="--", color='gray')
ax1.legend()

# color_loss = 'tomato'
# ax2 = ax1.twinx() 
# ax2.plot(h, loss_train, 'x--', color=color_loss, linewidth=2)
# ax2.set_yscale('log')
# ax2.set_ylabel('Loss value', color=color_loss)
# ax2.tick_params(axis='y', colors=color_loss)

# plt.title('nsr=0')   
plt.title(r'Learn $k(\lambda,\xi)$: error of $k(\lambda,\xi)$')          
plt.tight_layout()
name = '%s_k_error_k_ntrain_%s.png' % (ex, ntrain)
plt.savefig (os.path.join (current_dir, name))
plt.close()

