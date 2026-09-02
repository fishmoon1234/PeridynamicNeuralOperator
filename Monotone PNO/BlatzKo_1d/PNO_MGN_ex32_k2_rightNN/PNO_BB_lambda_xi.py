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

N = 4
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7)])
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
ntrain = 300

# model and training parameters
batch_size = 10
batch_size2 = batch_size
# layer_info = '128_4'
# phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
# phi_2_layer = [1, 128, 128, 128, 128, 1]
layer_info = '256_4'
phi_1_layer = [1, 10, 1]
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

# model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, alpha).to(device)
model = E_GCL_GKN(phi_1_layer, phi_2_layer, act_fun_xi).to(device)
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
g_fun = lambda x: x*np.exp(-50*x**2)*(delta-np.abs(x))


L2_err_k2 = np.zeros((N,))
rel_L2_err_k2 = np.zeros((N,))
Rho, Xi = [], []
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))
lambda_ave, Data_ksi, Weights = [], [], []
loss_train = np.zeros((N,))

colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))

for i in range(N):
    gap = int(h[i]/h0)
    base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
    # base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_lr%s_gap_%s' % (ex, layer_info, ntrain, lrs, lr, gap)
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'train'))
    loss_train[i] = loss[-1,1]

    m_fact = int(delta/h[i])
    
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_x[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[:ntrain,::gap].reshape(-1, S)
    data_f = reader.read_field('bodyforce')[:ntrain,::gap].reshape(-1, S)
    
    ksi_range = torch.range(-m_fact, m_fact).int()
    n_ksi = 2*m_fact+1
    # ksi_range = ksi_range[ksi_range != 0]
    # n_ksi = 2*m_fact
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
    lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
    # lambdaa = 1.0 + extension / (ksi_norm)
    
    Data_ksi.append(data_ksi)
    # lambda_ave.append(torch.mean(lambdaa, axis=[0,1]))
    index = 1
    lambda_ave.append(lambdaa[0,index,:])
    
    weights = torch.abs((lambdaa-lambdaa**(-3))*ksi_plus_eta/ksi_plus_eta_norm)
    rho = torch.sum(weights, axis=(0,1))/ntrain/s
    # rho = (torch.flip(rho[:m_fact], dims=[0])+rho[m_fact:])/2
    Rho.append(rho)
    Xi.append(data_ksi[m_fact:])
    Weights.append(weights)

    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    # print('min lambda: %s, max lambda: %s:' % (lambda_min_data, lambda_max_data))
    
    # lambda_min_data, lambda_max_data = 0.5064, 1.4935  # h=2**(-8)
    # lambda_min, lambda_max = 0.7, 1.3
    
    # plot 
    Nxi = int(1/h[i])+1
    # xi_norm = torch.linspace(h[-2],delta, N)
    xi_norm = torch.linspace(h[i], delta, Nxi)
    # xi_norm = torch.linspace(1e-8, delta, Nxi)
    # xi_norm = torch.linspace(h[-1], delta, Nxi)
    dxi = xi_norm[1]-xi_norm[0]
    xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
    k2_NN = model.phi_2(xi_norm_cuda)
    k2_NN = k2_NN.cpu().detach()
    k2_true = 2*c/xi_norm*g_fun(xi_norm)
    
    L2_err_k2[i] = torch.sqrt(torch.sum(dxi*(k2_NN.flatten()-k2_true)**2))
    rel_L2_err_k2[i] = L2_err_k2[i]/torch.sqrt(torch.sum(dxi*k2_true**2))
    
    plt.plot(xi_norm, k2_NN, color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))

plt.plot(xi_norm, k2_true, color='k', linewidth=1.5, label='true')    
plt.xlabel(r'$|\xi|$', fontsize=fontsize)
plt.ylabel(r'$k(\xi)$', fontsize=fontsize)
plt.title(r'learned $k(\xi)$')
plt.legend()
plt.tight_layout()
# plt.savefig('%s/%s_k_h_%s_%s.png' % (base_dir, ex, h[i], act_xi), format='png')
plt.savefig('%s/%s_k_%s.png' % (current_dir, ex, layer_info), format='png')
plt.close()
print('---')
   
print("L2 error for k2: %s" % L2_err_k2)   
print("relative L2 error for k2: %s" % rel_L2_err_k2)


################################################################# 
############# plot lambda(xi)
################################################################# 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
for i in range(N-1, N):
    ax.plot(Data_ksi[i], lambda_ave[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
ax.set_xlabel(r'$\xi$', fontsize=fontsize)
ax.set_ylabel(r'$\lambda(\xi)$', fontsize=fontsize)
ax.set_title(r'$\lambda(\xi)$, ntrain=%s'% ntrain)
ax.grid(True, which="both", ls="--", color='gray')
ax.legend(fontsize=12)
plt.tight_layout()
name = '%s_lambda_xi_%s.png' % (ex, index)
plt.savefig (os.path.join (current_dir, name))
plt.close()

