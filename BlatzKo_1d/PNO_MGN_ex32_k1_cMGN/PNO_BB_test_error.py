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

lrs = [0.005]
lr = [0.995, 0.998]

# model and training parameters
batch_size = 10
batch_size2 = batch_size
    
# layer_info = '32_8'
# embed_dim = 32
# num_layers = 8
layer_info = '128_4'
embed_dim = 128
num_layers = 4
model = E_GCL_GKN(num_layers, 1, embed_dim).to(device)


current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex32'
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
# k_fun = lambda x: np.ones_like(x)
# k_fun = lambda x: x*(1-x**2)
# k_fun = lambda x: x*np.exp(-50*x**2)
# k_fun = lambda x: np.abs(x)*np.sin(3*np.pi*np.abs(x))
# k_fun = lambda x: x*(np.abs(x)-delta)
k_fun = lambda x: 2*c*np.exp(-50*x**2)*(delta-np.abs(x))
# k_fun = lambda x: 2*c*np.cos(np.pi*np.abs(x))
g_fun = lambda x: x-x**(-3)


L2_err_k1 = np.zeros((N,))
rel_L2_err_k1 = np.zeros((N,))
L2_rho_err_k1 = np.zeros((N,))
rel_L2_rho_err_k1 = np.zeros((N,))
Rho, Xi, Lambda_rho_all = [], [], []
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))
lambda_ave, Data_ksi, Weights = [], [], []
loss_train = np.zeros((N,))
loss_test = np.zeros((N,))
ATA_cond = np.zeros((N,))

linestyles = ['--', '-.', ':', (0, (3, 5, 1, 5))] 
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 22
# Configure global plot styling
plt.rcParams.update({
    'font.size': fontsize,
    'font.family': 'serif',  # Use serif font for better readability in publications
    'axes.linewidth': 1.2,   # Border line width
    'axes.edgecolor': 'gray', # Border color
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize+2,  # Title slightly larger than labels
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'grid.linewidth': 0.8,   # Grid line thickness
    'grid.linestyle': '-',   # Solid grid lines
    'grid.alpha': 0.5,       # Grid transparency (50%)
    'grid.color': 'gray',    # Grid color
    'legend.fontsize': fontsize-4,  # Slightly smaller legend text
    'legend.frameon': True,  # Show legend border
    'legend.framealpha': 0.8,  # 95% opaque legend background
    'legend.fancybox': True  # Rounded legend corners
    # 'legend.shadow': True    # Add shadow effect to legend
})

# Create figure with specified size (width, height) in inches
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(N):
    gap = int(h[i]/h0)
    # layer_index = 6
    # model = E_GCL_GKN(phi_1_layer, layer_index).to(device)
    
        
    base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_sigmoid_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
    model_path = '%s/model_%s.ckpt' % (base_dir, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'train'))
    loss_train[i] = loss[-1,1]
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
    
    Data_ksi.append(data_ksi)
    lambda_ave.append(torch.mean(lambdaa, axis=[0,1]))
    
    total_nlambda = lambdaa.reshape(-1,).size(0)
    
    # lambda_min_all_data, lambda_max_all_data = 0.5, 1.5
    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    Nlambda = 100
    Lambda_rho = torch.linspace(lambda_min_data[i], lambda_max_data[i], Nlambda)
    dlambda_rho = Lambda_rho[1]- Lambda_rho[0]
        
    # weights = torch.abs((lambdaa-lambdaa**(-3))*ksi_plus_eta/ksi_plus_eta_norm)
    # weights = torch.abs((k_fun(ksi_norm)/ksi_norm)*ksi_plus_eta/ksi_plus_eta_norm)
    ksi_norm4rho = torch.tile(ksi_norm,(lambdaa.size(0),lambdaa.size(1),1))
    weights = torch.zeros((Nlambda,1))
    for k in range(Nlambda-1):
        indices = torch.nonzero((lambdaa >= Lambda_rho[k]) & (lambdaa <= Lambda_rho[k+1]), as_tuple=False)
        if indices.numel() != 0:
            index = (indices[:,0],indices[:,1],indices[:,2])
            weights[k] = torch.sum(torch.abs((k_fun(ksi_norm4rho[index]))*ksi_plus_eta[index]/ksi_plus_eta_norm[index]))/total_nlambda
        
    
    # rho = torch.sum(weights, axis=(0,1))/ntrain/s
    # rho = (torch.flip(rho[:m_fact], dims=[0])+rho[m_fact:])/2
    rho = weights
    # rho = weights*dlambda_rho
    Rho.append(rho)
    Lambda_rho_all.append(Lambda_rho)
    # Xi.append(data_ksi[m_fact:])
    # Weights.append(weights)
    
    # A = weights*dlambda_rho
    # ATA = np.dot(A.T, A)
    # # ATA_inv = np.linalg.pinv(ATA)  
    # # eig_min, _ = np.linalg.eig(ATA)
    # # A_L2 = np.linalg.norm(np.linalg.pinv(A), ord=2)
    # # A_cond = np.linalg.cond(A)
    # ATA_cond[i] = np.linalg.cond(ATA)
    # # # ATAinvAT_L2 = np.linalg.norm(np.dot(ATA_inv, A.T), ord=2)
    # print(ATA_cond[i])
    # print(np.linalg.cond(A))
    
    
    Lambdaa = Lambda_rho.reshape(-1,1)
    Lambdaa_cuda = Lambdaa.to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    # k1_true = 2*c*(Lambda_rho-Lambda_rho**(-3)).reshape(-1,1)
    k1_true = g_fun(Lambdaa)
    k1_NN = (model.phi_MGN(Lambdaa_cuda)- model.phi_MGN(Lambdaa_1_cuda)).reshape(-1,1)
    k1_NN = k1_NN.cpu().detach()
    # weights = weights.reshape(-1,n_ksi)
    L2_rho_err_k1[i] = torch.sqrt(torch.sum(weights*dlambda_rho*(k1_NN-k1_true)**2))
    rel_L2_rho_err_k1[i] = L2_rho_err_k1[i]/torch.sqrt(torch.sum(weights*dlambda_rho*k1_true**2))


    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    print('min lambda: %s, max lambda: %s:' % (lambda_min_data[i], lambda_max_data[i]))
    
    # lambda_min_data, lambda_max_data = 0.5064, 1.4935  # h=2**(-8)
    # lambda_min, lambda_max = lambda_min_data[i]*1.1, lambda_max_data[i]*0.9
    # lambda_min, lambda_max = 0.7, 1.3
    # lambda_min, lambda_max = 0.8, 1.4
    lambda_min, lambda_max = lambda_min_data[i], lambda_max_data[i]
    
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
    k1_true = g_fun(Lambdaa)
    k1_NN = (model.phi_MGN(Lambdaa_cuda)- model.phi_MGN(Lambdaa_1_cuda))
    k1_NN = k1_NN.cpu().detach()
    # k2_NN = model.phi_2(xi_norm_cuda)
    # k2_NN = k2_NN.cpu().detach()
    # k2_true = 2*c/xi_norm*k_fun(xi_norm)
    
    L2_err_k1[i] = torch.sqrt(torch.sum(dlambda*(k1_NN.flatten()-k1_true)**2))
    rel_L2_err_k1[i] = L2_err_k1[i]/torch.sqrt(torch.sum(dlambda*k1_true**2))
    
    
    plt.plot(Lambdaa, k1_NN, color=colors[i], linewidth=2.5, label=r'$\Delta x=2^{-%s}$' % (i+i0), linestyle=linestyles[i])

ntrain = 300
plt.plot(Lambdaa, k1_true, color='k', linewidth=2.5, label='True', linestyle='-')    
plt.xlabel(r'$\lambda$', fontsize=fontsize+5)
# plt.ylabel(r'$g(\lambda)$', fontsize=fontsize)
plt.title(r'(Ex-II) Learned $g(\lambda)$')
plt.legend(loc='best')
ax.grid(True)
plt.tight_layout()
# plt.savefig('%s/%s_k_h_%s_%s.png' % (base_dir, ex, h[i], act_xi), format='png')
plt.savefig('%s/%s_g_%s.png' % (current_dir, ex, layer_info), format='png', dpi=300)
plt.close()
print('---')
   
print("L2 error for k1: %s" % L2_err_k1)   
print("relative L2 error for k1: %s" % rel_L2_err_k1)
print("L2 rho error for k1: %s" % L2_rho_err_k1)   
print("relative L2 rho error for k1: %s" % rel_L2_rho_err_k1)
print(loss_test)



###################################
# plot error and train loss on the same figure 
################################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax1 = plt.subplots(figsize = (6,5))
color_err = 'dodgerblue'
# ax1.plot(h, L2_rho_err_k1, 's--', markerfacecolor='none',color='orange', linewidth=2, label=r"$L^2(\rho)$ error")
# ax1.plot(h, rel_L2_rho_err_k1, 'o--', markerfacecolor='none',color='palevioletred', linewidth=2, label=r"rel $L^2(\rho)$ error")
ax1.plot(h, rel_L2_rho_err_k1, 's--', markerfacecolor='none',color=color_err, linewidth=2)
# plt.plot(h, h/h[-1]*rel_L2_rho_err_k1[-1], 'k', label='slope=1', linewidth=2)
ax1.plot(h, (h/h[-1])**2*rel_L2_rho_err_k1[-1], 'k', label='slope=2', linewidth=2)
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Mesh size')
ax1.set_ylabel(r'rel $L^2(\rho)$ error of $g(\lambda)$', color=color_err)
ax1.tick_params(axis='y', colors=color_err)
ax1.grid(True, which="both", ls="--", color='gray')
ax1.legend()

color_loss = 'tomato'
ax2 = ax1.twinx() 
ax2.plot(h, loss_train, 'x--', color=color_loss, linewidth=2)
ax2.set_yscale('log')
ax2.set_ylabel('Loss value', color=color_loss)
ax2.tick_params(axis='y', colors=color_loss)

plt.title(r'Learn $g(\lambda)$') 
# plt.title(r'Learn $g(\lambda)$: $\lambda\in[%s, %s]$' % (lambda_min, lambda_max))            
plt.tight_layout()
name = '%s_error_g_%s.png' % (ex, layer_info)
plt.savefig (os.path.join (current_dir, name))
plt.close()



####################### 
# plot rho(\lambda)
####################### 
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
for i in range(N):
    plt.plot(Lambda_rho_all[i], Rho[i], color=colors[i], linewidth=1.5, label=r'$h=2^{-%s}$' % (i+i0))
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
# plot loss
####################### 
path = os.path.join(current_dir, 'Results')
os.makedirs(path, exist_ok=True)  
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
start = 30
# colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred']
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']

Loss_train = np.zeros((N,))
loss_name = ["train", "valid", "test"]
for j in range(3):
    fontsize = 15
    plt.rcParams.update({'font.size': fontsize}) 
    fig, ax = plt.subplots(figsize = (6,6))
    start = 30
    # colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'gray']
    for i in range(N):
        gap = int(h[i]/h0)
        # base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
        base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_gap_%s' % (ex, layer_info, ntrain, lrs, lr, gap)
        base_dir = os.path.join(current_dir, base_dir) 
        loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , loss_name[j]))
        
        plt.plot(loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
    
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss %s' % (loss_name[j]), fontsize=fontsize)
    plt.yscale('log')
    # plt.title('noise=%s'% noise_std)
    plt.grid(True, which="both", ls="--", color='gray')
    plt.legend()
    plt.show()
    plt.tight_layout()
    name = '%s_loss_%s_ntrain_%s_g.png' % (ex, loss_name[j], ntrain)
    plt.savefig (os.path.join (path, name))
    print('---')