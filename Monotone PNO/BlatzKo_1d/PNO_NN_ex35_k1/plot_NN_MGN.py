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
current_dir = os.path.dirname(os.path.realpath(__file__))
N = 4
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# h = np.array([2**(-5), 2**(-6), 2**(-7)])
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
ntrain = 300

lrs = [0.01]
lr = [0.99, 0.998]

# model and training parameters
batch_size = 10
batch_size2 = batch_size
# layer_info = '64_5'
# phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
# phi_2_layer = [1, 10, 10, 1]
layer_info = '128_5'
phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
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

model_NN = E_GCL_GKN(phi_1_layer, act_fun_xi).to(device)
total_params = sum(p.numel() for p in model_NN.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

layer_info = '128_5'
phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
model_MGN = E_GCL_GKN_MGN(phi_1_layer, nn.Sigmoid()).to(device)
model_MGN_path = '%s/../PNO_MGN_ex35_k1/Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.995, 0.998]_gap_1' % current_dir
model_path = os.path.join(model_MGN_path, 'model.ckpt')
model_MGN.load_state_dict(torch.load(model_path))
model_MGN.eval()
total_params = sum(p.numel() for p in model_MGN.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

ex = 'ex35'
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
g_fun = lambda x: np.exp(-50*x**2)*(delta-np.abs(x))
k1_fun = lambda x: np.pi*(x - x**(-3)) + np.sin(np.pi*x)


L2_err_k1 = np.zeros((N,))
rel_L2_err_k1 = np.zeros((N,))
L2_rho_err_k1 = np.zeros((N,))
rel_L2_rho_err_k1 = np.zeros((N,))
Rho, Xi, Lambda_rho_all = [], [], []
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))
lambda_ave, Data_ksi, Weights = [], [], []
loss_train = np.zeros((N,))
ATA_cond = np.zeros((N,))
err_NN = np.zeros((N,))
err_MGN = np.zeros((N,))

for i in range(3,N):
    gap = int(h[i]/h0)
    base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_gap_%s' % (ex, layer_info, ntrain, lrs, lr, gap)
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model_NN.load_state_dict(torch.load(model_path))
    model_NN.eval()
    
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'train'))
    loss_train[i] = loss[-1,1]

    m_fact = int(delta/h[i])
    
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    # ntrain = 100
    reader = MatReader(DATA)
    data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_X[m_fact:s+m_fact]
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
    
    total_nlambda = lambdaa.reshape(-1,).size(0)
    
    # lambda_min_all_data, lambda_max_all_data = 0.5, 1.5
    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    Nlambda = 50
    Lambda_rho = torch.linspace(lambda_min_data[i], lambda_max_data[i], Nlambda)
    dlambda_rho = Lambda_rho[1]- Lambda_rho[0]
        
    # weights = torch.abs((lambdaa-lambdaa**(-3))*ksi_plus_eta/ksi_plus_eta_norm)
    # weights = torch.abs((g_fun(ksi_norm)/ksi_norm)*ksi_plus_eta/ksi_plus_eta_norm)
    ksi_norm4rho = torch.tile(ksi_norm,(lambdaa.size(0),lambdaa.size(1),1))
    weights = torch.zeros((Nlambda,1))
    for k in range(Nlambda-1):
        indices = torch.nonzero((lambdaa >= Lambda_rho[k]) & (lambdaa <= Lambda_rho[k+1]), as_tuple=False)
        if indices.numel() != 0:
            index = (indices[:,0],indices[:,1],indices[:,2])
            weights[k] = torch.sum(torch.abs((g_fun(ksi_norm4rho[index]))*ksi_plus_eta[index]/ksi_plus_eta_norm[index]))/total_nlambda
        
    
    # rho = torch.sum(weights, axis=(0,1))/ntrain/s
    # rho = (torch.flip(rho[:m_fact], dims=[0])+rho[m_fact:])/2
    rho = weights
    # rho = weights*dlambda_rho
    Rho.append(rho)
    Lambda_rho_all.append(Lambda_rho)
    # Xi.append(data_ksi[m_fact:])
    # Weights.append(weights)s
    
    Lambda_rho = Lambda_rho.reshape(-1,1)
    Lambda_rho_cuda = Lambda_rho.to('cuda')
    Lambda_rho_1_cuda = torch.ones_like(Lambda_rho_cuda)
    # k1_true = 2*c*(Lambda_rho-Lambda_rho**(-3)).reshape(-1,1)
    k1_true = k1_fun(Lambda_rho)
    k1_NN = (model_NN.phi_1(Lambda_rho_cuda)- model_NN.phi_1(Lambda_rho_1_cuda)).reshape(-1,1).cpu().detach()
    k1_MGN = (model_MGN.phi_MGN(Lambda_rho_cuda)- model_MGN.phi_MGN(Lambda_rho_1_cuda)).reshape(-1,1).cpu().detach()
    err_NN[i] = torch.sqrt(torch.sum(rho*dlambda_rho*(k1_NN-k1_true)**2))/torch.sqrt(torch.sum(rho*dlambda_rho*k1_true**2))
    err_MGN[i] = torch.sqrt(torch.sum(rho*dlambda_rho*(k1_MGN-k1_true)**2))/torch.sqrt(torch.sum(rho*dlambda_rho*k1_true**2))
    
    
    # Lambdaa = Lambda_rho.reshape(-1,1)
    Lambdaa = torch.linspace(0.5, 5.5, 100).reshape(-1,1)
    Lambdaa_cuda = Lambdaa.to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    # k1_true = 2*c*(Lambda_rho-Lambda_rho**(-3)).reshape(-1,1)
    k1_true = k1_fun(Lambdaa)
    k1_NN = (model_NN.phi_1(Lambdaa_cuda)- model_NN.phi_1(Lambdaa_1_cuda)).reshape(-1,1).cpu().detach()
    k1_MGN = (model_MGN.phi_MGN(Lambdaa_cuda)- model_MGN.phi_MGN(Lambdaa_1_cuda)).reshape(-1,1).cpu().detach()
    # weights = weights.reshape(-1,n_ksi)
    
    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    print('min lambda: %s, max lambda: %s:' % (lambda_min_data[i], lambda_max_data[i]))
    

print(err_NN, err_MGN)
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 22
linewidth = 2.5
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize=(6, 5))
# ax.plot(lambdaa, dw1_exact, 'k',linewidth=2, label=r'true: $\lambda-\lambda^{-3}$')
# ax.plot(lambdaa, dw1_normlized, color='darkorange', linewidth=2, linestyle='--', label=r'normalized $k_1^{NN}(\lambda)$')
ax.plot(Lambdaa, k1_true, color='k', linewidth=linewidth, label='true') 
ax.plot(Lambdaa, k1_MGN, color=colors[0], linestyle='--', linewidth=linewidth, label='MGN')
ax.plot(Lambdaa, k1_NN, color='darkviolet', linestyle='-.', linewidth=linewidth, label='MLP')
 
# lambda_min_data, lambda_max_data = lambda_min_data[i], lambda_max_data[i]
lambda_min_data, lambda_max_data = 0.8109, 3.3456
# lambda_min_data, lambda_max_data = 0.5, 10
y_lim = ax.get_ylim()
ax.set_ylim(y_lim[0], y_lim[1])
ax.plot([lambda_min_data, lambda_min_data], [y_lim[0], y_lim[1]], color='gray', linestyle='--', linewidth=2)
ax.axvline(x=lambda_max_data, color='gray', linestyle='--', linewidth=2)
midpoint = (lambda_min_data + lambda_max_data) / 2
y_pos = (1-0.2)*y_lim[0]
ax.text(midpoint, y_pos, 'training data\ncoverage', fontsize=14, ha='center', va='center', color='gray')
ax.annotate('', xy=(lambda_min_data, y_pos), xytext=(midpoint-0.8, y_pos), arrowprops=dict(arrowstyle='->', color='gray', linestyle='--', linewidth=1.5))
ax.annotate('', xy=(midpoint+0.8, y_pos), xytext=(lambda_max_data, y_pos), arrowprops=dict(arrowstyle='<-', color='gray', linestyle='--', linewidth=1.5))

y_annotation = (1+0.2)*y_lim[0]
ax.text(lambda_min_data, y_annotation, f'$\lambda$={lambda_min_data}', ha='center', va='top', fontsize=fontsize, color='black')
ax.text(lambda_max_data, y_annotation, f'$\lambda$={lambda_max_data}', ha='center', va='top', fontsize=fontsize, color='black')

ax.legend(fontsize=fontsize-1, loc='lower right')
# ax.set_xlabel(r'$\lambda$')
ax.set_ylabel(r'$g(\lambda)$')
ax.xaxis.label.set_size(fontsize)
ax.yaxis.label.set_size(fontsize)
# plt.title(r"Learned $g(\lambda)$")
plt.tight_layout()
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.savefig('%s/%s_k_NN_MGN_new.png' % (current_dir, ex), format='png')
# plt.savefig('%s/%s_k_NN_MGN_lambda_%s_%s.png' % (base_dir, ex, lambda_min, lambda_max), format='png')


### plot u ###
data = []
for j in range(ntrain):
    data.append(Data(x=data_X , u=data_u[j, :, :], f=data_f[j, :, :], ksi=data_ksi[j, :, :]))

data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

plot_index = range(ndata)
# plot_index = [0, 1, 2, 3, 17, 22, 36, 38, 41, 47, 51, 92] 
plot_index = [3] 
compute, plot = 1, 1
if compute == 1:
    err_u = torch.zeros((ndata,1))
    err_b = torch.zeros((ndata,1))
    with torch.no_grad():
        # for j in range(ndata):
        for j in plot_index:
            u_true = data_u[j,:,:]
            
            data_u0 = torch.zeros_like(u_true)[m_fact:s+m_fact]
            
            # initial_type = 'interp'
            # index_bc = torch.cat((torch.arange(0,m_fact),torch.arange(s+m_fact, s+2*m_fact)))
            # x_bc = data_X[index_bc].squeeze()
            # u_bc = data_u_j[index_bc].squeeze()
            # interp_fun = interp1d(x_bc.numpy(), u_bc.numpy(), kind='linear')
            # data_u0 = torch.tensor(interp_fun(data_x.numpy()))
            
            def func_linear_NN(u):
                u = torch.tensor(u, dtype=torch.float64)
                full_u = (1-mask_bc)*u_true
                full_u[m_fact:s+m_fact, 0] = u
                # full_u[m_fact] = u_true[m_fact]
                # full_u[s+m_fact] = u_true[s+m_fact]
                data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                y = model_NN.linear(data)-data.f.squeeze()
                return y.cpu().detach().numpy()
            
            def func_NN(u):
                u = torch.tensor(u, dtype=torch.float64)
                full_u = (1-mask_bc)*u_true
                full_u[m_fact:s+m_fact, 0] = u
                # full_u[m_fact] = u_true[m_fact]
                # full_u[s+m_fact] = u_true[s+m_fact]
                data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                y = model_NN.nonlinear(data)-data.f.squeeze()
                return y.cpu().detach().numpy()
            
            if initial_type == 'linear_sol':
                # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                data_u0, info_linear, ier, msg = fsolve(func_linear_NN, data_u0, full_output=True)
                u_linear = u_true.clone()
                u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
            
            u, info, ier, msg = fsolve(func_NN, data_u0, xtol=1e-12, full_output=True)
            
            u = torch.tensor(u, dtype=torch.float64)
            uh_NN = u_true.clone()
            uh_NN[m_fact:s+m_fact,0] = u
            
            
            def func_linear_MGN(u):
                u = torch.tensor(u, dtype=torch.float64)
                full_u = (1-mask_bc)*u_true
                full_u[m_fact:s+m_fact, 0] = u
                # full_u[m_fact] = u_true[m_fact]
                # full_u[s+m_fact] = u_true[s+m_fact]
                data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                y = model_MGN.linear(data)-data.f.squeeze()
                return y.cpu().detach().numpy()
            
            def func_MGN(u):
                u = torch.tensor(u, dtype=torch.float64)
                full_u = (1-mask_bc)*u_true
                full_u[m_fact:s+m_fact, 0] = u
                # full_u[m_fact] = u_true[m_fact]
                # full_u[s+m_fact] = u_true[s+m_fact]
                data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                y = model_MGN.nonlinear(data)-data.f.squeeze()
                return y.cpu().detach().numpy()
            
            if initial_type == 'linear_sol':
                # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                data_u0, info_linear, ier, msg = fsolve(func_linear_MGN, data_u0, full_output=True)
                u_linear = u_true.clone()
                u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
            
            u, info, ier, msg = fsolve(func_MGN, data_u0, xtol=1e-12, full_output=True)
            
            u = torch.tensor(u, dtype=torch.float64)
            uh_MGN = u_true.clone()
            uh_MGN[m_fact:s+m_fact,0] = u
            
            
            plot_u_NN_MGN(data_X, u_true, uh_NN, Uh_MGN, j, base_dir, '%s_%s' % (ex_data, initial_type))
            
                
            # err_u[j] = (torch.norm(uh-u_true)/torch.norm(u_true)).item()
            # print((j, err_u[j], ier, np.max(info['fvec']), info['nfev']))
            
            # with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex_data, initial_type), "a") as file:
            #     file.write(f"{j}, {err_u[j]}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}\n")

                
            # # if plot == 1 and (err_u[j]>0.05):
            # if plot == 1 :
            #     # plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                
            #     if initial_type == 'linear_sol':
            #         plot_u_3(data_X, u_true, uh, u_linear, j, base_dir, '%s_%s' % (ex_data, initial_type))
            #     else: 
            #         plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                
                
            # j += 1
    # Err_u[i] = torch.mean(err_u)
    # # Err_b[i] = torch.mean(err_b)
    # print('Error of u:', torch.mean(err_u))
    # # print('Error of b:', torch.mean(err_b))
    
    # with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex_data, initial_type), "a") as file:
    #     file.write(f"mean err: {torch.mean(err_u)}\n")
