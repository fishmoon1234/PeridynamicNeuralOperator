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
# from CG_solver import *
from scipy.optimize import fsolve

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(os.path.realpath(__file__))

N = 4
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# h = np.array([2**(-7), 2**(-8)])
h = np.array([2**(-5),2**(-6), 2**(-7), 2**(-8)])
# h = np.array([2**(-4),2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
# ntrain = 300


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

ex_data = 'ex32'
# ex_data = 'ex20_sample_1'
# ex_data = 'ex20'
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
# DATA_NAME = 'BK_%s_ndata_100_Nx_257_delta_0.25_h_0.00390625' % (ex_data)
DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex_data)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257
delta = 0.25
ntrain = 300
ex_train = 'ex32'

# initial_type = 'zero'
initial_type = 'linear_sol'

colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))

Err_u = np.zeros((N,))
Err_b = np.zeros((N,))

for i in range(4):
    gap = int(h[i]/h0)
    # base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr%s_gap_%s' % (current_dir, ex_train, layer_info, ntrain, lrs, lr, gap)
    base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_sigmoid_gap_%s' % (current_dir, ex_train, layer_info, ntrain, lrs, lr, gap)
    model_path = '%s/model_%s.ckpt' % (base_dir, num_layers)
    # model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    m_fact = int(delta/h[i])
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    ndata = 100
    reader = MatReader(DATA)
    data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_X[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[:ndata,::gap].reshape(ndata, S, 1)
    data_f = reader.read_field('bodyforce')[:ndata,::gap].reshape(ndata, S, 1)
    # jump = torch.max(torch.abs(data_u[:,m_fact,:]), torch.abs(data_u[:,s+m_fact,:]))
    
    mask_bc = torch.zeros((S,1))
    mask_bc[m_fact:s+m_fact] = 1
    # mask_bc[m_fact+1:s+m_fact-1] = 1
    # mask_bc = mask_bc.to(device)
    
    ################## zero initial value  ####################
    # data_u0 = torch.zeros_like(data_u)
    # data_u0 = torch.zeros((s, 1))
    
    ################## interpolation initial value  ####################
    # index_bc = torch.cat((torch.arange(0,m_fact),torch.arange(s+m_fact, s+2*m_fact)))
    # x_bc = x[index_bc].squeeze()
    # u_bc = u[index_bc].squeeze()
    # interp_fun = interp1d(x_bc.numpy(), u_bc.numpy(), kind='cubic')
    # u = torch.tensor(interp_fun(xuf.x.numpy())).to(device)
    
    dx = h[i]
    ksi_range = torch.range(-m_fact, m_fact).int()
    ksi_range = ksi_range[ksi_range != 0]
    n_ksi = 2*m_fact
    data_ksi = ksi_range*dx
    data_ksi = data_ksi.repeat(ndata,s,1)
    
    data_eta = torch.zeros((ndata, s, n_ksi))
    for k in range(s):
        data_eta[:,k,:] = (data_u[:,m_fact+k+ksi_range].reshape(-1,1,n_ksi)-data_u[:,m_fact+k].reshape(-1,1,1)).squeeze()
    ksi_plus_eta_norm = torch.abs(data_ksi+data_eta)
    ksi_norm = torch.abs(data_ksi)
    extension = ksi_plus_eta_norm - ksi_norm
    lambdaa = 1.0 + extension / (ksi_norm + 1e-9)

    lambda_min_data = torch.min(torch.min(lambdaa, dim=1).values, dim=1).values
    lambda_max_data = torch.max(torch.max(lambdaa, dim=1).values, dim=1).values
    print(torch.min(lambda_min_data))
    print(torch.max(lambda_max_data))
    
    data = []
    for j in range(ndata):
        data.append(Data(x=data_X , u=data_u[j, :, :], f=data_f[j, :, :], ksi=data_ksi[j, :, :]))

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    plot_index = range(ndata)
    # plot_index = [0, 1, 2, 3, 17, 22, 36, 38, 41, 47, 51, 92] 
    # plot_index = [3, 17, 21, 36, 38, 41, 51, 62, 75]
    compute, plot = 1, 0
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
                
                def func_linear(u):
                    u = torch.tensor(u, dtype=torch.float64)
                    full_u = (1-mask_bc)*u_true
                    full_u[m_fact:s+m_fact, 0] = u
                    # full_u[m_fact] = u_true[m_fact]
                    # full_u[s+m_fact] = u_true[s+m_fact]
                    data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                    y = model.linear(data)-data.f.squeeze()
                    return y.cpu().detach().numpy()
                
                def func(u):
                    u = torch.tensor(u, dtype=torch.float64)
                    full_u = (1-mask_bc)*u_true
                    full_u[m_fact:s+m_fact, 0] = u
                    # full_u[m_fact] = u_true[m_fact]
                    # full_u[s+m_fact] = u_true[s+m_fact]
                    data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                    y = model.nonlinear(data)-data.f.squeeze()
                    return y.cpu().detach().numpy()
                
                if initial_type == 'linear_sol':
                    # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                    data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, full_output=True)
                    u_linear = u_true.clone()
                    u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
                
                u, info, ier, msg = fsolve(func, data_u0, xtol=1e-12, full_output=True)
                
                u = torch.tensor(u, dtype=torch.float64)
                uh = u_true.clone()
                uh[m_fact:s+m_fact,0] = u
                    
                err_u[j] = (torch.norm(uh-u_true)/torch.norm(u_true)).item()
                
                print((j, err_u[j], ier, np.max(info['fvec']), info['nfev']))
                
                with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex_data, initial_type), "a") as file:
                    file.write(f"{j}, {err_u[j]}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}\n")

                    
                # if plot == 1 and (err_u[j]>0.05):
                if plot == 1 :
                    # plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                    
                    if initial_type == 'linear_sol':
                        plot_u_3(data_X, u_true, uh, u_linear, j, base_dir, '%s_%s' % (ex_data, initial_type))
                    else: 
                        plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                    
                    
                # j += 1
        Err_u[i] = torch.mean(err_u)
        # Err_b[i] = torch.mean(err_b)
        print('Error of u:', torch.mean(err_u))
        # print('Error of b:', torch.mean(err_b))
        
        with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex_data, initial_type), "a") as file:
            file.write(f"mean err: {torch.mean(err_u)}\n")
            
        with open("%s/err_u_record.txt" % (current_dir), "a") as file:
            file.write(f"{ex_data}, {initial_type}, {dx}, {torch.mean(err_u)}\n")