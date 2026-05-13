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
from utilities_INO_PD import *
# from CG_solver import *
from scipy.optimize import fsolve

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    
print('PID:', os.getpid())   
parser = argparse.ArgumentParser()
parser.add_argument('--layer_info', type=str, default='128_5')
parser.add_argument('--ex_data', type=str, default='ex35')
parser.add_argument('--act_xi', type=str, default='Sigmoid')
parser.add_argument('--lr', type=float, default=0.995)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--initial_type', type=str, default='linear_sol')

# parser.add_argument('--config_path', type=str, help='Path to the configuration file')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 4
h = np.array([2**(-5),2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
# ntrain = 300

# model and training parameters
batch_size = 1
batch_size2 = batch_size
layer_info = args.layer_info
phi_1_layer = parse_layer_info(layer_info)

act_xi = args.act_xi
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
elif act_xi == 'Sigmoid': 
    act_fun_xi = torch.nn.Sigmoid()

model = E_GCL_GKN(phi_1_layer, act_fun_xi).to(device)

current_dir = os.path.dirname(os.path.realpath(__file__))


DATA_PATH = '%s/../../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
ex_data = args.ex_data
if ex_data == 'ex35':
    ndata = 400  # ex35
    DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex_data)
elif ex_data == 'ex35_ood':
    ndata = 100
    DATA_NAME = 'BK_%s_ndata_100_Nx_257_delta_0.25_h_0.00390625' % (ex_data)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257
delta = 0.25
ntrain = 300
ex_train = 'ex35'

# initial_type = 'zero'
initial_type = args.initial_type

colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))

Err_u = np.zeros((N,))
Err_b = np.zeros((N,))

lrs = [0.01]
lr = [args.lr, 0.998]

for i in range(3, N):
    gap = int(h[i]/h0)
    base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_gap_%s_%s_seed_%s' % (ex_train, layer_info, ntrain, lrs, lr, gap, act_xi, args.seed)
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    m_fact = int(delta/h[i])
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    ndata = 100
    reader = MatReader(DATA)
    data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_X[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[-ndata:,::gap].reshape(ndata, S, 1)
    data_f = reader.read_field('bodyforce')[-ndata:,::gap].reshape(ndata, S, 1)
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
    ksi_range = torch.arange(-m_fact, m_fact + 1, 1).int()
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
        data.append(Data(x=data_X , u=data_u[j, :, :], f=data_f[j, :, :], ksi=data_ksi[j, :, :], eta=data_eta[j, :, :]))

    # data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    
    # compute_index = range(ndata)
    # exclude = {0, 10, 39, 47}
    # compute_index = [i for i in range(ndata) if i not in exclude]
    # n_test = len(compute_index)
    compute_index = {0, 1,2,3,4}

    compute, plot = 1, 0
    if compute == 1:
        err_u = torch.zeros((ndata,1))
        err_b = torch.zeros((ndata,1))
        with torch.no_grad():
            for j in compute_index:
                batch = data[j].to(device)
                b = model(batch).view(s, 1).cpu().detach()
                b_true = batch.f.view(-1, 1).cpu().detach()
                b_true = b_true[m_fact:s+m_fact,:]
                err_b[j] = torch.norm(b-b_true)/torch.norm(b_true)
            
            
            # for j in range(ndata):
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
                
                print((j, "err_b:", f'{err_b[j].item():.4e}', "err_u:", f'{err_u[j].item():.4e}', ier, np.max(info['fvec']), info['nfev']))
                
                with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex_data, initial_type), "a") as file:
                    file.write(f"{j}, {err_b[j].item():.4e}, {err_u[j].item():.4e}, fsolve flag:{ier}, {np.max(info['fvec']):.4e}, {info['nfev']}\n")

                    
                # if plot == 1 and (err_u[j]>0.05):
                if plot == 1 :
                    # plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                    
                    if initial_type == 'linear_sol':
                        plot_u_3(data_X, u_true, uh, u_linear, j, base_dir, '%s_%s' % (ex_data, initial_type))
                    else: 
                        plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                    
                    
                # j += 1
        
        err_b = err_b[compute_index]
        err_u = err_u[compute_index]
        print('Mean error of b:', f'{torch.mean(err_b):.4e}')
        Err_u[i] = torch.mean(err_u)
        # Err_b[i] = torch.mean(err_b)
        print('Error of u:', f'{(torch.mean(err_u)):.4e}')
        # print('Error of b:', torch.mean(err_b))
        
        with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex_data, initial_type), "a") as file:
            file.write(f"mean err: {torch.mean(err_u):.4e}\n")
            
print("Done!")
