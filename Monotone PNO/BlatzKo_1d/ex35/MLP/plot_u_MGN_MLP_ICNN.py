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
from scipy.optimize import fsolve

parser = argparse.ArgumentParser()
parser.add_argument('--initial_type', type=str, default='zero', help='Initial type for solver: zero or linear_sol')
args = parser.parse_args()

torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(os.path.realpath(__file__))
N = 4
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
ntrain = 300

# model and training parameters
# Load NN model
batch_size = 10
batch_size2 = batch_size
layer_info = '128_5'
phi_1_layer = parse_layer_info(layer_info)
model_NN = E_GCL_GKN(phi_1_layer, torch.nn.ReLU()).to(device)
total_params = sum(p.numel() for p in model_NN.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
model_NN_path = '%s/Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.99, 0.998]_gap_1_ReLU_seed_42' % current_dir
model_path = os.path.join(model_NN_path, 'model.ckpt')
model_NN.load_state_dict(torch.load(model_path))
model_NN.eval()

# Load MGN model
layer_info = '128_5'
phi_1_layer = parse_layer_info(layer_info)
model_MGN = E_GCL_GKN_MGN(phi_1_layer, torch.nn.Sigmoid()).to(device)
total_params = sum(p.numel() for p in model_MGN.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
model_MGN_path = '%s/../MGN/Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.995, 0.998]_gap_1_Sigmoid_seed_42' % current_dir
model_path = os.path.join(model_MGN_path, 'model.ckpt')
model_MGN.load_state_dict(torch.load(model_path))
model_MGN.eval()


# Load ICNN model
layer_info = '128_5'
phi_1_layer = parse_layer_info(layer_info)
model_ICNN = E_GCL_GKN_ICNN(phi_1_layer, torch.nn.Softplus()).to(device)
total_params = sum(p.numel() for p in model_ICNN.parameters() if p.requires_grad)
print(f"ICNN Total number of parameters: {total_params}")
model_ICNN_path = '%s/../ICNN/Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.99, 0.998]_gap_1_Softplus_seed_42' % current_dir
model_path = os.path.join(model_ICNN_path, 'model.ckpt')
model_ICNN.load_state_dict(torch.load(model_path))
model_ICNN.eval()

ex = 'ex35'
DATA_PATH = '%s/../../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
Nx = 257

initial_type = args.initial_type

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
g_fun = lambda x: np.exp(-50*x**2)*(delta-np.abs(x))
k1_fun = lambda x: np.pi*(x - x**(-3)) + np.sin(np.pi*x)

for i in range(3, N):
    gap = int(h[i]/h0)
    m_fact = int(delta/h[i])
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    ndata = 100
    reader = MatReader(DATA)
    data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_X[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[-ndata:,::gap].reshape(ndata, S, 1)
    data_f = reader.read_field('bodyforce')[-ndata:,::gap].reshape(ndata, S, 1)
    
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
    

    ### plot u ###
    data = []
    for j in range(ndata):
        data.append(Data(x=data_X , u=data_u[j, :, :], f=data_f[j, :, :], ksi=data_ksi[j, :, :]))

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    # plot_index = range(ndata)
    # plot_index = [0, 1, 2, 3, 17, 22, 36, 38, 41, 47, 51, 92] 
    # plot_index = [3] 
    plot_index = {0, 1,2,3,4}
    compute, plot = 1, 0
    if compute == 1:
        err_u = torch.zeros((ndata,1))
        err_b = torch.zeros((ndata,1))
        with torch.no_grad():
            # for j in range(ndata):
            for j in plot_index:
                print(f"The j-th data is: {j}")
                u_true = data_u[j,:,:]
                
                
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
                
                data_u0 = torch.zeros_like(u_true)[m_fact:s+m_fact]
                if initial_type == 'linear_sol':
                    # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                    data_u0, info_linear, ier, msg = fsolve(func_linear_NN, data_u0, full_output=True)
                    print(f"NN linear solver for sample {j}: error={np.max(np.abs(info_linear['fvec'])):.2e}")
                    u_linear = u_true.clone()
                    u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
                
                u, info, ier, msg = fsolve(func_NN, data_u0, xtol=1e-12, full_output=True)
                print(f"NN nonlinear solver for sample {j}: error={np.max(np.abs(info['fvec'])):.2e}")
                
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
                
                data_u0 = torch.zeros_like(u_true)[m_fact:s+m_fact]
                if initial_type == 'linear_sol':
                    # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                    data_u0, info_linear, ier, msg = fsolve(func_linear_MGN, data_u0, full_output=True)
                    print(f"MGN linear solver for sample {j}: error={np.max(np.abs(info_linear['fvec'])):.2e}")
                    u_linear = u_true.clone()
                    u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
                
                u, info, ier, msg = fsolve(func_MGN, data_u0, xtol=1e-12, full_output=True)
                print(f"MGN nonlinear solver for sample {j}: error={np.max(np.abs(info['fvec'])):.2e}")
                
                u = torch.tensor(u, dtype=torch.float64)
                uh_MGN = u_true.clone()
                uh_MGN[m_fact:s+m_fact,0] = u
                
                
                # ICNN solver functions
                def func_linear_ICNN(u):
                    # Enable gradients for this computation
                    with torch.enable_grad():
                        u = torch.tensor(u, dtype=torch.float64, requires_grad=True)
                        # Create full_u with proper gradient handling
                        full_u = u_true.clone()
                        full_u[m_fact:s+m_fact, 0] = u
                        # Apply boundary conditions by zeroing out the boundary regions
                        # This preserves gradients for the interior region
                        full_u = mask_bc * full_u + (1-mask_bc) * u_true
                        data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                        y = model_ICNN.linear(data)-data.f.squeeze()
                        return y.cpu().detach().numpy()
                
                def func_ICNN(u):
                    # Enable gradients for this computation
                    with torch.enable_grad():
                        u = torch.tensor(u, dtype=torch.float64, requires_grad=True)
                        # Create full_u with proper gradient handling
                        full_u = u_true.clone()
                        full_u[m_fact:s+m_fact, 0] = u
                        # Apply boundary conditions by zeroing out the boundary regions
                        # This preserves gradients for the interior region
                        full_u = mask_bc * full_u + (1-mask_bc) * u_true
                        data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                        y = model_ICNN.nonlinear(data)-data.f.squeeze()
                        return y.cpu().detach().numpy()
                
                # Solve with ICNN
                data_u0_icnn = torch.zeros_like(u_true)[m_fact:s+m_fact]
                if initial_type == 'linear_sol':
                    # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                    data_u0_icnn, info_linear, ier, msg = fsolve(func_linear_ICNN, data_u0_icnn, full_output=True)
                    print(f"ICNN linear solver for sample {j}: error={np.max(np.abs(info_linear['fvec'])):.2e}")
                    u_linear = u_true.clone()
                    u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0_icnn, dtype=torch.float64)
                
                u_icnn, info, ier, msg = fsolve(func_ICNN, data_u0_icnn, xtol=1e-12, full_output=True)
                print(f"ICNN nonlinear solver for sample {j}: error={np.max(np.abs(info['fvec'])):.2e}")
                
                u_icnn = torch.tensor(u_icnn, dtype=torch.float64)
                uh_ICNN = u_true.clone()
                uh_ICNN[m_fact:s+m_fact,0] = u_icnn
                
                
                plot_u_NN_MGN_ICNN(data_X, u_true, uh_NN, uh_MGN, uh_ICNN, j, current_dir, '%s_%s' % (ex, initial_type))
                
                    
