import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
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
parser.add_argument('--layer_info', type=str, default='32_4')
# parser.add_argument('--config_path', type=str, help='Path to the configuration file')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
# ex_data = 'ex35'
# DATA_NAME = 'BK_%s_ndata_100_Nx_257_delta_0.25_h_0.00390625' % (ex_data)
ex_data = 'ex35_ood'
DATA_NAME = 'BK_%s_ndata_100_Nx_257_delta_0.25_h_0.00390625' % (ex_data)
DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
delta = 0.25


N = 4
h = np.array([2**(-5),2**(-6), 2**(-7), 2**(-8)])
# h = np.array([2**(-4),2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
Nx = 257
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
    
    deformation_1 = 0.82
    deformation_2 = 3.35
    index_strain = np.where((lambda_min_data <= (deformation_1)) | (lambda_max_data >= (deformation_2)))
    print(f"Found {len(index_strain[0])} samples that meet the strain criteria out of {ndata} total samples")
    data_u = data_u[index_strain]
    data_f = data_f[index_strain]
    
    data_name = "BK_ex35_ood_filtered_ndata_%s_Nx_%s_delta_%s_h_%s.mat" % (len(index_strain[0]), Nx, delta, h)
    scipy.io.savemat(os.path.join(DATA_PATH, data_name), {'coords': data_X,'displacement': data_u,'bodyforce':data_f})
    
    
            
print("Done!")
