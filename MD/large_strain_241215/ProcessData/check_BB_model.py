import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
# from egnn_gcl import *
from timeit import default_timer
import os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import bisect
import random
from scipy.io import savemat
from scipy.interpolate import interp2d

random.seed(1)
torch.manual_seed(12)
np.random.seed(1)


# def k1_fun(x):
#     A = 0.8611065
#     C = 0.180
#     result = torch.empty_like(x)
#     result[x < 0] = x[x < 0]
#     result[x >= 0] = A/2*(2*x[x>=0]/(1+torch.abs(x[x>=0])/C)-x[x>=0]**2/C/(1+torch.abs(x[x>=0])/C)**2)

#     return result


def main():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    DATA_NAME = 'md-241215'
    DATA = os.path.join(current_dir, "./%s.mat" % DATA_NAME)
    Nx = 441
    
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')
    data_mass = reader.read_field('mass')
    data_u = reader.read_field('disps')
    data_f = reader.read_field('forces').reshape(-1,21,21,2)
    ndata = data_u.size(0)
    
    
    
    # data_x = data_x*0.01
    # data_u = data_u*0.01
    # data_f = data_f/83.75 *100
    data_f = data_f/83.75
    # f1 = data_f[0,:,:,0]
    # f2 = data_f[0,:,:,1]
    
    B_norm = np.sqrt(torch.mean((torch.norm(data_f, dim=-1))**2, dim=[1,2]))
    
    S = 21
    m_fact = 2.80
    # m_fact = 10.01
    dx = data_x[1,1]-data_x[0,1]
    delta = m_fact * dx
    delta = 14.0
    # cond_f = torch.abs(data_x.view(S,S,2)[0,:,1]) <= (data_x[-1,-1]- delta + 1e-10) 
    cond_f = torch.abs(data_x.view(S,S,2)[0,:,1]) <= (31 + 1e-10) 
    
   
    ntrain = ndata
    x_train = data_x
    u_train = data_u
    
    data_f = data_f[:,cond_f,:,:]
    data_f = data_f[:,:,cond_f,:]

    # import mesh and dataset
    edge_index = {}
    edge_attr = {}

    meshgenerator = IrregularMeshGenerator(data_x, [S, S])
    edge_index = meshgenerator.ball_connectivity(float(delta))
    edge_attr = meshgenerator.attributes(theta=0)
     
    
    # A = 0.8611065
    # B = 4.80
    # C = 0.180
    A = 1.026651
    # A = 0.3469978
    B = 4.80
    C = 0.180
    
    # k1_fun = lambda s: s if s<0 else (A/2*(2*s/(1+np.abs(s)/C)-s**2/C/(1+np.abs(s)/C)**2)) 
    # k1_fun = lambda s: A/2*(2*s/(1+torch.abs(s)/C)-s**2/C/(1+torch.abs(s)/C)**2)
    k1_fun = lambda s: A/2*(2*s/(1+torch.abs(s)/C)-s*torch.abs(s)/C/(1+torch.abs(s)/C)**2)
    k2_fun = lambda q: np.exp(-B*q)
    k_fun = lambda s, q: k1_fun(s)*k2_fun(q)
    

    if_compute_lambda = 1
    if if_compute_lambda == 1:
        err = np.zeros((ntrain, 1)) 
        # B_norm = np.zeros((ntrain, 1)) 
        lambda_min_max = np.zeros((ntrain, 2))     
        for i in range(ntrain):
            col, row = edge_index
            ksi = x_train[col] - x_train[row]
            eta = u_train[i,col] - u_train[i,row]
            ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
            ksi_plus_eta = ksi+eta
            ksi_plus_eta_norm = torch.norm(ksi_plus_eta, dim=1).unsqueeze(1)
            extension = ksi_plus_eta_norm - ksi_norm
            bond_dir = ksi_plus_eta / (ksi_plus_eta_norm)
            lambdaa = 1.0 + extension / (ksi_norm )
            # print(torch.mean(lambdaa))
            lambda_min_max[i,0] = torch.min(lambdaa)
            lambda_min_max[i,1] = torch.max(lambdaa)
            
            # L = dx**2*unsorted_segment_sum(k_fun(lambdaa-1, ksi_norm/delta)* bond_dir, row, num_segments=x_train.size(0)).reshape(21,21,2)
            L = unsorted_segment_sum(k_fun(lambdaa-1, ksi_norm/delta)* bond_dir, row, num_segments=x_train.size(0)).reshape(21,21,2)
            # L = dx**2*unsorted_segment_sum(k_fun(lambdaa-1, ksi_norm/14.0)* bond_dir, row, num_segments=x_train.size(0)).reshape(21,21,2)
            # L = unsorted_segment_sum( k_fun(lambdaa-1, ksi_norm/delta)* bond_dir, row, num_segments=x_train.size(0)).reshape(21,21,2)
            L = L[cond_f,:,:]
            L = L[:,cond_f,:]
            
    
            abs_err = np.sqrt(torch.mean(torch.norm(L+data_f[i,:,:,:], dim=-1)**2))
            # # abs_err = np.sqrt(torch.mean(torch.norm(L, dim=-1)**2))
            # B_norm[i] = np.sqrt(torch.mean(torch.norm(data_f[i,:,:,:], dim=-1)**2))
            err[i] = abs_err/B_norm[i]
            
            # print(err[i])
            # print('--')
        print(np.mean(err))
            
        plt.rcParams.update({'font.size': 15}) 
        fig, ax = plt.subplots(figsize = (6,5))
        fontsize = 15
        ax.plot(range(ntrain), lambda_min_max[:,0], color='darkorange', linewidth=2)
        ax.plot(range(ntrain), lambda_min_max[:,1], color='green', linewidth=2)
        plt.tight_layout()
        plt.savefig('%s/figure/%s_lambda.png' % (current_dir, DATA_NAME), format='png')
        plt.close(fig)
        
    print(np.mean(err))
    plt.rcParams.update({'font.size': 15}) 
    fig, ax = plt.subplots(figsize = (6,5))
    fontsize = 15
    ax.plot(range(ntrain), err, color='darkorange', linewidth=2)
    plt.tight_layout()
    plt.savefig('%s/figure/%s_model_err.png' % (current_dir, DATA_NAME), format='png')
    plt.close(fig)


if __name__ == "__main__":
    
    main()
