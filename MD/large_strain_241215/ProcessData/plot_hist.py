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

random.seed(1)
torch.manual_seed(12)
np.random.seed(1)


def main():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    # base_dir = './Results/%s_lr%s_dr%s_wd%s_step%s_bs%s' % (
    #     layer_info, lrs[0], gammas[0], wds[0], scheduler_step, batch_size)
    # base_dir = os.path.join(current_dir, base_dir)
    # if not os.path.exists(base_dir):
    #     os.makedirs(base_dir)
    
    # DATA = os.path.join(current_dir, "./graphene.mat")
    # data_name = 'lenjo-md-241204_5_60'
    # data_name = 'md-241215_41'
    # data_name = 'md-241215'
    # data_name = 'md-241215_large_strain'
    # data_name = 'cg-md-250212'
    # data_name = 'cg-md-250212_10_60_0.4_0.01'
    # data_name = 'cg-md-250212_0.5'
    data_name = 'cg-md-250212_0.2'
    DATA = "./%s.mat" % data_name
    DATA = os.path.join(current_dir, DATA)
    Nx = 441
    # ndata = 298
    # ntrain = 200
    # nvalid = 10
    # ntest = 10
    # train_index = random.sample(range(N_data), ntrain)
    # # start = 1700
    # # train_index = list(range(start, start+ntrain))
    # remaining_numbers = list(set(range(N_data)) - set(train_index))
    # valid_index = random.sample(remaining_numbers, nvalid)
    # remaining_numbers = list(set(range(N_data)) - set(train_index)- set(valid_index))
    # test_index = random.sample(remaining_numbers, ntest)
    # print(np.max(train_index))
    # print(np.max(valid_index))
    # print(np.max(test_index))
    
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')
    data_mass = reader.read_field('mass')
    data_u = reader.read_field('disps')
    data_f = reader.read_field('forces')
    
    ndata = data_u.size(0)
    
    ntrain = ndata
    x_train = data_x
    u_train = data_u
    f_train = data_f
    
    if_compute_lambda = 1

    sub = 1
    S = 21
    
    
    ################## Dirichlet BC ########################
    m_fact = 3.01
    dx = data_x[1,1]-data_x[0,1]
    delta = m_fact * dx

    # import mesh and dataset
    edge_index = {}
    edge_attr = {}

    meshgenerator = IrregularMeshGenerator(data_x, [S, S])
    edge_index = meshgenerator.ball_connectivity(float(delta))
    edge_attr = meshgenerator.attributes(theta=0)

    data_train = []
    for j in range(ntrain):
        data_train.append(Data(x=data_x, u=u_train[j, :, :], f=f_train[j, :, :], edge_index=edge_index, edge_attr=edge_attr, delta=delta, dx=dx))
        

    if if_compute_lambda == 1:
        lambda_all = []
        lambda_min_max = np.zeros((ntrain,2))     
        for i in range(ntrain):
            col, row = edge_index
            ksi = x_train[col] - x_train[row]
            eta = u_train[i,col] - u_train[i,row]
            ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
            ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
            extension = ksi_plus_eta_norm - ksi_norm
            lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
            lambda_all.append(lambdaa.numpy())
            # print(torch.mean(lambdaa))
            lambda_min_max[i,0] = torch.min(lambdaa) 
            lambda_min_max[i,1] = torch.max(lambdaa)
            
        plt.rcParams.update({'font.size': 15}) 
        fig, ax = plt.subplots(figsize = (6,5))
        fontsize = 15
        ax.plot(range(ntrain), lambda_min_max[:,0], color='darkorange', linewidth=2)
        ax.plot(range(ntrain), lambda_min_max[:,1], color='green', linewidth=2)
        plt.title(r'%s: min and max of $\lambda$' % data_name)
        plt.tight_layout()
        plt.savefig('%s/figure/%s_lambda.png' % (current_dir, data_name), format='png')
        plt.close(fig)
        
        data = np.array(lambda_all).reshape(-1,)
        print('mean:', np.mean(data))

        std = np.std(data)
        print('std:', std)
        print('---')
        
        
        ###################  plot histogram ################
        plt.rcParams.update({'font.size': 15}) 
        fig, ax = plt.subplots(figsize = (6, 5))
        # plt.hist(lambda_min_max, bins=30, edgecolor='black', alpha=0.7)
        plt.hist(np.array(lambda_all).reshape(-1,), bins=30, edgecolor='black', alpha=0.7) 
        plt.xlabel(r"$\lambda$")
        plt.ylabel("number of bonds")
        plt.title('%s: std=%.4f' % (data_name, std.item()))
        plt.tight_layout()
        plt.savefig('%s/figure/%s_strain.png' % (current_dir, data_name), format='png')
        
        
        
        # deformation = 1
        # indices0 = np.where((lambda_min_max[:, 0] <= (1-0.01*deformation)) & (lambda_min_max[:, 1] >= (1+ 0.01*deformation)))
        # print(torch.min(torch.norm(data_f, dim=[1])))
        # print(torch.min(torch.norm(data_f[indices0[0],:,:], dim=[1,2])))
        
        # ndata_new = 400
        # ndata0 = np.size(indices0)
        # remaining_numbers = list(set(range(ndata)) - set(indices0[0]))
        # indices1 = random.sample(remaining_numbers, ndata_new-ndata0)
        # indices = np.concatenate((indices0[0], indices1))
        # disps = data_u[indices,:,:].numpy().astype('float64')
        # forces = data_f[indices,:,:].numpy().astype('float64')
        # mass = reader.read_field('mass').numpy().astype('float64')
        # coords = reader.read_field('coords').numpy().astype('float64')
        # output_file_name = "%s/%s_%s.mat" % (current_dir, data_name, deformation)
        # savemat(output_file_name, {'ndata': ndata_new, 'coords': coords, "mass": mass, "disps": disps, "forces": forces})
    
     
       


if __name__ == "__main__":
    
    main()
