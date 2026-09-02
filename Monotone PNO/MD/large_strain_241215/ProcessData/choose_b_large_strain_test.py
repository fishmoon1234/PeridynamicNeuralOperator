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


def plt_strain(index, lambda_min_max, fig_name):
    
    plt.rcParams.update({'font.size': 15}) 
    fig, ax = plt.subplots(figsize = (20,5))
    fontsize = 15
    ax.plot(lambda_min_max[index,0], color='darkorange', linewidth=2)
    ax.plot(lambda_min_max[index,1], color='green', linewidth=2)
    n_points = len(index)
    if n_points > 0:
        x_ticks = range(0, n_points, 10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i) for i in x_ticks])
    plt.xlabel('Data Index', fontsize=fontsize)
    plt.ylabel('Strain Value', fontsize=fontsize)
    plt.title('min and max strain')
    plt.tight_layout()
    # plt.savefig('%s/figure/%s_lambda.png' % (current_dir, data_name), format='png')
    plt.savefig(fig_name, format='png')
    plt.close(fig)


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
    data_name = 'cg-md-250212'
    # data_name = 'cg-md-250212_10_70'
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
    
    index = range(ndata)
    
    # INDEX_NAME = 'cg-md-250212_index_b1'
    # DATA_index = os.path.join(current_dir, "../ProcessData/%s.mat" % INDEX_NAME)
    # reader = MatReader(DATA_index)
    # index = reader.read_field('index').flatten().int()
    data_u = data_u[index]
    data_f = data_f[index]
    
    u_L2 = torch.norm(data_u, dim=[1,2])
    b_L2 = torch.norm(data_f, dim=[1,2])
    

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
            
        print(np.min(lambda_min_max[:,0]), np.max(lambda_min_max[:,1]))
            
        # plt.rcParams.update({'font.size': 15}) 
        # fig, ax = plt.subplots(figsize = (20,5))
        # fontsize = 15
        # ax.plot(range(ntrain), lambda_min_max[:,0], color='darkorange', linewidth=2)
        # ax.plot(range(ntrain), lambda_min_max[:,1], color='green', linewidth=2)
        # plt.title('min and max strain')
        # plt.tight_layout()
        # # plt.savefig('%s/figure/%s_lambda.png' % (current_dir, data_name), format='png')
        # plt.savefig('%s/figure/%s_%s_lambda.png' % (current_dir, data_name, INDEX_NAME), format='png')
        # plt.close(fig)
        
        
        # deformation = 0.1
        # indices0 = np.where((lambda_min_max[:, 0] <= (1-deformation)) | (lambda_min_max[:, 1] >= (1+deformation)))
        # # index_new = index[indices0[0]].numpy()
        # index_new = indices0[0]
        # output_file_name = "%s/%s_%s.mat" % (current_dir, data_name, deformation)
        # savemat(output_file_name, {'index': index_new})
        
        # plt_strain(indices0[0],lambda_min_max, "%s/figure/%s_%s_lambda.png" % (current_dir, data_name, deformation))
        
        b_threshold = 0.01
        index_b = []
        index_b.append(0)
        for i in range(1,ndata):
            if np.abs(b_L2[i]-b_L2[i-1])/np.abs(b_L2[i])>=b_threshold:
                index_b.append(i)
        # index_b = index_b[0]  # Remove this line to keep index_b as a list
        
        deformation_1 = 0.1
        deformation_2 = 0.15
        index_strain = np.where(((lambda_min_max[:, 0] <= (1-deformation_1)) | (lambda_min_max[:, 1] >= (1+deformation_1))) & (lambda_min_max[:, 0] >= (1-deformation_2)) & (lambda_min_max[:, 1] <= (1+deformation_2)))
        # print(torch.min(torch.norm(data_f, dim=[1])))
        # print(torch.min(torch.norm(data_f[indices0[0],:,:], dim=[1,2])))
        index_strain = index_strain[0]

        
        # Calculate intersection of index_b and index_strain
        index_intersection = np.intersect1d(index_strain, index_b)
        print(f"index_b length: {len(index_b)}")
        print(f"index_strain length: {len(index_strain)}")
        print(f"index_intersection length: {len(index_intersection)}")
        print(f"index_intersection: {index_intersection}")
        
        output_file_name = "%s/index_%s_%s_%s_%s.mat" % (current_dir, data_name, deformation_1, deformation_2, b_threshold)
        savemat(output_file_name, {'index': index_intersection})
        
        plt_strain(index_intersection,lambda_min_max, "%s/figure/%s_%s_%s_%s_lambda.png" % (current_dir, data_name, deformation_1, deformation_2, b_threshold))
    
     
       

if __name__ == "__main__":
    
    main()
