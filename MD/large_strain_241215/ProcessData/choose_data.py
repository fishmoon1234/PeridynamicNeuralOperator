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
    
    DATA_NAME = 'cg-md-250212'
    DATA = os.path.join(current_dir, "./%s.mat" % DATA_NAME)
    Nx = 441
    ndata = 1400
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
    data_mass = reader.read_field('mass').reshape(Nx,)
    data_u = reader.read_field('disps')
    data_f = reader.read_field('forces')
    
    n0 = 10 # number of time points for one test
    n1 = 60 
    ndata = n0*n1
    indices = np.zeros((ndata))
    for i in range(n1):
        start = i*n0
        indices[start:start+n0] = range(i*20+1,i*20+1+n0)
 
    indices.astype(int)
    
    ntrain = ndata
    x_train = data_x
    u_train = data_u
    f_train = data_f
    
    ndata = np.size(indices)
    disps = data_u[indices,:,:].numpy().astype('float64')
    forces = data_f[indices,:,:].numpy().astype('float64')
    # disps = np.delete(disps, [18,160], axis=0)
    # forces = np.delete(forces, [18,160], axis=0)
    mass = reader.read_field('mass').numpy().astype('float64')
    coords = reader.read_field('coords').numpy().astype('float64')
    output_file_name = "%s/%s_%s_%s.mat" % (current_dir, DATA_NAME, n0, n1)
    savemat(output_file_name, {'ndata': ndata, 'coords': coords, "mass": mass, "disps": disps, "forces": forces})
    


if __name__ == "__main__":
    
    main()
