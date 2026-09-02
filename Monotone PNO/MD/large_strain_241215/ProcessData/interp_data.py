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


def main():

    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    # DATA_NAME = 'md-241215'
    # DATA_NAME = 'md-241215_10_60'
    DATA_NAME = 'cg-md-250212'
    DATA = os.path.join(current_dir, "./%s.mat" % DATA_NAME)
    Nx = 441
    
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')
    data_mass = reader.read_field('mass').reshape(21, 21)
    data_u = reader.read_field('disps').reshape(-1, 21, 21, 2)
    data_f = reader.read_field('forces').reshape(-1, 21, 21, 2)
    ndata = data_u.size(0)
    
    N_new = 41
    x = np.linspace(-50, 50, 21)
    x_new = np.linspace(-50, 50, N_new )
    X1_new, X2_new = np.meshgrid(x_new, x_new)
    coords =  np.concatenate([X2_new.reshape(-1,1), X1_new.reshape(-1,1)], axis=1)
    
    data_u_new = np.zeros((ndata, N_new, N_new, 2))
    data_f_new = np.zeros((ndata, N_new, N_new, 2))
    data_mass_new = np.zeros((N_new, N_new))
    interp_mass = interp2d(x, x, data_mass, kind='cubic')
    data_mass_new = interp_mass(x_new, x_new)
    
    for i in range(ndata):
        interp_u_0 = interp2d(x, x, data_u[i,:,:,0], kind='cubic')
        data_u_new[i,:,:,0]  = interp_u_0(x_new, x_new)
        interp_u_1 = interp2d(x, x, data_u[i,:,:,1], kind='cubic')
        data_u_new[i,:,:,1]  = interp_u_1(x_new, x_new)
        interp_f_0 = interp2d(x, x, data_f[i,:,:,0], kind='cubic')
        data_f_new[i,:,:,0]  = interp_f_0(x_new, x_new)
        interp_f_1 = interp2d(x, x, data_f[i,:,:,1], kind='cubic')
        data_f_new[i,:,:,1]  = interp_f_1(x_new, x_new)
    
    
    disps = data_u_new.reshape(ndata,-1,2)
    forces = data_f_new.reshape(ndata,-1,2)
    mass = data_mass_new.reshape(-1)
    output_file_name = "%s/%s_%s.mat" % (current_dir, DATA_NAME, N_new)
    savemat(output_file_name, {'ndata': ndata, 'coords': coords, "mass": mass, "disps": disps, "forces": forces})
    
    


if __name__ == "__main__":
    
    main()
