import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from timeit import default_timer
import os, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import bisect
import glob, re
from scipy.io import savemat


current_dir = os.path.dirname(os.path.realpath(__file__))
input_folder = "../DATA/cg-md-250212/"
input_folder = os.path.join(current_dir, input_folder)
output_folder = current_dir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_file_name = os.path.join(output_folder, "cg-md-250212.mat")

keyword_coords = "CG node positions and masses"
keyword_disp = "displacements" 
keyword_force = "load vectors" 
# output_file_coords = os.path.join(output_folder, "coords.mat")
# output_file_mass = os.path.join(output_folder, "mass.mat")
# output_file_disps = "disps.mat"
# output_file_forces = "forces.mat"

N_data = 1400 # 70*20
Nx = 441 # 21*21
coords = np.zeros((Nx,2))
mass = np.zeros((Nx,))
disps = np.zeros((N_data,Nx,2))
forces = np.zeros((N_data,Nx,2))


# coords, mass
with open(input_folder+"cg1.dat", "r") as infile:
    lines = infile.readlines()
    for i, line in enumerate(lines):
        if keyword_coords in line:
            # number of lines to extract after the keyword
            line_data = lines[i + 1:i + 1 + Nx]
            data_array = np.array([list(map(float, line.split())) for line in line_data])
            coords[:,0] = data_array[:,1]
            coords[:,1] = data_array[:,2]
            mass = data_array[:,3]
            break
# savemat(output_file_name, {'coords': coords, "mass": mass})
                
j=0
k=0
file_paths = glob.glob(input_folder + "*.dat")
sorted_files = sorted(
    file_paths, 
    key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group())
)
for file_path in sorted_files:
    with open(file_path, "r") as infile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            if keyword_disp in line:
                line_data = lines[i + 1:i + 1 + Nx]
                data_array = np.array([list(map(float, line.split())) for line in line_data])
                disps[j,:,0] = data_array[:,1]
                disps[j,:,1] = data_array[:,2]
                j +=1
            if keyword_force in line:
                line_data = lines[i + 1:i + 1 + Nx]
                data_array = np.array([list(map(float, line.split())) for line in line_data])
                forces[k,:,0] = data_array[:,1]
                forces[k,:,1] = data_array[:,2]
                k +=1
                
savemat(output_file_name, {'coords': coords, "mass": mass, "disps": disps, "forces": forces})
        