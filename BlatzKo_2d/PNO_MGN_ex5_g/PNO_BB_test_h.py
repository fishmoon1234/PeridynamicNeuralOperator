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

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

N=5
h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
h0 = 2**(-7)
ntrain = 300

# model and training parameters
batch_size = 10
batch_size2 = batch_size
phi_layer= [1, 64, 64, 64, 64, 64, 1]
model = EGKN(phi_layer).to(device)

current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex15'
DATA_NAME = 'BK_%s_ndata_500_Nx_129_delta_0.25_h_0.0078125' % ex


err_k = np.zeros((N,))

for i in range(N):
    gap = int(h[i]/h0)
    base_dir = 'Results/%s_gap_%s_ntrain_%s' % (DATA_NAME, gap, ntrain)
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    delta = 0.25
    mu = 0.3846
    c = 2*mu/math.pi/delta**2
    # g_fun = lambda x: np.ones_like(x)
    g_fun = lambda x: x  # notice: this is not true for generating data.
    # g_fun = lambda x: x**2
    
    lambda_min_data, lambda_max_data = 0.5439, 1.4560  # h=2**(-7)
    lambda_min, lambda_max = 0.7, 1.3
    
    # plot 
    N=100
    lambdaa = torch.linspace(lambda_min, lambda_max, N)
    lambdaa_cuda = lambdaa.unsqueeze(1).to('cuda')
    lambdaa_1_cuda = torch.ones_like(lambdaa_cuda)
    k_NN = (model.phi_MGN(lambdaa_cuda)- model.phi_MGN(lambdaa_1_cuda))
    k_NN = k_NN.cpu().detach().numpy()
    k_true =(lambdaa-lambdaa**(-3))*2*c
    
    err_k[i] = np.linalg.norm(k_true-k_NN.flatten())/np.linalg.norm(k_true)
    
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # fontsize = 15
    # ax.plot(lambdaa, dw1_exact, 'k',linewidth=2, label=r'true: $\lambda-\lambda^{-3}$')
    # ax.plot(lambdaa, dw1_normlized, color='darkorange', linewidth=2, linestyle='--', label=r'normalized $k_1^{NN}(\lambda)$')
    
    # # y_lim = ax.get_ylim()
    # # ax.set_ylim(y_lim[0], y_lim[1])
    # # ax.plot([lambda_min_data, lambda_min_data], [y_lim[0], y_lim[1]], color='gray', linestyle='--', linewidth=2)
    # # ax.axvline(x=lambda_max_data, color='gray', linestyle='--', linewidth=2)
    # # # place text between the two lines
    # # midpoint = (lambda_min_data + lambda_max_data) / 2
    # # y_pos = (1-0.3)*y_lim[0]
    # # ax.text(midpoint, y_pos, 'training data coverage', fontsize=15, ha='center', va='center', color='gray')
    # # ax.annotate('', xy=(lambda_min_data, y_pos), xytext=(lambda_min_data+0.25, y_pos), arrowprops=dict(arrowstyle='->', color='gray', linestyle='--', linewidth=1.5))
    # # ax.annotate('', xy=(lambda_max_data-0.25, y_pos), xytext=(lambda_max_data, y_pos), arrowprops=dict(arrowstyle='<-', color='gray', linestyle='--', linewidth=1.5))
    
    # # y_annotation = (1+0.1)*y_lim[0]
    # # ax.text(lambda_min_data, y_annotation, f'$\lambda$={lambda_min_data}', ha='center', va='top', fontsize=fontsize, color='black')
    # # ax.text(lambda_max_data, y_annotation, f'$\lambda$={lambda_max_data}', ha='center', va='top', fontsize=fontsize, color='black')

    # ax.legend(fontsize=fontsize, loc='lower right')
    # # ax.set_xlabel(r'$\lambda$')
    # # ax.set_ylabel(r'$k_1^{NN}(\lambda)$')
    # ax.xaxis.label.set_size(fontsize)
    # ax.yaxis.label.set_size(fontsize)
    # ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # plt.savefig('%s/%s_k_MGN_lambda_%s_%s_h_%s.png' % (base_dir, ex, lambda_min, lambda_max, h[i]), format='png')
    
    
print("relative error for k: %s" % err_k)

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(h, err_k, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
plt.plot(h, h/h[0]*err_k[0], 'k', label='slope=1', linewidth=2)
plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Mesh size', fontsize=fontsize)
plt.ylabel(r'Relative $L^2$ error of $k(\lambda,\xi)$', fontsize=fontsize)
# show the figure
plt.title(r'$\lambda\in$[%s,%s]' % (lambda_min, lambda_max), fontsize=14)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.show()
plt.tight_layout()
name = '%s_error_k_ntrain_%s_lambda_%s_%s_h.png' % (ex, ntrain, lambda_min, lambda_max)
plt.savefig (os.path.join (current_dir, name))
plt.close()
