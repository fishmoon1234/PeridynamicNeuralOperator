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

N=4
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
h = 2**(-5)
# ntrain = np.array([50, 100, 150, 200])
ntrain = np.array([100, 200, 300, 400])

# model and training parameters
batch_size = 10
batch_size2 = batch_size
# kernel_layer = [1, 64, 128, 64, 1]
# phi_1_layer = [1, 64, 64, 64, 64,64, 64, 1]
# phi_2_layer = [1, 64, 64, 64, 1]
phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
phi_2_layer = [1, 64, 64, 64, 64, 64, 1]
alpha = 1.0

# model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, alpha).to(device)
model = E_GCL_GKN(phi_1_layer, phi_2_layer, alpha).to(device)
current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex14'
DATA_NAME = 'BK_%s_ndata_600_Nx_129_delta_0.25_h_0.0078125' % ex

err_k1 = np.zeros((N,))
err_k = np.zeros((N,))

for i in range(N):
    gap = 4
    base_dir = 'Results/%s_gap_%s_ntrain_%s' % (DATA_NAME, gap, ntrain[i])
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    lambda_min_data = 0.84
    lambda_max_data = 1.15
    delta = 0.25
    mu = 0.3846
    c = 2*mu/math.pi/delta**2
    # g_fun = lambda x: np.ones_like(x)
    g_fun = lambda x: x*np.exp(-50*x**2)
    # g_fun = lambda x: x**2
    alpha = 1
    
    # plot 
    xi_norm = h*torch.ones((100,))
    xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
    # lambda_min = 0.85
    # lambda_max = 1.15
    lambda_min = 0.85
    lambda_max = 1.15
    lambdaa = torch.linspace(lambda_min, lambda_max, 100)
    lambdaa_cuda = lambdaa.unsqueeze(1).to('cuda')
    lambdaa_1_cuda = torch.ones_like(lambdaa_cuda)
    dw1 = (model.phi_MGN(lambdaa_cuda)- model.phi_MGN(lambdaa_1_cuda))
    dw2 = model.phi_2(xi_norm_cuda)
    dw =  dw1*dw2
    dw = dw.cpu().detach().numpy()
    dw_exact =(lambdaa-lambdaa**(-3))*2*c/xi_norm*g_fun(xi_norm)
    dw1_exact = (lambdaa-lambdaa**(-3))
    
    dlambdaa = lambdaa[1]-lambdaa[0]
    weights = torch.ones((100,))
    weights[[0,-1]] = torch.tensor([1/2,1/2])
    coe_exact = torch.sum(dw1_exact*weights)*dlambdaa
    dw1 = torch.squeeze(dw1.cpu().detach())
    coe_nn = torch.sum(dw1*weights)*dlambdaa
    
    dw1_normlized = dw1/coe_nn*coe_exact
    err_k1[i] = np.linalg.norm(dw1_exact-dw1_normlized.flatten())/np.linalg.norm(dw1_exact)
    
    
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
    
    N=100
    xi_norm = torch.linspace(2**(-5),delta, N)
    # lambdaa = torch.linspace(lambda_min_data,lambda_max_data, N)
    lambdaa = torch.linspace(lambda_min,lambda_max, N)
    [Xi_norm, Lambdaa] = torch.meshgrid(xi_norm, lambdaa)
    Xi_norm_cuda = Xi_norm.reshape(-1,1).to('cuda')   
    Lambdaa_cuda = Lambdaa.reshape(-1,1).to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    dw = ((model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)) *model.phi_2(Xi_norm_cuda)).reshape(N,N)
    dw = dw.cpu().detach().numpy()
    dw_exact = (Lambdaa-Lambdaa**(-3))*2*c/Xi_norm*g_fun(Xi_norm)
    err_k[i] = np.linalg.norm(dw_exact-dw)/np.linalg.norm(dw_exact)
    
    
print("relative error for k1: %s" % err_k1)
print("relative error for k: %s" % err_k)

fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(1/ntrain, err_k1, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
plt.plot(1/ntrain, 1/ntrain/2*ntrain[0]*err_k1[0], 'k', label='slope=0.5', linewidth=2)
plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('1/N', fontsize=fontsize)
plt.ylabel(r'Relative $L^2$ error', fontsize=fontsize)
# show the figure
plt.title(r'$\lambda\in$[%s,%s]' % (lambda_min, lambda_max), fontsize=14)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.show()
plt.tight_layout()
name = '%s_error_k1_ntrain_%s_lambda_%s_%s.png' % (ex, ntrain, lambda_min, lambda_max)
plt.savefig (os.path.join (current_dir, name))
plt.close()


fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
plt.plot(1/ntrain, err_k, 's--', markerfacecolor='none',color='darkorange', linewidth=2)
plt.plot(1/ntrain, 1/ntrain/2*ntrain[0]*err_k[0], 'k', label='slope=0.5', linewidth=2)
plt.gca().invert_xaxis() 
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('1/N', fontsize=fontsize)
plt.ylabel(r'Relative $L^2$ error', fontsize=fontsize)
# show the figure
plt.title(r'$\lambda\in$[%s,%s]' % (lambda_min, lambda_max), fontsize=14)
plt.grid(True, which="both", ls="--", color='gray')
plt.legend()
plt.show()
plt.tight_layout()
name = '%s_error_k_ntrain_%s_lambda_%s_%s.png' % (ex, ntrain, lambda_min, lambda_max)
plt.savefig (os.path.join (current_dir, name))
plt.close()
