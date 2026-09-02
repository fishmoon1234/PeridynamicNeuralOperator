#quadrature weights
import torch

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scipy.integrate as integrate
import matplotlib.ticker as ticker

# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7)])
# err = np.array([0.0111, 0.0024, 0.00145, 0.00121, 0.000872])
# err = np.array([0.0726488, 0.045723725, 0.04161059, 0.0386333, 0.031260498])
# err = np.array([0.037390035, 0.008931619, 0.0033416722, 0.0023599407, 0.0034566324])
# err = np.array([5.20, 0.11, 0.0187, 0.014, 0.016])*1e-3

N = 5
# h = np.array([2**(-3), 2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
h0 = 2**(-8)
current_dir = os.path.dirname(os.path.realpath(__file__))
ex = 'ex22'
DATA_NAME = 'BK_%s_ndata_500_Nx_129_delta_0.25_h_0.0078125' % ex
ntrain = 300
batch_size = 10
act_xi = 'ReLU'
# act_xi = 'GELU'
# act_xi = 'Tanh'
# act_xi = 'Softplus'
if act_xi == 'ReLU':
    act_fun_xi = torch.nn.ReLU
elif act_xi == 'GELU': 
    act_fun_xi = torch.nn.GELU
elif act_xi == 'Tanh': 
    act_fun_xi = torch.nn.Tanh
elif act_xi == 'Softplus': 
    act_fun_xi = torch.nn.Softplus
    
# layer_info = '64_5_64_5'
# phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
# phi_2_layer = [1, 64, 64, 64, 64, 64, 1]
layer_info = '128_5_64_5'
phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
phi_2_layer = [1, 64, 64, 64, 64, 64, 1]
    
path = os.path.join(current_dir, 'Results')
os.makedirs(path, exist_ok=True)  
fontsize = 15
plt.rcParams.update({'font.size': fontsize}) 
fig, ax = plt.subplots(figsize = (6,5))
start = 30
# colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred']
colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'palevioletred', 'gray']

Loss_train = np.zeros((N,))
loss_name = ["train", "valid", "test"]
for j in range(3):
    fontsize = 15
    plt.rcParams.update({'font.size': fontsize}) 
    fig, ax = plt.subplots(figsize = (6,6))
    start = 30
    # colors = ['darkorange', 'yellowgreen','deepskyblue', 'mediumslateblue', 'gray']
    for i in range(N):
        gap = int(h[i]/h0)
        # base_dir = 'Results/%s_gap_%s_ntrain_%s_%s' % (layer_info, gap, ntrain, act_xi)
        base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
        base_dir = os.path.join(current_dir, base_dir) 
        loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , loss_name[j]))
        # valid_loss = np.loadtxt('%s/loss_valid.txt' % (base_dir))
        # test_loss = np.loadtxt('%s/loss_test.txt' % (base_dir))
        
        
        plt.plot(loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
        # plt.plot(valid_loss[start:,0], valid_loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
        # plt.plot(test_loss[start:,0], test_loss[start:,1], color=colors[i], linewidth=1.5, label='h=%s' % h[i])
        # plt.plot(valid_loss[start:,0], valid_loss[start:,1], color='mediumorchid', linewidth=1.5, label='slope=1')
        # Loss[i] = loss[-1,1]
    
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss %s' % (loss_name[j]), fontsize=fontsize)
    # plt.ylabel(r'Training loss', fontsize=fontsize)
    # plt.ylabel(r'Valid loss', fontsize=fontsize)
    # plt.ylabel(r'Test loss', fontsize=fontsize)
    plt.yscale('log')
    # plt.title('noise=%s'% noise_std)
    plt.grid(True, which="both", ls="--", color='gray')
    plt.legend()
    plt.show()
    plt.tight_layout()
    name = '%s_loss_%s_ntrain_%s_gk.png' % (ex, loss_name[j], ntrain)
    # name = 'loss_valid.png'
    # name = 'loss_test.png'
    plt.savefig (os.path.join (path, name))
    print('---')
