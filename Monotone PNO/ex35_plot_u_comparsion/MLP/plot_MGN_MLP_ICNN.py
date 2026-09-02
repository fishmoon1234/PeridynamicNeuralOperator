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

# Add path for ICNN modules
icnn_path = os.path.join(os.path.dirname(__file__), '../PNO_ICNN_ex35_k1')
sys.path.append(icnn_path)
try:
    from convex_modules import ICNN_init
except ImportError:
    # If convex_modules is not available, we'll define a placeholder
    class ICNN_init(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(1, 1)
        
        def forward(self, x):
            return self.linear(x)

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

parser = argparse.ArgumentParser()
# parser.add_argument('--layer_info', type=str, default='20_4')
# parser.add_argument('--layer_info', type=str, default='128_5')
# parser.add_argument('--act_xi', type=str, default='ReLU')

# args = parser.parse_args()
# layer_info = args.layer_info
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_dir = os.path.dirname(os.path.realpath(__file__))
N = 4
h = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
i0 = 5
h0 = 2**(-8)
ntrain = 300

lrs = [0.01]
lr = [0.99, 0.998]

# model and training parameters
batch_size = 10
batch_size2 = batch_size
    
    
# model and training parameters
# load MLP model
layer_info = '128_5'
phi_1_layer = parse_layer_info(layer_info)
model_NN = E_GCL_GKN(phi_1_layer, torch.nn.ReLU).to(device)
total_params = sum(p.numel() for p in model_NN.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

# load MGN model
layer_info = '128_5'
phi_1_layer = parse_layer_info(layer_info)
model_MGN = E_GCL_GKN_MGN(phi_1_layer, torch.nn.Sigmoid()).to(device)
total_params = sum(p.numel() for p in model_MGN.parameters() if p.requires_grad)
print(f"MGN Total number of parameters: {total_params}")
model_MGN_path = '%s/../MGN/Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.995, 0.998]_gap_1_Sigmoid_seed_42' % current_dir
model_path = os.path.join(model_MGN_path, 'model.ckpt')
model_MGN.load_state_dict(torch.load(model_path))
model_MGN.eval()

# Load ICNN model
model_ICNN = E_GCL_GKN_ICNN(phi_1_layer, torch.nn.Softplus()).to(device)
total_params = sum(p.numel() for p in model_ICNN.parameters() if p.requires_grad)
print(f"ICNN Total number of parameters: {total_params}")
model_ICNN_path = '%s/../ICNN/Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.99, 0.998]_gap_1_Softplus_seed_42' % current_dir
model_path = os.path.join(model_ICNN_path, 'model.ckpt')
model_ICNN.load_state_dict(torch.load(model_path))
model_ICNN.eval()


ex = 'ex35'
DATA = '%s/../DATA/BK_ex35_ndata_400_Nx_257_delta_0.25_h_0.00390625.mat' % current_dir
Nx = 257

delta = 0.25
mu = 0.3846
c = 2*mu/math.pi/delta**2
g_fun = lambda x: np.exp(-50*x**2)*(delta-np.abs(x))
k1_fun = lambda x: np.pi*(x - x**(-3)) + np.sin(np.pi*x)


L2_err_k1 = np.zeros((N,))
rel_L2_err_k1 = np.zeros((N,))
L2_rho_err_k1 = np.zeros((N,))
rel_L2_rho_err_k1 = np.zeros((N,))
Rho, Xi, Lambda_rho_all = [], [], []
lambda_min_data = np.zeros((N,))
lambda_max_data = np.zeros((N,))
lambda_ave, Data_ksi, Weights = [], [], []
loss_train = np.zeros((N,))
ATA_cond = np.zeros((N,))
err_NN = np.zeros((N,))
err_MGN = np.zeros((N,))
err_ICNN = np.zeros((N,))

# Initialize variables for plotting
Lambdaa = torch.linspace(0.5, 5.5, 100).reshape(-1,1)
k1_true = None
k1_NN = None
k1_MGN = None
k1_ICNN = None

for i in range(3,N):
    gap = int(h[i]/h0)
    # base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_gap_%s_%s_L1loss_seed_43' % (ex, layer_info, ntrain, lrs, lr, gap, act_xi)
    # chosed model
    base_dir = 'Results/ex35_128_5_ntrain_300_lrs_[0.01]_lr_[0.99, 0.998]_gap_1_ReLU_seed_42'
    base_dir = os.path.join(current_dir, base_dir)
    model_path = os.path.join(base_dir, 'model.ckpt')
    model_NN.load_state_dict(torch.load(model_path))
    model_NN.eval()
    
    loss = np.loadtxt('%s/loss_%s.txt' % (base_dir , 'train'))
    loss_train[i] = loss[-1,1]

    m_fact = int(delta/h[i])
    
    s = int((Nx-1)/gap)+1
    S = s+2*m_fact
    
    # ntrain = 100
    reader = MatReader(DATA)
    data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
    data_x = data_X[m_fact:s+m_fact]
    data_u = reader.read_field('displacement')[:ntrain,::gap].reshape(-1, S)
    data_f = reader.read_field('bodyforce')[:ntrain,::gap].reshape(-1, S)
    
    ksi_range = torch.range(-m_fact, m_fact).int()
    # n_ksi = 2*m_fact+1
    ksi_range = ksi_range[ksi_range != 0]
    n_ksi = 2*m_fact
    data_ksi = ksi_range*h[i]
    data_eta = torch.zeros((ntrain, s, n_ksi))
    for ii in range(s):
        data_eta[:,ii,:] = (data_u[:,m_fact+ii+ksi_range].reshape(-1,1,n_ksi)-data_u[:,m_fact+ii].reshape(-1,1,1)).squeeze()
        
    # A = data_eta.reshape(-1, n_ksi)
    # print(np.linalg.cond(np.dot(A.T, A)))
    ksi_plus_eta = data_ksi+data_eta
    ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
    ksi_norm = torch.abs(data_ksi)
    extension = ksi_plus_eta_norm - ksi_norm
    # lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
    lambdaa = 1.0 + extension / (ksi_norm)
    
    Data_ksi.append(data_ksi)
    lambda_ave.append(torch.mean(lambdaa, dim=[0,1]))
    
    total_nlambda = lambdaa.reshape(-1,).size(0)
    
    # lambda_min_all_data, lambda_max_all_data = 0.5, 1.5
    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    Nlambda = 50
    Lambda_rho = torch.linspace(lambda_min_data[i], lambda_max_data[i], Nlambda)
    dlambda_rho = Lambda_rho[1]- Lambda_rho[0]
        
    # weights = torch.abs((lambdaa-lambdaa**(-3))*ksi_plus_eta/ksi_plus_eta_norm)
    # weights = torch.abs((g_fun(ksi_norm)/ksi_norm)*ksi_plus_eta/ksi_plus_eta_norm)
    ksi_norm4rho = torch.tile(ksi_norm,(lambdaa.size(0),lambdaa.size(1),1))
    weights = torch.zeros((Nlambda,1))
    for k in range(Nlambda-1):
        indices = torch.nonzero((lambdaa >= Lambda_rho[k]) & (lambdaa <= Lambda_rho[k+1]), as_tuple=False)
        if indices.numel() != 0:
            index = (indices[:,0],indices[:,1],indices[:,2])
            weights[k,0] = torch.sum(torch.abs((g_fun(ksi_norm4rho[index]))*ksi_plus_eta[index]/ksi_plus_eta_norm[index]))/total_nlambda
        
    
    # rho = torch.sum(weights, axis=(0,1))/ntrain/s
    # rho = (torch.flip(rho[:m_fact], dims=[0])+rho[m_fact:])/2
    rho = weights
    # rho = weights*dlambda_rho
    Rho.append(rho)
    Lambda_rho_all.append(Lambda_rho)
    # Xi.append(data_ksi[m_fact:])
    # Weights.append(weights)s
    
    Lambda_rho = Lambda_rho.reshape(-1,1)
    Lambda_rho_cuda = Lambda_rho.to('cuda')
    Lambda_rho_1_cuda = torch.ones_like(Lambda_rho_cuda)
    # k1_true = 2*c*(Lambda_rho-Lambda_rho**(-3)).reshape(-1,1)
    k1_true = k1_fun(Lambda_rho)
    k1_NN = (model_NN.phi_1(Lambda_rho_cuda)- model_NN.phi_1(Lambda_rho_1_cuda)).reshape(-1,1).cpu().detach()
    k1_MGN = (model_MGN.phi_MGN(Lambda_rho_cuda)- model_MGN.phi_MGN(Lambda_rho_1_cuda)).reshape(-1,1).cpu().detach()
    # k1_ICNN = (model_ICNN.phi_1(Lambda_rho_cuda)- model_ICNN.phi_1(Lambda_rho_1_cuda)).reshape(-1,1).cpu().detach()
    
    Lambda_rho_cuda.requires_grad_(True)
    Lambda_rho_1_cuda.requires_grad_(True)
    icnn_output = model_ICNN.icnn(Lambda_rho_cuda)
    phi_NN1 = torch.autograd.grad(icnn_output, Lambda_rho_cuda, 
                                grad_outputs=torch.ones_like(icnn_output), 
                                create_graph=True, retain_graph=True)[0]
    
    # Forward pass through ICNN for Lambdaa_1_cuda (lambda=1)
    icnn_output_1 = model_ICNN.icnn(Lambda_rho_1_cuda)
    phi_NN1_1 = torch.autograd.grad(icnn_output_1, Lambda_rho_1_cuda, 
                                grad_outputs=torch.ones_like(icnn_output_1), 
                                create_graph=True, retain_graph=True)[0]
    
    # Calculate g_NN as (phi_NN1 - phi_NN1_1)
    k1_ICNN = (phi_NN1 - phi_NN1_1).reshape(-1,1)
    k1_ICNN = k1_ICNN.cpu().detach()
    err_NN[i] = torch.sqrt(torch.sum(rho*dlambda_rho*(k1_NN-k1_true)**2))/torch.sqrt(torch.sum(rho*dlambda_rho*k1_true**2))
    err_MGN[i] = torch.sqrt(torch.sum(rho*dlambda_rho*(k1_MGN-k1_true)**2))/torch.sqrt(torch.sum(rho*dlambda_rho*k1_true**2))
    err_ICNN[i] = torch.sqrt(torch.sum(rho*dlambda_rho*(k1_ICNN-k1_true)**2))/torch.sqrt(torch.sum(rho*dlambda_rho*k1_true**2))
    
    
    # Lambdaa = Lambda_rho.reshape(-1,1)
    Lambdaa = torch.linspace(0.5, 5.5, 100).reshape(-1,1)
    Lambdaa_cuda = Lambdaa.to('cuda')
    Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
    # k1_true = 2*c*(Lambda_rho-Lambda_rho**(-3)).reshape(-1,1)
    k1_true = k1_fun(Lambdaa)
    k1_NN = (model_NN.phi_1(Lambdaa_cuda)- model_NN.phi_1(Lambdaa_1_cuda)).reshape(-1,1).cpu().detach()
    k1_MGN = (model_MGN.phi_MGN(Lambdaa_cuda)- model_MGN.phi_MGN(Lambdaa_1_cuda)).reshape(-1,1).cpu().detach()
    # k1_ICNN = (model_ICNN.phi_1(Lambdaa_cuda)- model_ICNN.phi_1(Lambdaa_1_cuda)).reshape(-1,1).cpu().detach()
    Lambdaa_cuda.requires_grad_(True)
    Lambdaa_1_cuda.requires_grad_(True)
    icnn_output = model_ICNN.icnn(Lambdaa_cuda)
    phi_NN1 = torch.autograd.grad(icnn_output, Lambdaa_cuda, 
                                grad_outputs=torch.ones_like(icnn_output), 
                                create_graph=True, retain_graph=True)[0]
    
    # Forward pass through ICNN for Lambdaa_1_cuda (lambda=1)
    icnn_output_1 = model_ICNN.icnn(Lambdaa_1_cuda)
    phi_NN1_1 = torch.autograd.grad(icnn_output_1, Lambdaa_1_cuda, 
                                grad_outputs=torch.ones_like(icnn_output_1), 
                                create_graph=True, retain_graph=True)[0]
    
    # Calculate g_NN as (phi_NN1 - phi_NN1_1)
    k1_ICNN = (phi_NN1 - phi_NN1_1).reshape(-1,1)
    k1_ICNN = k1_ICNN.cpu().detach()
    # weights = weights.reshape(-1,n_ksi)
    
    lambda_min_data[i] = torch.min(lambdaa[:ntrain, :,:])
    lambda_max_data[i] = torch.max(lambdaa[:ntrain, :,:])
    print('min lambda: %s, max lambda: %s:' % (lambda_min_data[i], lambda_max_data[i]))
    
    
    
    
print("MGN errors:", err_MGN)
print("NN errors:", err_NN)
print("ICNN errors:", err_ICNN)

if_plot = 1
if if_plot == 1:

    # Enhanced color palette and styling
    colors = ['#FF6B35', '#004E98', '#8E44AD', '#2ECC71', '#F39C12', '#95A5A6']  # Modern color palette
    fontsize = 22
    linewidth = 3.0
    markersize = 8

    # Set up matplotlib styling for better appearance
    plt.rcParams.update({
        'font.size': fontsize,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.edgecolor': 'gray', 
        # 'xtick.major.width': 2,
        # 'ytick.major.width': 2,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize+2,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize-2,
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',
        'grid.alpha': 0.5,
        'grid.color': 'gray',
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot with enhanced styling
    ax.plot(Lambdaa, k1_true, color='black', linewidth=linewidth+0.5, 
            label='True', zorder=4, alpha=0.9)
    ax.plot(Lambdaa, k1_MGN, color=colors[0], linestyle='--', linewidth=linewidth, 
            label='MGN', marker='o', markevery=10, markersize=markersize-2, zorder=3)
    ax.plot(Lambdaa, k1_NN, color=colors[1], linestyle='-.', linewidth=linewidth, 
            label='MLP', marker='s', markevery=10, markersize=markersize-2, zorder=2)
    ax.plot(Lambdaa, k1_ICNN, color=colors[2], linestyle=':', linewidth=linewidth, 
            label='ICNN', marker='^', markevery=10, markersize=markersize-1, zorder=1)
    
    # Enhanced training data coverage visualization
    lambda_min_data, lambda_max_data = 0.8109, 3.3456
    y_lim = ax.get_ylim()
    ax.set_ylim(y_lim[0], y_lim[1])

    # Add shaded region for training data coverage
    ax.axvspan(lambda_min_data, lambda_max_data, alpha=0.15, color='lightblue', zorder=0)

    # Vertical lines for data boundaries
    ax.axvline(x=lambda_min_data, color='#34495E', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.axvline(x=lambda_max_data, color='#34495E', linestyle='--', linewidth=2.5, alpha=0.8)

    # Enhanced annotations
    midpoint = (lambda_min_data + lambda_max_data) / 2
    y_pos = y_lim[0] + 0.2 * (y_lim[1] - y_lim[0])

    # Annotation text with better styling
    ax.text(midpoint, y_pos, 'Training Data\nCoverage', 
            fontsize=fontsize-4, ha='center', va='center', 
            color='#2C3E50', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                    edgecolor='#34495E', alpha=0.8))

    # Lambda value annotations
    y_annotation = y_lim[0] + 0.05 * (y_lim[1] - y_lim[0])
    ax.text(lambda_min_data, y_annotation, f'λ={lambda_min_data:.3f}', 
            ha='center', va='bottom', fontsize=fontsize-4, color='#2C3E50', fontweight='bold')
    ax.text(lambda_max_data, y_annotation, f'λ={lambda_max_data:.3f}', 
            ha='center', va='bottom', fontsize=fontsize-4, color='#2C3E50', fontweight='bold')

    # Enhanced legend
    ax.legend(loc='lower right', framealpha=0.95, fancybox=True, shadow=True, bbox_to_anchor=(0.98, 0.02))

    # Enhanced axis labels
    ax.set_xlabel(r'$\lambda$', fontweight='bold')
    ax.set_ylabel(r'$g(\lambda)$', fontweight='bold')

    # # Grid for better readability
    # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    # Enhanced title
    # ax.set_title('Comparison of Neural Network Models for $g(\\lambda)$', fontsize=fontsize-2, fontweight='bold', pad=20)

    plt.tight_layout()
    # ax.tick_params(axis='both', which='both', length=0)

    # Save with higher DPI for better quality
    plt.savefig('%s/%s_g_NN_MGN_ICNN_%s.png' % (current_dir, ex, layer_info), 
                format='png', dpi=300, bbox_inches='tight', facecolor='white')
    # plt.savefig('%s/%s_k_NN_MGN_lambda_%s_%s.png' % (base_dir, ex, lambda_min, lambda_max), format='png')

    print("Plot saved successfully!")
