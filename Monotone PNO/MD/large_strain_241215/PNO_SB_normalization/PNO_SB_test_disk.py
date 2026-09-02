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
import random
from CG_solver import *
from scipy.optimize import fsolve
from scipy.interpolate import griddata, RectBivariateSpline


# random.seed(1)
# torch.manual_seed(12)
# np.random.seed(1)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)

print(f'PID: {os.getpid()}')

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def main():
    parser = argparse.ArgumentParser('Train or Test Arg! ')
    parser.add_argument('--test', action='store_true')
    # parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--layer_info', type=str, default='64_4_256_4')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')

    # if_layer_train = '0'
    # # num_modules = 8
    # # start_layer = 8
    # # embed_dim = 32
    # # layer_info = '32_8_256_4'
    # num_modules = 16
    # start_layer = 16
    # embed_dim = 16
    # layer_info = '16_16_256_4'
    # phi_2_layer = [2, 256, 256, 256, 256, 1]
    # model = E_GCL_GKN(num_modules, embed_dim, phi_2_layer, nn.ReLU(), nn.Sigmoid()).to(device)
    
    layer_info = args.layer_info
    ker_layers, phi_layers = parse_layer_info(layer_info)
    model = EGKN(ker_layers, phi_layers).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    
    lrs = [1e-3]
    # lrs = [1e-4]
    # gammas = [0.9, 0.7, 0.5]
    # wds = [0.0, 1e-5, 1e-4]
    
    # lrs = [1e-5]
    gammas = [0.5]
    wds = [0.0]

    epochs = 1000
    scheduler_step = 50
    batch_size = 2
    batch_size_valid = batch_size
    batch_size_test = 1
    
    current_dir = os.path.dirname(os.path.realpath(__file__))

    DATA_NAME = 'test-250328'
    DATA = os.path.join(current_dir, "../ProcessData/%s.mat" % DATA_NAME)
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')
    data_mass = reader.read_field('mass')
    data_u = reader.read_field('disps')
    data_f = reader.read_field('forces')
    data_f = -data_f
    
    data_f = data_f/(data_mass.unsqueeze(2)/83.75)  # 83.75 is the volume
    
    ntest = 80
    # ntest = 30
    # data_u = data_u[50:,:,:]
    # data_f = data_f[50:,:,:]
    
    Nx = 1681
    m_fact = 3.01
    S = 41
    s = 21
    s_bc = 10
    # n = s ** 2
    # cond_f = torch.abs(data_x.view(S,S,2)[0,:,·]) <= (35 + 1e-10)
    mask_bc_logic = (data_x.view(S,S,2)[:,:,0])**2+ (data_x.view(S,S,2)[:,:,1])**2 <= (50**2 + 1e-10)
    mask_bc_logic = torch.stack((mask_bc_logic, mask_bc_logic), dim=2)
    mask_bc = mask_bc_logic.int()
    mask_bc_logic_cuda = mask_bc_logic.to(device)
    mask_bc_cuda = mask_bc.to(device)
       
    
    # for plot
    mask_bc_square = torch.abs(data_x.view(S,S,2)) <= (50 + 1e-10)
    mask_bc_square = mask_bc_square[:,:,0] & mask_bc_square[:,:,1]
    mask_bc_square = torch.stack((mask_bc_square, mask_bc_square), dim= 2)
    mask_bc_square_logic = mask_bc_square

    # data_x_plot = data_x.view(S,S,2)
    x_min = -50
    x_max = 50
    x = torch.linspace(x_min, x_max, 21)
    xi = torch.linspace(x_min, x_max, 201)
    Xi, Yi = torch.meshgrid(xi, xi)
    radius = 50
    mask_bc_logic_i = (Xi**2 + Yi**2) <= radius**2
    mask_bc_logic_i = torch.stack((mask_bc_logic_i, mask_bc_logic_i), dim= 2)
    
    # normalization
    norm = '1'
    norm_scale = 0.1
    data_x = data_x*norm_scale
    data_u = data_u*norm_scale
    data_f = data_f*norm_scale
    # mass_mean = torch.mean(data_mass)
    # data_f = data_f*100/ 83.75*mass_mean / data_mass.unsqueeze(0).unsqueeze(2) 
    # data_f = data_f/ 83.75 
    
    # ff = data_f[0,:,:].reshape(S,S,2)
    
    
    tests = range(ntest) 

    dx = data_x[1,1] - data_x[0,1]
    delta = m_fact * dx

    t1 = default_timer()

    # import mesh and dataset
    edge_index = {}
    edge_attr = {}
    
    meshgenerator = IrregularMeshGenerator(data_x, [S, S])
    edge_index = meshgenerator.ball_connectivity(float(delta))
    edge_attr = meshgenerator.attributes(theta=0)

    data_test = []
    for j in tests:
        data_test.append(Data(x=data_x, u=data_u[j, :, :], f=data_f[j, :, :], edge_index=edge_index, edge_attr=edge_attr, delta=delta, dx=dx))

    
    test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)


    # # base_dir = '%s/Results/cg-md-250212_41_index_b_ntrain200_256_4_512_4_lr0.001_dr0.5_step50_bs2_norm1' % current_dir
    base_dir = '%s/Results/MD_index_274_nvalid50_64_4_256_4_lr0.001_dr0.5_step100_bs1_norm1_z_seed_1_42' % current_dir

            
    model_path = os.path.join(base_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    def test(data, name, ntest):
        batch_size_test = 1
        test_loader = DataLoader(data, batch_size=batch_size_test, shuffle=False)
        j = 0
        rel_L2_err_test = 0
        err_b_test = np.zeros((ntest,))
        rel_err_b_test = np.zeros((ntest,))
        rel_max_err_b_test = np.zeros((ntest,))
        err_u_test = np.zeros((ntest,))
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            out = out.view(batch_size_test, -1)
            out = out.view( S, S, 2)
            # out = out[:, cond_f, :, :]
            # out = out[:, :, cond_f, :]
            out = out * mask_bc_cuda
            out = out.view(batch_size_test, -1)
            f_gt = batch.f.view( S, S, 2)
            # f_gt = f_gt[:, cond_f, :, :]
            # f_gt = f_gt[:, :, cond_f, :]
            f_gt = f_gt * mask_bc_cuda
            f_gt = f_gt.view(batch_size_test, -1)

            # err_test = myloss(out.view(batch_size_test, -1), f_gt.view(batch_size_test, -1)).item()
            # print('err_test:', err_test)
            
            out = out.cpu().detach()
            f_gt = f_gt.cpu().detach()
            # rel_L2_err_test += torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
            # err_b_test[j] = torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
            err_b_test[j] = torch.max(torch.abs(out-f_gt))
            rel_err_b_test[j] = torch.norm(out-f_gt)/torch.norm(f_gt)
            rel_max_err_b_test[j] = torch.max(torch.abs(out-f_gt))/torch.max(torch.abs(f_gt))
            # normalization
            err_b_test[j] = err_b_test[j]*1/norm_scale
            
            ###### solve u  ######                
            def func(u):
                u = torch.tensor(u, dtype=torch.float32)
                u_new = u_true.clone()
                u_new[mask_bc_logic] = u
                # u_new[s_bc: S-s_bc, s_bc: S-s_bc, :] = u
                
                u_new = u_new.view(-1, 2)

                input = Data(x=batch.x, u=u_new, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
                f = model(input).view(S, S, 2).cpu().detach()
                f_gt = batch.f.squeeze().view(S, S, 2).cpu().detach()
                y = (f-f_gt)[mask_bc_logic]
                
                return y
            
            u_true = batch.u.view(S, S, 2).detach().cpu()
            data_u0 = (1 - mask_bc) * u_true
            data_u0 = data_u0[mask_bc_logic]
            u, info, ier, msg = fsolve(func, data_u0, xtol=1e-6, full_output=True)
            
                
            u = torch.tensor(u, dtype=torch.float32)
            u_true_i = u_true[mask_bc_logic]
            err_u_test[j] = torch.norm(u-u_true_i.view(-1,))/torch.norm(u_true_i.view(-1,))
            
            print(f"{j}, err b: {err_b_test[j]:.2e}, rel err b: {rel_err_b_test[j]:.2e}, rel max err b: {rel_max_err_b_test[j]:.2e}, err u: {err_u_test[j]:.2e}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}\n")
            
            if_plot = 0
            if if_plot == 1:
                b_plot = out[0,:].view(S, S, 2)
                b_true_plot = f_gt[0,:].view(S, S, 2)
                # plt_b(b_true_plot, b_plot, base_dir, j, b_true_plot-b_plot, 'test')
                # u_plot = u.view(s, s, 2)
                u_true_plot = u_true.view(S, S, 2)
                u_plot = u_true_plot.clone()
                u_plot[mask_bc_logic] = u
                
                # normalization and truncation
                b_plot = 1/norm_scale* b_plot[mask_bc_square_logic].view(s,s,2)
                b_true_plot = 1/norm_scale* b_true_plot[mask_bc_square_logic].view(s,s,2)
                u_plot = 1/norm_scale*u_plot[mask_bc_square_logic].view(s,s,2)
                u_true_plot = 1/norm_scale*u_true_plot[mask_bc_square_logic].view(s,s,2)
                
                b_plot_i = np.stack((RectBivariateSpline(x, x, b_plot[:,:,0], kx=3, ky=3)(xi, xi), RectBivariateSpline(x, x, b_plot[:,:,1], kx=3, ky=3)(xi, xi)), axis=2)
                b_true_plot_i = np.stack((RectBivariateSpline(x, x, b_true_plot[:,:,0], kx=3, ky=3)(xi, xi), RectBivariateSpline(x, x, b_true_plot[:,:,1], kx=3, ky=3)(xi, xi)), axis=2)
                u_plot_i = np.stack((RectBivariateSpline(x, x, u_plot[:,:,0], kx=3, ky=3)(xi, xi), RectBivariateSpline(x, x, u_plot[:,:,1], kx=3, ky=3)(xi, xi)), axis=2)
                u_true_plot_i = np.stack((RectBivariateSpline(x, x, u_true_plot[:,:,0], kx=3, ky=3)(xi, xi), RectBivariateSpline(x, x, u_true_plot[:,:,1], kx=3, ky=3)(xi, xi)), axis=2)
                
                
                b_plot_masked = np.where(mask_bc_logic_i, b_plot_i, np.nan)
                b_true_plot_masked = np.where(mask_bc_logic_i, b_true_plot_i, np.nan)
                u_plot_masked = np.where(mask_bc_logic_i, u_plot_i, np.nan)
                u_true_plot_masked = np.where(mask_bc_logic_i, u_true_plot_i, np.nan)
                plt_b_u_test(u_plot_masked, u_true_plot_masked,  b_plot_masked, b_true_plot_masked, base_dir, j, name)
            
            j +=1
            
       
        
        mean_error_b = np.mean(err_b_test)
        mean_error_rel_b = np.mean(rel_err_b_test)
        mean_error_rel_max_b = np.mean(rel_max_err_b_test)
        mean_error_u = np.mean(err_u_test)
        print('Mean of max error for %s data: %.2e' % (name, mean_error_b) )
        print('Mean of relative L2 error for %s data b: %.2e' % (name, mean_error_rel_b) )
        print('Mean of relative max error for %s data b: %.2e' % (name, mean_error_rel_max_b) )
        print('Mean of relative L2 error for %s data u: %.2e' % (name, mean_error_u) )
        with open("%s/err_record.txt" % (base_dir), "a") as file:
            file.write(f"{name}: {mean_error_b:.2e}, {mean_error_rel_b:.2e}, {mean_error_rel_max_b:.2e}, {mean_error_u:.2e}\n")
        
        return err_b_test, err_u_test
        
    
    # err_b_train, err_u_train = test(data_train, 'train', ntrain)
    # err_b_valid, err_u_valid = test(data_valid, 'valid', nvalid)
    err_b_test, err_u_test = test(data_test, 'test_disk', ntest)
    
    


if __name__ == "__main__":
    
    main()
