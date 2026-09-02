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
from CG_solver import *

torch.set_default_dtype(torch.float64)

torch.manual_seed(12)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def generate_data_with_ksi_eta(data_x, data_u, data_f, m_fact, dx, s):
    """
    Generate Data objects with ksi and eta calculations for train/valid/test data.
    
    Args:
        data_x: Spatial coordinates (s, 1)
        data_u: Displacement data (n_samples, s, 1)
        data_f: Force data (n_samples, s, 1)
        m_fact: Multiplier factor for ksi range
        dx: Spatial step size
        s: Number of spatial points
    
    Returns:
        List of Data objects with calculated ksi and eta
    """
    n_samples = data_u.shape[0]
    
    # Calculate ksi range and values
    # ksi_range = torch.arange(-m_fact, m_fact + 1, 1).int()
    ksi_range = torch.arange(-m_fact, m_fact + 1, 1).int()
    ksi_range = ksi_range[ksi_range != 0]
    n_ksi = 2 * m_fact
    data_ksi = ksi_range * dx
    
    # Calculate eta values
    data_eta = torch.zeros((n_samples, s, n_ksi))
    for i in range(s):
        data_eta[:, i, :] = (data_u[:, m_fact + i + ksi_range].reshape(-1, 1, n_ksi) - 
                            data_u[:, m_fact + i].reshape(-1, 1, 1)).squeeze()
    
    # Repeat ksi for all samples and spatial points
    data_ksi = data_ksi.repeat(n_samples, s, 1)
    
    data_u = data_u[:, m_fact:s+m_fact].reshape(n_samples, s, 1)
    data_f = data_f[:, m_fact:s+m_fact].reshape(n_samples, s, 1)
    
    # Generate Data objects
    data_list = []
    for j in range(n_samples):
        data_list.append(Data(x=data_x, u=data_u[j, :, :], f=data_f[j, :, :], ksi=data_ksi[j, :, :], eta=data_eta[j, :, :]))
    
    return data_list

# def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
#     # return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))
#     return 0.995 ** epoch if epoch < epochs * 0.5 else 0.99 ** (epoch - epochs * 0.5)


def parse_layer_info(layer_info):
    """
    Parse layer_info string to generate ker_layers and phi_layers
    
    Args:
        layer_info (str): String in format 'ker_width_ker_layers_phi_width_phi_layers'
                         e.g., '128_4_256_2' means:
                         - ker_layers: 4 hidden layers with width 128 each
                         - phi_layers: 2 hidden layers with width 256 each
    
    Returns:
        tuple: (ker_layers, phi_layers) as lists
    """
    parts = layer_info.split('_')
    if len(parts) != 4:
        raise ValueError(f"layer_info must have 4 parts separated by '_', got: {layer_info}")
    
    ker_width = int(parts[0])
    ker_layers = int(parts[1])
    phi_width = int(parts[2])
    phi_layers = int(parts[3])
    
    # Generate ker_layers: [input_dim, hidden_layers..., output_dim]
    # input_dim=2, output_dim=1
    ker_layers_list = [1] + [ker_width] * ker_layers + [1]
    
    # Generate phi_layers: [input_dim, hidden_layers..., output_dim]  
    # input_dim=4, output_dim=1
    phi_layers_list = [4] + [phi_width] * phi_layers + [1]
    
    return ker_layers_list, phi_layers_list



def main():
    parser = argparse.ArgumentParser('Train or Test Arg! ')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--layer_info', default='128_4_128_4')
    parser.add_argument('--test_type', default='ID')
    # parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    print(f'PID: {os.getpid()}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')
        
    t1 = default_timer()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
    ex = 'ex35'
    DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
    DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
    
    # dx = 2**(-7)    # change
    dx_all = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
    for index in range(3, 4):
        dx = dx_all[index]
    
        ndata = 400
        ntrain = 300
        nvalid = 50
        ntest = 50
        
        # model and training parameters
        # layer_info = '64_5_64_5'
        # phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
        # phi_2_layer = [1, 64, 64, 64, 64, 64, 1]
        # layer_info = '128_5'
        # phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
        act_xi = 'ReLU'
        # act_xi = 'GELU'
        # act_xi = 'Tanh'
        # act_xi = 'Softplus'
        if act_xi == 'ReLU':
            act_fun_xi = torch.nn.ReLU()
        elif act_xi == 'GELU': 
            act_fun_xi = torch.nn.GELU()
        elif act_xi == 'Tanh': 
            act_fun_xi = torch.nn.Tanh()
        elif act_xi == 'Softplus': 
            act_fun_xi = torch.nn.Softplus()

        # model = E_GCL_GKN(phi_1_layer, torch.nn.Sigmoid()).to(device)
        
        layer_info = args.layer_info
        ker_layers, phi_layers = parse_layer_info(layer_info)
        model = E_GCL_GKN(phi_layers, ker_layers, act_fun_xi).to(device)
        
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        
        
        # lrs = [0.001]
        lrs = [0.01]
        gammas = [0.6]
        wds = [0.0]
        betas = [1.0]
        
        epochs = 3000
        # scheduler_step = 200
        scheduler_step = 100
        
        lr = [0.995, 0.998]
        lambda_fn = lambda epoch: lr[0] ** epoch if epoch < epochs * 0.3 else lr[0] **(epochs * 0.3)*lr[1] ** (epoch - epochs * 0.3)
        

        Nx = 257 
        h0 = 2**(-8)
        
        delta = 0.25
        # dx_cuda = dx.to(device)
        gap = int(dx/h0)
        m_fact = int(delta/dx)
        
        s = int((Nx-1)/gap)+1
        S = s+2*m_fact
        print(f'>> Training Mesh resolution: {s}x{s}')
        
        # batch_size = 50
        batch_size = 10
        batch_size2 = batch_size
        
        base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        reader = MatReader(DATA)
        data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
        data_x = data_X[m_fact:s+m_fact]
        data_u = reader.read_field('displacement')[:,::gap].reshape(-1, S)
        data_f = reader.read_field('bodyforce')[:,::gap].reshape(-1, S)
        
        # Prepare data for different splits
        # data_u = data_u_all[:,m_fact:s+m_fact].reshape(ndata,s,1)
        # data_f = data_f_all[:,m_fact:s+m_fact].reshape(ndata,s,1)
        
        # Split data into train/valid/test sets
        u_train = data_u[:ntrain, :]
        f_train = data_f[:ntrain, :]
        u_valid = data_u[-(nvalid+ntest):-ntest, :]
        f_valid = data_f[-(nvalid+ntest):-ntest, :]
        u_test = data_u[-ntest:, :]
        f_test = data_f[-ntest:, :]
        
        # Generate Data objects with ksi and eta calculations
        data_train = generate_data_with_ksi_eta(data_x, u_train, f_train, m_fact, dx, s)
        data_valid = generate_data_with_ksi_eta(data_x, u_valid, f_valid, m_fact, dx, s)
        data_test = generate_data_with_ksi_eta(data_x, u_test, f_test, m_fact, dx, s)
        
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(data_valid, batch_size=batch_size2, shuffle=False)
        test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)
        
        # Calculate lambda statistics for training data (for monitoring)
        compute_strain = False
        if compute_strain:
            ksi_range = torch.arange(-m_fact, m_fact + 1, 1).int()
            ksi_range = ksi_range[ksi_range != 0]
            n_ksi = 2 * m_fact
            data_ksi = ksi_range * dx
            data_eta = torch.zeros((ntrain, s, n_ksi))
            for i in range(s):
                data_eta[:, i, :] = (u_train[:, m_fact + i + ksi_range].reshape(-1, 1, n_ksi) - 
                                    u_train[:, m_fact + i].reshape(-1, 1, 1)).squeeze()
            
            ksi_plus_eta_norm = torch.abs(data_ksi + data_eta)
            ksi_norm = torch.abs(data_ksi)
            extension = ksi_plus_eta_norm - ksi_norm
            lambdaa = 1.0 + extension / (ksi_norm + 1e-9)

            lambda_min_data = torch.min(lambdaa)
            lambda_max_data = torch.max(lambdaa)
            print(lambda_min_data)
            print(lambda_max_data)


        
        t2 = default_timer()
        print(f'>> Preprocessing completed, time elapsed: {(t2 - t1): .2f}s')
        
        ##################################################################################################
        #                                        training
        ##################################################################################################
        if not args.test:

            for learning_rate in lrs:
                for scheduler_gamma in gammas:
                    for weight_decay in wds:
                        print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')

                        myloss = LpLoss(size_average=False)
                        best_epoch = np.zeros(1)
                        bl_train, bl_valid, bl_test = [], [], []

                        print("-" * 100)

                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        # LR_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
                        
                        model_filename = '%s/model.ckpt' % base_dir
                        ttrain, ttrain_disp, tvalid, tvalid_disp, ttest = [], [], [], [], []
                        best_train_loss = best_valid_loss = best_test_loss = 1e8
                        early_stop = 0
                        for ep in range(epochs):
                            model.train()
                            # optimizer = scheduler(optimizer,
                            #                     LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
                            # print(LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
                            optimizer = scheduler(optimizer, learning_rate*lambda_fn(ep))
                            # print(learning_rate*lambda_fn(ep))
                            # print('---')
                            t1 = default_timer()
                            train_mse = 0.0
                            train_l2 = 0.0
                            train_loss = 0.0
                            for batch in train_loader:
                                batch = batch.to(device)

                                optimizer.zero_grad()
                                out = model(batch)
                                #out = f_normalizer.decode(out)
                                out = out.reshape(batch_size, -1)
                                #crop 2delta layer from computed force
                                out = out.view(batch_size, s, 1)
                                out = out.view(-1,1)
                                #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                                f_gt = batch.f.view(batch_size, -1)
                                f_gt = f_gt.view(batch_size, s, 1)
                                f_gt = f_gt.view(-1, 1)

                                #mask = torch.abs(batch.f.view(batch_size, -1) - 0.5) <= (0.5 - 2 * delta_train)
                                #mask = mask.float()

                                mse = F.mse_loss(out.view(-1, 1), f_gt.view(-1, 1))
                                # mse.backward()
                                # loss = torch.norm(out.view(-1) - f_gt.view(-1), 2)
                                # loss = (myloss(out.view(batch_size, -1), -f_gt.view(batch_size, -1)) +
                                #         model.egkn_conv.phi_mlp(torch.tensor(1.0).unsqueeze(0).to('cuda')) ** 2.0 +
                                #         1.0/model.egkn_conv.phi_mlp(torch.tensor(0.0).unsqueeze(0).to('cuda')) ** 2.0)
                                # loss = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # loss = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                loss = torch.sum((out.view(batch_size, -1)-f_gt.view(batch_size, -1))**2)
                                loss.backward()

                                # l2 = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # l2 = torch.norm(out.view(-1) + f_gt.view(-1), 2)/torch.norm(f_gt.view(-1), 2)
                                #l2.backward()

                                optimizer.step()
                                # scheduler.step()
                                train_loss += loss.item()
                                train_mse += mse.item()
                                train_l2 += loss.item()

                            ttrain.append([ep, train_l2 / ntrain/s])
                            t2 = default_timer()

                            model.eval()
                            valid_l2 = 0.0
                            test_l2 = 0.0
                            if train_l2 / ntrain/s < best_train_loss:
                                with torch.no_grad():
                                    for batch in valid_loader:
                                        batch = batch.to(device)
                                        out = model(batch)
                                        #out = f_normalizer.decode(out)
                                        out = out.reshape(batch_size, -1)
                                        # crop 2delta layer from computed force
                                        out = out.view(batch_size, s, 1)
                                        out = out.view(-1, 1)
                                        #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                                        f_gt = batch.f.view(batch_size, -1)
                                        f_gt = f_gt.view(batch_size, s, 1)
                                        f_gt = f_gt.view(-1, 1)

                                        # valid_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()
                                        # valid_l2 = torch.sum(torch.norm(out.view(batch_size, -1)-f_gt.view(batch_size, -1), 2, 1))
                                        valid_l2 = torch.sum((out.view(batch_size, -1)-f_gt.view(batch_size, -1))**2).item()

                                    for batch in test_loader:
                                        batch = batch.to(device)
                                        out = model(batch)
                                        #out = f_normalizer.decode(out)
                                        out = out.reshape(batch_size, -1)
                                        # crop 2delta layer from computed force
                                        out = out.view(batch_size, s, 1)
                                        out = out.view(-1, 1)
                                        out = out.reshape(batch_size, -1)
                                        f_gt = batch.f.view(batch_size, -1)
                                        f_gt = f_gt.view(batch_size, s, 1)
                                        f_gt = f_gt.view(-1, 1)

                                        # test_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()
                                        # test_l2 = torch.sum(torch.norm(out.view(batch_size, -1)-f_gt.view(batch_size, -1), 2, 1))
                                        test_l2 = torch.sum((out.view(batch_size, -1)-f_gt.view(batch_size, -1))**2).item()
                                        

                                tvalid.append([ep, valid_l2 / nvalid/s])
                                ttest.append([ep, test_l2 / ntest/s])

                                if valid_l2 / nvalid/s < best_valid_loss:
                                    early_stop = 0
                                    best_train_loss = train_l2 / ntrain/s
                                    best_valid_loss = valid_l2 / nvalid/s
                                    best_test_loss = test_l2 / ntest/s
                                    best_epoch = ep
                                    torch.save(model.state_dict(), model_filename)
                                    
                                    '''
                                    # dump kernel & force
                                    x_values = torch.linspace(-1, 1, 7)
                                    y_values = torch.linspace(-1, 1, 7)
                                    x_mesh, y_mesh = torch.meshgrid(x_values, y_values)
                                    x_mesh = x_mesh.permute(1, 0)
                                    y_mesh = y_mesh.permute(1, 0)

                                    grid = torch.cat([x_mesh.view(7, 7, 1), y_mesh.view(7, 7, 1)], dim=2)
                                    inp = torch.sqrt(grid[:, :, 0]**2 + grid[:, :, 1]**2).to('cuda')
                                    kernel_values = model.egkn_conv.kernel(inp.unsqueeze(2))
                                    visible = (x_mesh ** 2.0 + y_mesh ** 2.0 < (1 + 1e-4) ** 2)
                                    kmax = max(kernel_values.detach().cpu().numpy()[visible])
                                    kmin = min(kernel_values.detach().cpu().numpy()[visible])
                                    img = plt.imshow(kernel_values.detach().cpu().numpy(), vmin=kmin, vmax=kmax, cmap='plasma', interpolation='spline16')
                                    cbar = plt.colorbar(img)
                                    plt.savefig('%s/kernel_interp.png' % base_dir, format='png')
                                    plt.clf()

                                    omega = kernel_values.squeeze().detach().cpu().numpy()
                                    np.savetxt('%s/updating_kernel.txt' % base_dir, omega)
                                    '''
                                    
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1):.2f}s, train err *1e8: {(train_l2 / ntrain/s)*1e8:.4f}, '
                                        f'valid err*1e8: {(valid_l2 / nvalid/s)*1e8:.4f} , test err*1e8: {(test_l2 / ntest/s)*1e8:.4f}')
                                else:
                                    early_stop += 1
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1): .2f}s, train err*1e8: {(train_l2 / ntrain/s)*1e8: .4f} '
                                        f'(best*1e8: {best_train_loss*1e8: .4f}/{best_valid_loss*1e8: .4f})')
                            else:
                                early_stop += 1
                                print(
                                    f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: '
                                    f'{(t2 - t1): .2f}s, train err*1e8: {(train_l2 / ntrain/s)*1e8: .4f} '
                                    f'(best*1e8: {best_train_loss*1e8: .4f}/{best_valid_loss*1e8: .4f})')

                            if early_stop > 100: break

                        bl_train.append(best_train_loss)
                        bl_valid.append(best_valid_loss)
                        bl_test.append(best_test_loss)
                        with open('%s/loss_train.txt' % (base_dir), 'w') as file:
                            np.savetxt(file, ttrain)
                        with open('%s/loss_valid.txt' % (base_dir), 'w') as file:
                            np.savetxt(file, tvalid)
                        with open('%s/loss_test.txt' % (base_dir), 'w') as file:
                            np.savetxt(file, ttest)

                        print("-" * 100)
                        print("-" * 100)
                        print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
                        print(f'>> Best train error*1e8: {best_train_loss*1e8: .4f}')
                        print(f'>> Best valid error*1e8: {best_valid_loss*1e8: .4f}')
                        print(f'>> Best test error*1e8: {best_test_loss*1e8: .4f}')
                        print(f'>> Best epoch: {best_epoch}')
                        print("-" * 100)
                        print("-" * 100)

                        f = open("training_record.txt", "a")
                        f.write(f'{ntrain}, {lrs}, {lr}, {layer_info}, {dx}: ')
                        f.write(','.join(str(err) for err in bl_train))
                        f.write(',')
                        f.write(','.join(str(err) for err in bl_valid))
                        f.write(',')
                        f.write(','.join(str(err) for err in bl_test))
                        f.write(',')
                        f.write(f'{best_epoch}\n')
                        f.close()

            print('>> Training Completed!!')
            
        else: 
        ################################################################
        # testing
        ################################################################
            model_path = os.path.join(base_dir, 'model.ckpt')
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            test_type = args.test_type
            
            if test_type == 'ID':
                data_u_test = data_u[-ntest:, :]
                data_f_test = data_f[-ntest:, :]
                data_test = generate_data_with_ksi_eta(data_x, data_u_test, data_f_test, m_fact, dx, s)
                
            elif test_type == 'OOD':
                pass
                    
            mask_bc = torch.zeros((S,1))
            mask_bc[m_fact:s+m_fact] = 1
            
            def test_solver(data, ntest, base_dir, name):
                # batch_size = 1 
                # test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
                # j = 0
                err_b = torch.zeros((ntest,))
                err_u = torch.zeros((ntest,))
                for j in range(ntest):
                    data_j = data[j]
                    data_input = Data(x=data_X , u=data_j.u, f=data_j.f, ksi=data_j.ksi, eta=data_j.eta).to(device)
                    # batch = batch.to(device)
                    out = model(data_input)
                    out = out.view(s, 1)
                    out = out.cpu().detach().numpy()
                    f_gt = data_j.f.view(s, 1)
                    f_gt = f_gt.cpu().detach().numpy()
                    err_b[j] = np.linalg.norm(out-f_gt)/np.linalg.norm(f_gt)
                    
                    # u_true = batch.u.cpu().detach()
                    u_true = data_u_test[j,:].reshape(S,1)
                    data_u0 = torch.zeros_like(u_true)[m_fact:s+m_fact]  # zero initial displacement
                    
                    def func(u):
                        u = torch.tensor(u, dtype=torch.float64)
                        full_u = (1-mask_bc)*u_true
                        full_u[m_fact:s+m_fact, 0] = u
                        # full_u[m_fact] = u_true[m_fact]
                        # full_u[s+m_fact] = u_true[s+m_fact]
                        # data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :], eta=data_eta[0, :, :]).to(device)
                        # data = Data(x=data_X , u=full_u, f=batch.f[j,m_fact:s+m_fact,:], ksi=batch.ksi[j, :, :], eta=batch.eta[j, :, :]).to(device)
                        new_data = generate_data_with_ksi_eta(data_x, full_u.reshape(1,S), data_f_test[j,:].reshape(1,S), m_fact, dx, s)
                        new_data = new_data[0].to(device)
                        # data = Data(x=data_X, u=full_u, f=data_j.f, ksi=data_j.ksi, eta=data_j.eta).to(device)
                        y = model(new_data)-new_data.f.squeeze()
                        return y.cpu().detach().numpy()
                    
                    u, info, ier, msg = scipy.optimize.fsolve(func, data_u0, xtol=1e-12, full_output=True)
                    
                    
                    u = torch.tensor(u, dtype=torch.float64)
                    uh = u_true.clone()
                    uh[m_fact:s+m_fact,0] = u
                    
                    err_u[j] = (torch.norm(uh-u_true)/torch.norm(u_true)).item()
                    
                    print(f"{j}, {err_b[j]}, {err_u[j]}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}\n")
                    
                    plot_test = 1
                    if plot_test == 1 and j<=5:                
                        plot_u(data_X, u_true.squeeze(), uh, j, base_dir, 'test')
                        plot_b(data_x, f_gt.squeeze(), out.squeeze(), j, base_dir, 'test')
                    
                    # j += 1
                    
                print('Relative L2 error for %s b: %s' % (name, torch.mean(err_b)) )
                print('Relative L2 error for %s u: %s' % (name, torch.mean(err_u)) )
                
                # Save error arrays to files
                np.savetxt('%s/err_b_%s.txt' % (base_dir, name), err_b)
                np.savetxt('%s/err_u_%s.txt' % (base_dir, name), err_u)
                print(f'Error arrays saved to {base_dir}/err_b_{name}.txt and {base_dir}/err_u_{name}.txt')
            
             
            test_solver(data_test, ntest, base_dir, 'test')       
            
            #******************************************** plot k2(xi) ***************************
            # # lambda_min_data = 0.8
            # # lambda_max_data = 1.2
            # delta = 0.25
            # mu = 0.3846
            # c = 2*mu/math.pi/delta**2
            # # g_fun = lambda x: np.ones_like(x)
            # # g_fun = lambda x: x
            # g_fun = lambda x: x*np.exp(-50*x**2)
            
            # xi_norm = torch.linspace(dx, delta, 100)
            # xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
            # k2_NN = (model.phi_2(xi_norm_cuda)).cpu().detach().numpy()
            # k2_true = 2*c/xi_norm*g_fun(xi_norm)
            
            # fontsize = 15
            # # plt.plot(xi_norm, k2_NN_normalized, color='darkorange', linestyle='--', linewidth=2)
            # plt.plot(xi_norm, k2_NN, color='darkorange', linestyle='--', linewidth=2, label='NN')
            # plt.plot(xi_norm, k2_true, color='k', linewidth=2, label='true')
            # plt.legend()
            # plt.xlabel(r'$|\xi|$', fontsize=fontsize)
            # plt.ylabel(r'$k_2(\xi)$', fontsize=fontsize)
            # plt.savefig('%s/%s_k2_%s.png' % (base_dir, ex, dx), format='png')
            # plt.close()
            


if __name__ == "__main__":
    
    main()

