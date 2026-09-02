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


torch.manual_seed(1)
np.random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


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
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')
        
    t1 = default_timer()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % current_dir
    ex = 'ex37'
    # DATA_NAME = 'BK_%s_ndata_500_Nx_129_delta_0.25_h_0.0078125' % (ex)
    DATA_NAME = 'BK_%s_ndata_400_Nx_257_delta_0.25_h_0.00390625' % (ex)
    DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
    
    dx_all = np.array([2**(-5), 2**(-6), 2**(-7), 2**(-8)])
    # Lrs = [0.005, 0.01, 0.01, 0.008]
    lr_all = [0.99, 0.99, 0.99, 0.99]
    for index in range(2,4):
        dx = dx_all[index]    # change

        gammas = [0.8]
        wds = [0.0]
        betas = [1.0]
        scheduler_step = 100
        
        early_stop_num = 50
        epochs = 3000
        lrs = [0.005]
        lr = [0.994, 0.998]

        ndata = 400
        ntrain = 300
        nvalid = 50
        ntest = 50
        
        # model and training parameters
        batch_size = 10
        batch_size2 = batch_size
        batch_size_test = 1
        
        # layer_info = '128_4_128_4'
        # phi_2_layer = [1, 128, 128, 128, 128, 1]
        layer_info = '128_4_256_4'
        phi_2_layer = [1, 256, 256, 256, 256, 1]
        embed_dim = 128
        num_layers = 4
        start_layer = 4
        # layer_add_times = 2

        # model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, alpha).to(device)
        model = E_GCL_GKN(num_layers, embed_dim, phi_2_layer, torch.nn.ReLU(), nn.Sigmoid()).to(device)
        # print(model)
        
        MGN_params = sum(p.numel() for p in model.phi_MGN.parameters() if p.requires_grad)
        print(f"Number of parameters for MGN: {MGN_params}")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")

        lambda_fn = lambda epoch: lr[0] ** epoch if epoch < epochs * 0.3 else lr[0] **(epochs * 0.3)*lr[1] ** (epoch - epochs * 0.3)

        Nx = 257 
        h0 = 2**(-8)
        delta = 0.25
        
        gap = int(dx/h0)
        m_fact = int(delta/dx)
        
        s = int((Nx-1)/gap)+1
        S = s+2*m_fact
        print(f'>> Training Mesh resolution: {s}x{s}')
        
        reader = MatReader(DATA)
        data_X = reader.read_field('coords')[:,::gap].reshape(S,1)
        data_x = data_X[m_fact:s+m_fact]
        data_u = reader.read_field('displacement')[:,::gap].reshape(-1, S)
        data_f = reader.read_field('bodyforce')[:,::gap].reshape(-1, S)
        
        data_u_test = data_u[-ntest:, :].reshape(-1, S,1)
        data_f_test = data_f[-ntest:, :].reshape(-1, S,1)
        
        # base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_gamma%s_step%s_gap_%s' % (ex, layer_info, ntrain, lrs, gammas, scheduler_step, gap)
        base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_Sigmoid_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # dx = data_x[1]-data_x[0]
        # delta = m_fact_train * dx
        ksi_range = torch.range(-m_fact, m_fact).int()
        # n_ksi = 2*m_fact+1
        ksi_range = ksi_range[ksi_range != 0]
        n_ksi = 2*m_fact
        data_ksi = ksi_range*dx
        data_eta = torch.zeros((ndata, s, n_ksi))
        for i in range(s):
            data_eta[:,i,:] = (data_u[:,m_fact+i+ksi_range].reshape(-1,1,n_ksi)-data_u[:,m_fact+i].reshape(-1,1,1)).squeeze()
            
        A = data_eta.reshape(-1, n_ksi)
        print(np.linalg.cond(np.dot(A.T, A)))
            
        ksi_plus_eta_norm = torch.abs(data_ksi+data_eta)
        ksi_norm = torch.abs(data_ksi)
        extension = ksi_plus_eta_norm - ksi_norm
        lambdaa = 1.0 + extension / (ksi_norm + 1e-9)

        lambda_min_data = torch.min(lambdaa[:ntrain, :,:])
        lambda_max_data = torch.max(lambdaa[:ntrain, :,:])
        print(lambda_min_data)
        print(lambda_max_data)

        # separate training, validation and testing sets
        # x_train = data_x[0, :, :]
        # data_x = data_x.repeat(ndata,1,1)
        data_ksi = data_ksi.repeat(ndata,s,1)
        data_u = data_u[:,m_fact:s+m_fact].reshape(ndata,s,1)
        data_f = data_f[:,m_fact:s+m_fact].reshape(ndata,s,1)
        
        x_train = data_x
        u_train = data_u[:ntrain, :, :]
        f_train = data_f[:ntrain, :, :]
        ksi_train = data_ksi[:ntrain, :, :]
        eta_train = data_eta[:ntrain, :, :]

        x_valid = x_train
        u_valid = data_u[-(nvalid+ ntest):-ntest, :, :]
        f_valid = data_f[-(nvalid+ ntest):-ntest, :, :]
        ksi_valid = data_ksi[-(nvalid+ ntest):-ntest, :, :]
        eta_valid = data_eta[-(nvalid+ ntest):-ntest, :, :]

        x_test = x_train
        u_test = data_u[-ntest:, :, :]
        f_test = data_f[-ntest:, :, :]
        ksi_test = data_ksi[-ntest:, :, :]
        eta_test = data_eta[-ntest:, :, :]
        
        data_train = []
        for j in range(ntrain):
            data_train.append(Data(x=x_train , u=u_train[j, :, :], f=f_train[j, :, :], ksi=ksi_train[j, :, :], eta=eta_train[j, :, :]))
        data_valid = []
        for j in range(nvalid):
            data_valid.append(Data(x=x_valid, u=u_valid[j, :, :], f=f_valid[j, :, :], ksi=ksi_valid[j, :, :], eta=eta_valid[j, :, :]))
        data_test = []
        for j in range(ntest):
            data_test.append(Data(x=x_test, u=u_test[j, :, :], f=f_test[j, :, :], ksi=ksi_test[j, :, :], eta=eta_test[j, :, :]))

        
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(data_valid, batch_size=batch_size2, shuffle=False)
        test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)

        
        t2 = default_timer()
        print(f'>> Preprocessing completed, time elapsed: {(t2 - t1): .2f}s')
        
        def train_single_layer(model, train_loader, valid_loader, test_loader, model_filename, optimizer, epochs, layer_index):
            myloss = LpLoss(size_average=False)
            best_epoch = np.zeros(1)
            bl_train, bl_valid, bl_test = [], [], []
            ttrain, ttrain_disp, tvalid, tvalid_disp, ttest = [], [], [], [], []
            best_train_loss = best_valid_loss = best_test_loss = 1e8
            early_stop = 0
            model.train()
            for ep in range(epochs):
                t1 = default_timer()
                
                optimizer = scheduler(optimizer, learning_rate*lambda_fn(ep))
                
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

                    # mse = F.mse_loss(out.view(-1, 1), f_gt.view(-1, 1))
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
                    train_loss += loss.item()
                    # train_mse += mse.item()
                    train_l2 += loss.item()

                ttrain.append([ep, train_loss / ntrain /s])
                t2 = default_timer()

                model.eval()
                valid_l2 = 0.0
                test_l2 = 0.0
                if train_loss / ntrain /s < best_train_loss:
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
                            valid_l2 += torch.sum((out.view(batch_size2, -1)-f_gt.view(batch_size2, -1))**2).item()

                        for batch in test_loader:
                            batch = batch.to(device)
                            out = model(batch)
                            #out = f_normalizer.decode(out)
                            out = out.reshape(batch_size_test, -1)
                            # crop 2delta layer from computed force
                            out = out.view(batch_size, s, 1)
                            out = out.view(-1, 1)
                            out = out.reshape(batch_size, -1)
                            f_gt = batch.f.view(batch_size, -1)
                            f_gt = f_gt.view(batch_size, s, 1)
                            f_gt = f_gt.view(-1, 1)

                            # test_l2 += myloss(out.view(batch_size_test, -1), f_gt.view(batch_size_test, -1)).item()
                            test_l2 += torch.sum((out.view(batch_size_test, -1)-f_gt.view(batch_size_test, -1))**2).item()


                    tvalid.append([ep, valid_l2 / nvalid /s])
                    ttest.append([ep, test_l2 / ntest/s])

                    if valid_l2 / nvalid/s < best_valid_loss:
                        early_stop = 0
                        best_train_loss = train_l2 / ntrain/s
                        best_valid_loss = valid_l2 / nvalid/s
                        best_test_loss = test_l2 / ntest/s
                        best_epoch = ep
                        torch.save(model.state_dict(), model_filename)
                        
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
                    with open('%s/loss_train.txt' % (base_dir), 'w') as file:
                        np.savetxt(file, ttrain)
                    with open('%s/loss_valid.txt' % (base_dir), 'w') as file:
                        np.savetxt(file, tvalid)
                    with open('%s/loss_test.txt' % (base_dir), 'w') as file:
                        np.savetxt(file, ttest)

                if early_stop > early_stop_num: break

            bl_train.append(best_train_loss)
            bl_valid.append(best_valid_loss)
            bl_test.append(best_test_loss)
            

            print("-" * 100)
            print("-" * 100)
            print(f'>> {layer_index}, ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
            print(f'>> Best train error*1e8: {best_train_loss*1e8: .4f}')
            print(f'>> Best valid error*1e8: {best_valid_loss*1e8: .4f}')
            print(f'>> Best test error*1e8: {best_test_loss*1e8: .4f}')
            print(f'>> Best epoch: {best_epoch}')
            print("-" * 100)
            print("-" * 100)

            f = open("training_record.txt", "a")
            f.write(f'{layer_index}, {lrs}, {lr}, {layer_info}, {dx}: ')
            f.write(','.join(str(err) for err in bl_train))
            f.write(',')
            f.write(','.join(str(err) for err in bl_valid))
            f.write(',')
            f.write(','.join(str(err) for err in bl_test))
            f.write(',')
            f.write(f'{best_epoch}\n')
            f.close()
        
        ##################################################################################################
        #                                        training
        ##################################################################################################
        if not args.test:

            for learning_rate in lrs:
                for scheduler_gamma in gammas:
                    for weight_decay in wds:
                        print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}')
                        
                        current_layer = start_layer
                        model_filename = '%s/model_%s.ckpt' % (base_dir, current_layer)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                            
                        train_single_layer(model, train_loader, valid_loader, test_loader, model_filename, optimizer, epochs, current_layer)
                    
                        # for layer_index in range(start_layer+1, model.phi_MGN.num_modules + 1):
                        # for _ in range(layer_add_times):
                        #     # print(f"Training with {layer_index} layers...")
                            
                        #     prev_layer = current_layer
                        #     current_layer  = current_layer*2

                        #     model.phi_MGN.add_layer(current_layer-prev_layer)
                        #     model_filename = '%s/model_%s.ckpt' % (base_dir, current_layer)
                            
                        #     learning_rate = learning_rate*0.1
                        #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                            
                        #     train_single_layer(model, train_loader, valid_loader, test_loader, model_filename, optimizer, epochs, current_layer)


            print('>> Training Completed!!')
            
        else: 
        ################################################################
        # testing
        ################################################################
            model_path = '%s/model_%s.ckpt' % (base_dir, num_layers)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            mask_bc = torch.zeros((S,1))
            mask_bc[m_fact:s+m_fact] = 1
            # mask_bc = mask_bc.to(device)
            
            compute_train = 0       
            plot_train = 0            
            if compute_train == 1:
                j = 0
                err_u_train = 0
                with torch.no_grad():
                    for batch in train_loader:
                        batch = batch.to(device)
                        for i_data in range(batch_size):
                            batch_i = batch[i_data]
                            tr_u_err, tr_b_err, u = solver_u_tr(batch_i, model, mask_bc, tol_cg, k_cg_max, tol_ls, i_ls_max, device)
                            err_u_train += tr_u_err.item()
                            
                            b = model(batch_i).cpu().detach()
                            b_true = batch_i.f.view(s, 1).cpu().detach()
                            err_b_train += torch.norm(b-b_true)/torch.norm(b_true)
                                
                            if plot_train == 1 and j<=5:                
                                plot_u(batch_i.x.detach().cpu(), batch_i.u.detach().cpu(), u.detach().cpu() , j, base_dir, 'train')
                                plot_b(batch_i.x.detach().cpu(), batch_i.u.detach().cpu(), u.detach().cpu() , j, base_dir, 'train')
                                
                            j += 1
            
                print('Error of u:', err_u_train/ntrain)
                print('Error of b:', err_b_train/ntrain)
            
            compute_test = 1
            plot_test = 1
            if compute_test == 1:
                batch_size_test = 1
                test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)
                j = 0
                err_b = np.zeros((ntest,))
                err_u = np.zeros((ntest,))
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    out = out.view(batch_size_test, s, 1)
                    out = out.cpu().detach().numpy()
                    f_gt = batch.f.view(batch_size_test, s, 1)
                    f_gt = f_gt.cpu().detach().numpy()
                    err_b[j] = np.linalg.norm(out-f_gt)/np.linalg.norm(f_gt)
                    
                    # u_true = batch.u.cpu().detach()
                    u_true = data_u_test[j,:,:]
                    data_u0 = torch.zeros_like(u_true)[m_fact:s+m_fact]

                    def func_linear(u):
                        u = torch.tensor(u, dtype=torch.float64)
                        full_u = (1-mask_bc)*u_true
                        full_u[m_fact:s+m_fact, 0] = u
                        # full_u[m_fact] = u_true[m_fact]
                        # full_u[s+m_fact] = u_true[s+m_fact]
                        # data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                        data = Data(x=data_X , u=full_u, f=data_f_test[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                        y = model.linear(data)-data.f.squeeze()
                        return y.cpu().detach().numpy()
                    
                    def func(u):
                        u = torch.tensor(u, dtype=torch.float64)
                        full_u = (1-mask_bc)*u_true
                        full_u[m_fact:s+m_fact, 0] = u
                        # full_u[m_fact] = u_true[m_fact]
                        # full_u[s+m_fact] = u_true[s+m_fact]
                        # data = Data(x=data_X , u=full_u, f=data_f[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                        data = Data(x=data_X , u=full_u, f=data_f_test[j,m_fact:s+m_fact,:], ksi=data_ksi[0, :, :]).to(device)
                        y = model.nonlinear(data)-data.f.squeeze()
                        return y.cpu().detach().numpy()
                    
                    initial_type = 'linear_sol'
                    if initial_type == 'linear_sol':
                        # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                        data_u0, info_linear, ier, msg = scipy.optimize.fsolve(func_linear, data_u0, full_output=True)
                        u_linear = u_true.clone()
                        u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
                    
                    u, info, ier, msg = scipy.optimize.fsolve(func, data_u0, xtol=1e-12, full_output=True)
                    
                    u = torch.tensor(u, dtype=torch.float64)
                    uh = u_true.clone()
                    uh[m_fact:s+m_fact,0] = u
                    
                    err_u[j] = (torch.norm(uh-u_true)/torch.norm(u_true)).item()
                    
                    if plot_test == 1 and j<=2:                
                        plot_u(data_X, u_true.squeeze(), uh, j, base_dir, '%s_test' % ex)
                        plot_b(data_x, f_gt.squeeze(), out.squeeze(), j, base_dir, '%s_test' % ex)
                    
                    j += 1
                    
                    print(err_b[j],err_u[j])
                    print('----')
            


if __name__ == "__main__":
    
    main()

