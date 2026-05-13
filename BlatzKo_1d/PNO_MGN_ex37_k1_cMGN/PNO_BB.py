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
from scipy.optimize import fsolve

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


def main():
    parser = argparse.ArgumentParser('Train or Test Arg! ')
    parser.add_argument('--test', action='store_true', default=True)
    # parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')
        
    t1 = default_timer()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = '%s/../1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/' % (current_dir)
    ex = 'ex37'
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

        layer_info = '128_4'
        # phi_1_layer = [1,  128, 128, 128, 128, 128, 128, 128, 1]
        embed_dim = 128
        num_layers = 4
        # start_layer = 8
        # layer_add_times = 2
        model = E_GCL_GKN(num_layers, 1, embed_dim).to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad )
        print(f"Total number of parameters: {total_params}")
        
        lrs = [0.005]
        epochs = 3000
        # scheduler_step = 200
        scheduler_step = 100
        early_stop_num = 50
        
        lr = [0.995, 0.998]
        lambda_fn = lambda epoch: lr[0] ** epoch if epoch < epochs * 0.3 else lr[0] **(epochs * 0.3)*lr[1] ** (epoch - epochs * 0.3)
        
        gammas = [0.6]
        wds = [0.0]
        betas = [1.0]

        Nx = 257 
        h0 = 2**(-8)
        
        delta = 0.25
        # dx_cuda = dx.to(device)
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
        data_U = data_u.clone()
        data_U_test = data_U[-ntest:,:].reshape(ntest,S,1)
        
        # batch_size = 50
        batch_size = 10
        batch_size2 = batch_size
        batch_size_test = 1
        
        # base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
        base_dir = 'Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_sigmoid_gap_%s' % (ex, layer_info, ntrain, lrs, lr, gap)
        base_dir = os.path.join(current_dir, base_dir)
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
            
        # A = data_eta.reshape(-1, n_ksi)
        # print(np.linalg.cond(np.dot(A.T, A)))
            
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
                            out = out.view(batch_size_test, s, 1)
                            out = out.view(-1, 1)
                            out = out.reshape(batch_size_test, -1)
                            f_gt = batch.f.view(batch_size_test, -1)
                            f_gt = f_gt.view(batch_size_test, s, 1)
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
            print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
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
                    
                        current_layer = num_layers
                        model_filename = '%s/model_%s.ckpt' % (base_dir, current_layer)
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                            
                        train_single_layer(model, train_loader, valid_loader, test_loader, model_filename, optimizer, epochs, current_layer)
                    
                        # # for layer_index in range(start_layer+1, model.phi_MGN.num_modules + 1):
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
            model.load_state_dict(torch.load(model_path))
            model.eval()
            
            mask_bc = torch.zeros((S,1))
            mask_bc[m_fact:s+m_fact] = 1
            # mask_bc = mask_bc.to(device)
            
            
            # plot_index = [0, 1, 2, 3, 17, 22, 36, 38, 41, 47, 51, 92] 
            # plot_index = [3, 17, 21, 36, 38, 41, 51, 62, 75]
            plot = 0
            # initial_type = 'linear_sol'
            initial_type = 'zero'
            # if compute == 1:
            def test(data, ntest):
                err_u = torch.zeros((ndata,1))
                err_b = torch.zeros((ndata,1))
                
                plot_index = range(ndata)
                
                with torch.no_grad():
                    for j in range(ntest):
                        data_j = data[j]
                        # data_j = data_j.to(device)
                        out = model(data_j.to(device))
                        out = out.view(s, 1).cpu().detach()
                        f_gt = data_j.f.view(s, 1).cpu().detach()
                        err_b[j] += torch.norm(out-f_gt)/torch.norm(f_gt)
                        
                        
                    # j = 0
                    # for batch in test_loader:
                    #     batch = batch.to(device)
                    #     b = model(batch).view(s, 1).cpu().detach()
                    #     b_true = batch.f.view(-1, 1).cpu().detach()
                    #     b_true = b_true[m_fact:s+m_fact,:]
                    #     err_b[j] += torch.norm(b-b_true)/torch.norm(b_true)
                    #     j += 1
                    # print('Error of b:', err_b/ndata)
                    
                    # for j in range(ndata):
                    # for j in plot_index:
                        data_u = data_U_test[j,:,:]
                        data_f = data_j.f
                        data_ksi = data_j.ksi
                        
                        u_true = data_u
                        
                        data_u0 = torch.zeros_like(u_true)[m_fact:s+m_fact]
                        
                        # initial_type = 'interp'
                        # index_bc = torch.cat((torch.arange(0,m_fact),torch.arange(s+m_fact, s+2*m_fact)))
                        # x_bc = data_X[index_bc].squeeze()
                        # u_bc = data_u_j[index_bc].squeeze()
                        # interp_fun = interp1d(x_bc.numpy(), u_bc.numpy(), kind='linear')
                        # data_u0 = torch.tensor(interp_fun(data_x.numpy()))
                        
                        def func_linear(u):
                            u = torch.tensor(u, dtype=torch.float64)
                            full_u = (1-mask_bc)*u_true
                            full_u[m_fact:s+m_fact, 0] = u
                            # full_u[m_fact] = u_true[m_fact]
                            # full_u[s+m_fact] = u_true[s+m_fact]
                            data = Data(x=data_X , u=full_u, f=data_f, ksi=data_ksi).to(device)
                            y = model.linear(data)-data.f.squeeze()
                            return y.cpu().detach().numpy()
                        
                        def func(u):
                            u = torch.tensor(u, dtype=torch.float64)
                            full_u = (1-mask_bc)*u_true
                            full_u[m_fact:s+m_fact, 0] = u
                            # full_u[m_fact] = u_true[m_fact]
                            # full_u[s+m_fact] = u_true[s+m_fact]
                            data = Data(x=data_X , u=full_u, f=data_f, ksi=data_ksi).to(device)
                            y = model.nonlinear(data)-data.f.squeeze()
                            return y.cpu().detach().numpy()
                        
                        if initial_type == 'linear_sol':
                            # data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, maxfev=100, full_output=True)
                            data_u0, info_linear, ier, msg = fsolve(func_linear, data_u0, full_output=True)
                            u_linear = u_true.clone()
                            u_linear[m_fact:s+m_fact,0] = torch.tensor(data_u0, dtype=torch.float64)
                        
                        u, info, ier, msg = fsolve(func, data_u0, xtol=1e-9, full_output=True)
                        
                        u = torch.tensor(u, dtype=torch.float64)
                        uh = u_true.clone()
                        uh[m_fact:s+m_fact,0] = u
                            
                        err_u[j] = (torch.norm(uh-u_true)/torch.norm(u_true)).item()
                        
                        print(f"{j}, err b: {err_b[j]}, err u: {err_u[j]}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}")
                        
                        with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex, initial_type), "a") as file:
                            file.write(f"{j}, err b: {err_b[j]}, err u: {err_u[j]}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}\n")

                            
                        # if plot == 1 and (err_u[j]>0.05):
                        if plot == 1 and j<5:
                            # plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex_data, initial_type))
                            
                            # if initial_type == 'linear_sol':
                            #     plot_u_3(data_X, u_true, uh, u_linear, j, base_dir, '%s_%s' % (ex, initial_type))
                            # else: 
                            #     plot_u(data_X, u_true, uh, j, base_dir, '%s_%s' % (ex, initial_type))
                            
                            plot_u_b(data_x, f_gt, out, data_X, u_true, uh, j, base_dir, '%s_%s_%s_u_b' % (ex, initial_type, j))
                            
                            
                    print('Error of u:', f'{torch.mean(err_u):.4e}')
                    print('Error of b:', f'{torch.mean(err_b):.4e}')
                
                with open("%s/%s_err_u_record_%s.txt" % (base_dir, ex, initial_type), "a") as file:
                    file.write(f"mean err u: {torch.mean(err_u):.4e}, mean err b: {torch.mean(err_b):.4e}\n")
                    
                # with open("%s/err_u_record.txt" % (current_dir), "a") as file:
                #     file.write(f"{ex}, {initial_type}, {dx}, {torch.mean(err_u)}, {torch.mean(err_b)}\n")
            
            if_test_u_b = 0
            if if_test_u_b == 1:
                test(data_test, ntest) 
            
            
            plot_model = 1
            if plot_model == 1:
                
                delta = 0.25
                mu = 0.3846
                c = 2*mu/math.pi/delta**2
                k_fun = lambda x: 2*c*np.cos(np.pi*np.abs(x))
                g_fun = lambda x: x-x**(-3)
                
                Lambdaa = torch.linspace(lambda_min_data, lambda_max_data, 100)
                Lambdaa_cuda = Lambdaa.unsqueeze(1).to('cuda')
                Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
                g_NN = (model.phi_MGN(Lambdaa_cuda)- model.phi_MGN(Lambdaa_1_cuda)).reshape(-1,1)
                g_NN = g_NN.cpu().detach()
                g_true = g_fun(Lambdaa)
                
                g_err = torch.norm(g_NN.reshape(-1,1)-g_true.reshape(-1,1))/torch.norm(g_true.reshape(-1,1))
                print(f'Error of g: {g_err}')
                
                fontsize = 22
                plt.rcParams.update({
                    'font.size': fontsize,
                    'font.family': 'serif',  # Use serif font for better readability in publications
                    'axes.linewidth': 1.2,   # Border line width
                    'axes.edgecolor': 'gray', # Border color
                    'axes.labelsize': fontsize,
                    'axes.titlesize': fontsize+2,  # Title slightly larger than labels
                    'xtick.labelsize': fontsize,
                    'ytick.labelsize': fontsize,
                    'grid.linewidth': 0.8,   # Grid line thickness
                    'grid.linestyle': '-',   # Solid grid lines
                    'grid.alpha': 0.5,       # Grid transparency (50%)
                    'grid.color': 'gray',    # Grid color
                    'legend.fontsize': fontsize-2,  # Slightly smaller legend text
                    'legend.frameon': True,  # Show legend border
                    'legend.framealpha': 0.95,  # 95% opaque legend background
                    'legend.fancybox': True,  # Rounded legend corners
                    'legend.shadow': True    # Add shadow effect to legend
                })
                
                # Create figure with specified size (width, height) in inches
                fig, ax = plt.subplots(figsize=(8, 6))
                # plt.plot(xi_norm, k2_NN_normalized, color='darkorange', linestyle='--', linewidth=2)
                plt.plot(Lambdaa, g_true, color='k', linewidth=2.5, label='True $g$')
                plt.plot(Lambdaa, g_NN, color='mediumslateblue', linestyle='--', linewidth=2.5, label='Learned $g$')
                plt.legend()
                plt.xlabel(r'$\lambda$', fontsize=fontsize)
                plt.ylabel(r'$g(\lambda)$', fontsize=fontsize)
                plt.grid(True)
                plt.savefig('%s/%s_g.png' % (base_dir, ex), format='png', bbox_inches='tight',dpi=300)
                plt.close()
                

if __name__ == "__main__":
    
    main()

