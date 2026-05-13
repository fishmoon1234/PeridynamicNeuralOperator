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

# def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
#     # return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))
#     return 0.995 ** epoch if epoch < epochs * 0.5 else 0.99 ** (epoch - epochs * 0.5)


def main():
    parser = argparse.ArgumentParser('Train or Test Arg! ')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--layer_info', type=str, default='32_4')
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
        layer_info = args.layer_info
        phi_1_layer = parse_layer_info(layer_info)
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

        model = E_GCL_GKN(phi_1_layer, torch.nn.Sigmoid()).to(device)
        
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
        
        reader = MatReader(DATA)
        data_x = reader.read_field('coords')[:,::gap].reshape(S,1)
        data_x = data_x[m_fact:s+m_fact]
        data_u = reader.read_field('displacement')[:,::gap].reshape(-1, S)
        data_f = reader.read_field('bodyforce')[:,::gap].reshape(-1, S)
        
        # batch_size = 50
        batch_size = 10
        batch_size_valid = 50
        batch_size_test = 1
        
        base_dir = '%s/Results/%s_%s_ntrain_%s_lrs_%s_lr_%s_gap_%s' % (current_dir, ex, layer_info, ntrain, lrs, lr, gap)
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
        valid_loader = DataLoader(data_valid, batch_size=batch_size_valid, shuffle=False)
        test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)

        
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
                                loss_batch = torch.norm(out.view(batch_size, -1) - f_gt.view(batch_size, -1), 2)/torch.norm(f_gt.view(batch_size, -1), 2)
                                loss = torch.mean(loss_batch)
                                # loss = (myloss(out.view(batch_size, -1), -f_gt.view(batch_size, -1)) +
                                #         model.egkn_conv.phi_mlp(torch.tensor(1.0).unsqueeze(0).to('cuda')) ** 2.0 +
                                #         1.0/model.egkn_conv.phi_mlp(torch.tensor(0.0).unsqueeze(0).to('cuda')) ** 2.0)
                                # loss = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # loss = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # loss = torch.sum((out.view(batch_size, -1)-f_gt.view(batch_size, -1))**2)
                                loss.backward()

                                # l2 = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # train_err = torch.norm(out.view(-1) + f_gt.view(-1), 2)/torch.norm(f_gt.view(-1), 2)
                                #l2.backward()

                                optimizer.step()
                                # scheduler.step()
                                train_loss += torch.sum(loss_batch).item()

                            train_loss = train_loss / ntrain/s
                            ttrain.append([ep, train_loss])
                            t2 = default_timer()

                            model.eval()
                            valid_loss = 0.0
                            test_loss = 0.0
                            if train_loss < best_train_loss:
                                with torch.no_grad():
                                    for batch in valid_loader:
                                        batch = batch.to(device)
                                        out = model(batch)
                                        #out = f_normalizer.decode(out)
                                        out = out.reshape(batch_size_valid, -1)
                                        # crop 2delta layer from computed force
                                        out = out.view(batch_size_valid, s, 1)
                                        out = out.view(-1, 1)
                                        #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                                        f_gt = batch.f.view(batch_size_valid, -1)
                                        f_gt = f_gt.view(batch_size_valid, s, 1)
                                        f_gt = f_gt.view(-1, 1)

                                        # valid_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()
                                        # valid_l2 = torch.sum(torch.norm(out.view(batch_size, -1)-f_gt.view(batch_size, -1), 2, 1))
                                        valid_loss_batch = torch.norm(out.view(batch_size_valid, -1) - f_gt.view(batch_size_valid, -1), 2)/torch.norm(f_gt.view(batch_size_valid, -1), 2)
                                        
                                        valid_loss += (torch.sum(valid_loss_batch)).item()
                                        
                                    valid_loss = valid_loss / nvalid/s

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

                                        # test_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()
                                        # test_l2 = torch.sum(torch.norm(out.view(batch_size, -1)-f_gt.view(batch_size, -1), 2, 1))
                                        test_loss_batch = torch.norm(out.view(batch_size_test, -1) - f_gt.view(batch_size_test, -1), 2)/torch.norm(f_gt.view(batch_size_test, -1), 2)
                                        test_loss += (torch.sum(test_loss_batch)).item()
                                    
                                    test_loss = test_loss / ntest/s  

                                tvalid.append([ep, valid_loss])
                                ttest.append([ep, test_loss])

                                if valid_loss < best_valid_loss:
                                    early_stop = 0
                                    best_train_loss = train_loss
                                    best_valid_loss = valid_loss
                                    best_test_loss = test_loss
                                    best_epoch = ep
                                    torch.save(model.state_dict(), model_filename)
                                
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1):.2f}s, train loss: {(train_loss):.4e}, '
                                        f'valid loss: {(valid_loss):.4e} , test loss: {(test_loss):.4e}')
                                else:
                                    early_stop += 1
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1): .2f}s, train loss: {(train_loss):.4e} '
                                        f'(best: {best_train_loss:.4e}/{best_valid_loss:.4e})')
                            else:
                                early_stop += 1
                                print(
                                    f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: '
                                    f'{(t2 - t1): .2f}s, train loss: {(train_loss):.4e} '
                                    f'(best: {best_train_loss:.4e}/{best_valid_loss:.4e})')

                            if early_stop > 50: break

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
                        print(f'>> Best train loss: {best_train_loss:.4e}')
                        print(f'>> Best valid loss: {best_valid_loss:.4e}')
                        print(f'>> Best test loss: {best_test_loss:.4e}')
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
            
            mask_bc = torch.zeros((S,1))
            mask_bc[m_fact:s+m_fact] = 1
            mask_bc = mask_bc.to(device)
            tol_cg, k_cg_max = 1e-4, 40
            tol_ls, i_ls_max = 1e-4, 40
                    
            # plot_train = 1
            # j = 0
            # err_u_train = 0
            # with torch.no_grad():
            #     for batch in train_loader:
            #         batch = batch.to(device)
            #         for i_data in range(batch_size):
            #             batch_i = batch[i_data]
            #             tr_u_err, tr_b_err, u = solver_u_tr(batch_i, model, mask_bc, tol_cg, k_cg_max, tol_ls, i_ls_max, device)
            #             err_u_train += tr_u_err.item()
                        
            #             b = model(batch_i).cpu().detach()
            #             b_true = batch_i.f.view(s, 1).cpu().detach()
            #             err_b_train += torch.norm(b-b_true)/torch.norm(b_true)
                            
            #             if plot_train == 1 and j<=5:                
            #                 plot_u(batch_i.x.detach().cpu(), batch_i.u.detach().cpu(), u.detach().cpu() , j, base_dir, 'train')
            #                 plot_b(batch_i.x.detach().cpu(), batch_i.u.detach().cpu(), u.detach().cpu() , j, base_dir, 'train')
                            
            #             j += 1
            
            # print('Error of u:', err_u_train/ntrain)
            # print('Error of b:', err_b_train/ntrain)
            
            j = 0
            err = np.zeros((ntest,))
            if_plot = 1
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                out = out.view(batch_size_test, s, 1)
                out = out.reshape(batch_size_test, -1)
                out = out.cpu().detach().numpy()
                f_gt = batch.f.view(batch_size_test, s, 1)
                f_gt = f_gt.cpu().detach().numpy()
                f_gt = f_gt.reshape(batch_size_test, -1)
                err[j] = np.linalg.norm(out-f_gt)/np.linalg.norm(f_gt)
                print(f'{j}, Error of b: {err[j]:.2e}')
                
                if if_plot == 1:
                    fontsize = 18
                    plt.rcParams.update({'font.size': fontsize}) 
                    fig, ax = plt.subplots(figsize = (8,6))
                    plt.plot(data_x, f_gt.reshape(-1,1), 'k',linewidth=2, label='true')
                    plt.plot(data_x, out.reshape(-1,1), color='forestgreen', linewidth=2,linestyle='--', label='ICNN')
                    plt.legend()
                    plt.xlabel(r'$x$', fontsize=fontsize)
                    plt.ylabel(r'$b(x)$', fontsize=fontsize)
                    plt.savefig('%s/%s_b%s_MGN.png' % (base_dir, ex, j), format='png', bbox_inches='tight',dpi=300)
                    plt.close()
                
                j += 1
            print(np.mean(err))
            


if __name__ == "__main__":
    
    main()

