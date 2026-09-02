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
    ex = 'ex22'
    # ex = 'ex15'
    # DATA_NAME = 'BK_%s_ndata_500_Nx_129_delta_0.25_h_0.0078125' % (ex)
    DATA_NAME = 'BK_%s_ndata_500_Nx_257_delta_0.25_h_0.00390625' % (ex)
    DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)
    
    dx_all = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
    for index in range(4,5):
        dx = dx_all[index]    # change

        ndata = 500
        ntrain = 300
        nvalid = 50
        ntest = 50
        
        # model and training parameters
        batch_size = 10
        batch_size2 = batch_size
        # layer_info = '128_5_64_5'
        # phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
        # phi_2_layer = [1, 64, 64, 64, 64, 64, 1]
        # layer_info = '128_5_256_4'
        # phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
        # phi_2_layer = [1, 256, 256, 256, 256, 1]
        layer_info = '128_5_128_5'
        phi_1_layer = [1, 128, 128, 128, 128, 128, 1]
        phi_2_layer = [1, 128, 128, 128, 128, 128, 1]
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

        # model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, alpha).to(device)
        model = E_GCL_GKN(phi_1_layer, phi_2_layer, act_fun_xi).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        
        lrs = [0.01]
        # lrs = [0.006]
        gammas = [0.6]
        wds = [0.0]
        # betas = [1.0, 0.0001]
        betas = [1.0]
        
        epochs = 4000
        scheduler_step = 100
        
        # lr = [0.992, 0.998]
        lr = [0.995, 0.998]
        # lambda_fn = lambda epoch: 0.995 ** epoch if epoch < epochs * 0.3 else 0.995 **(epochs * 0.3)*0.998 ** (epoch - epochs * 0.3)
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
        data_x = reader.read_field('coords')[:,::gap].reshape(S,1)
        data_x = data_x[m_fact:s+m_fact]
        data_u = reader.read_field('displacement')[:,::gap].reshape(-1, S)
        data_f = reader.read_field('bodyforce')[:,::gap].reshape(-1, S)
        
        # base_dir = 'Results/%s_gap_%s_ntrain_%s_%s' % (layer_info, gap, ntrain, act_xi)
        base_dir = 'Results/%s_%s_ntrain_%s_bs_%s_%s_gap_%s' % (ex, layer_info, ntrain, batch_size, act_xi, gap)
        base_dir = os.path.join(current_dir, base_dir)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # dx = data_x[1]-data_x[0]
        # delta = m_fact_train * dx
        ksi_range = torch.range(-m_fact, m_fact).int()
        n_ksi = 2*m_fact+1
        # ksi_range = ksi_range[ksi_range != 0]
        # n_ksi = 2*m_fact
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
        test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

        
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
                        model_filename = '%s/model.ckpt' % base_dir
                        ttrain, ttrain_disp, tvalid, tvalid_disp, ttest = [], [], [], [], []
                        best_train_loss = best_valid_loss = best_test_loss = 1e8
                        early_stop = 0
                        for ep in range(epochs):
                            model.train()
                            # optimizer = scheduler(optimizer,
                            #                     LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
                            optimizer = scheduler(optimizer, learning_rate*lambda_fn(ep))
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
                                loss = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                loss.backward()

                                # l2 = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # l2 = torch.norm(out.view(-1) + f_gt.view(-1), 2)/torch.norm(f_gt.view(-1), 2)
                                #l2.backward()

                                optimizer.step()
                                train_loss += loss.item()
                                train_mse += mse.item()
                                train_l2 += loss.item()

                            ttrain.append([ep, train_l2 / ntrain])
                            t2 = default_timer()

                            model.eval()
                            valid_l2 = 0.0
                            test_l2 = 0.0
                            if train_l2 / ntrain < best_train_loss:
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

                                        valid_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()

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

                                        test_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()


                                tvalid.append([ep, valid_l2 / nvalid])
                                ttest.append([ep, test_l2 / ntest])

                                if valid_l2 / nvalid < best_valid_loss:
                                    early_stop = 0
                                    best_train_loss = train_l2 / ntrain
                                    best_valid_loss = valid_l2 / nvalid
                                    best_test_loss = test_l2 / ntest
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
                                        f'runtime: {(t2 - t1):.2f}s, train err *1e3: {(train_l2 / ntrain)*1e3:.4f}, '
                                        f'valid err*1e3: {(valid_l2 / nvalid)*1e3:.4f} , test err*1e3: {(test_l2 / ntest)*1e3:.4f}')
                                else:
                                    early_stop += 1
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1): .2f}s, train err*1e3: {(train_l2 / ntrain)*1e3: .4f} '
                                        f'(best*1e3: {best_train_loss*1e3: .4f}/{best_valid_loss*1e3: .4f})')
                            else:
                                early_stop += 1
                                print(
                                    f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: '
                                    f'{(t2 - t1): .2f}s, train err*1e3: {(train_l2 / ntrain)*1e3: .4f} '
                                    f'(best*1e3: {best_train_loss*1e3: .4f}/{best_valid_loss*1e3: .4f})')
                                with open('%s/loss_train.txt' % (base_dir), 'w') as file:
                                    np.savetxt(file, ttrain)
                                with open('%s/loss_valid.txt' % (base_dir), 'w') as file:
                                    np.savetxt(file, tvalid)
                                with open('%s/loss_test.txt' % (base_dir), 'w') as file:
                                    np.savetxt(file, ttest)

                            if early_stop > 100: break

                        bl_train.append(best_train_loss)
                        bl_valid.append(best_valid_loss)
                        bl_test.append(best_test_loss)
                        

                        print("-" * 100)
                        print("-" * 100)
                        print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
                        print(f'>> Best train error*1e3: {best_train_loss*1e3: .4f}')
                        print(f'>> Best valid error*1e3: {best_valid_loss*1e3: .4f}')
                        print(f'>> Best test error*1e3: {best_test_loss*1e3: .4f}')
                        print(f'>> Best epoch: {best_epoch}')
                        print("-" * 100)
                        print("-" * 100)

                        f = open("training_record.txt", "a")
                        f.write(f'{ntrain}, {learning_rate}, {layer_info}, {dx}: ')
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
            j = 0
            err = 0
            err_true_w = 0
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                out = out.view(batch_size2, s, 1)
                out = out.cpu().detach().numpy()
                f_gt = batch.f.view(batch_size2, s, 1)
                f_gt = f_gt.cpu().detach().numpy()
                err += np.linalg.norm(out-f_gt)/np.linalg.norm(f_gt)
                
                # compute the ingetral of the true w
                out_true_w = Integral_w(batch)
                out_true_w = out_true_w.view(batch_size2, s, 1).cpu().detach().numpy()
                err_true_w += np.linalg.norm(out_true_w-f_gt)/np.linalg.norm(f_gt)
                j += 1
            print(err/j)
            print(err_true_w/j)
            
            n = 1
            fontsize = 15
            plt.plot(data_x, f_gt[n,:], 'k',linewidth=2, label='true')
            plt.plot(data_x, out[n,:], color='forestgreen', linewidth=2,linestyle='--', label='MGN')
            plt.legend()
            plt.xlabel(r'$x$', fontsize=fontsize)
            plt.ylabel(r'body force $b$', fontsize=fontsize)
            plt.savefig('%s/%s_b%s_MGN.png' % (base_dir, ex, n), format='png')
            plt.close()
            
            #******************************************** plot ***************************
            # lambda_min_data = 0.8
            # lambda_max_data = 1.2
            delta = 0.25
            mu = 0.3846
            c = 2*mu/math.pi/delta**2
            # g_fun = lambda x: np.ones_like(x)
            # g_fun = lambda x: x
            # g_fun = lambda x: x**2
            g_fun = lambda x: x*np.exp(-50*(x)**2)
            # alpha = 1
            
            # plot 
            xi_norm = dx*torch.ones((100,))
            xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
            lambda_min = 0.6
            lambda_max = 1.4
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
            err = np.linalg.norm(dw1_exact-dw1_normlized.flatten())/np.linalg.norm(dw1_exact)
            print("relative error for normalized k1: %s" % err )
            err1 = np.linalg.norm(dw1_exact-dw1)/np.linalg.norm(dw1_exact)
            print("relative error for k1: %s" % err1 )
            
            
            fig, ax = plt.subplots(figsize=(8, 6))
            fontsize = 15
            ax.plot(lambdaa, dw1_exact, 'k',linewidth=2, label=r'true: $\lambda-\lambda^{-3}$')
            ax.plot(lambdaa, dw1, color='mediumorchid', linewidth=2, linestyle='--', label=r'$k_1^{NN}(\lambda)$')
            ax.plot(lambdaa, dw1_normlized, color='darkorange', linewidth=2, linestyle='--', label=r'normalized $k_1^{NN}(\lambda)$')
            y_lim = ax.get_ylim()
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.plot([lambda_min_data, lambda_min_data], [y_lim[0], y_lim[1]], color='gray', linestyle='--', linewidth=2)
            ax.axvline(x=lambda_max_data, color='gray', linestyle='--', linewidth=2)
            # place text between the two lines
            midpoint = (lambda_min_data + lambda_max_data) / 2
            y_pos = (1-0.3)*y_lim[0]
            ax.text(midpoint, y_pos, 'training data coverage', fontsize=15, ha='center', va='center', color='gray')
            ax.annotate('', xy=(lambda_min_data, y_pos), xytext=(lambda_min_data+0.25, y_pos), arrowprops=dict(arrowstyle='->', color='gray', linestyle='--', linewidth=1.5))
            ax.annotate('', xy=(lambda_max_data-0.25, y_pos), xytext=(lambda_max_data, y_pos), arrowprops=dict(arrowstyle='<-', color='gray', linestyle='--', linewidth=1.5))
            
            y_annotation = (1+0.1)*y_lim[0]
            ax.text(lambda_min_data, y_annotation, f'$\lambda$={lambda_min_data}', ha='center', va='top', fontsize=fontsize, color='black')
            ax.text(lambda_max_data, y_annotation, f'$\lambda$={lambda_max_data}', ha='center', va='top', fontsize=fontsize, color='black')

            ax.legend(fontsize=fontsize, loc='lower right')
            # ax.set_xlabel(r'$\lambda$')
            # ax.set_ylabel(r'$k_1^{NN}(\lambda)$')
            ax.xaxis.label.set_size(fontsize)
            ax.yaxis.label.set_size(fontsize)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.savefig('%s/%s_k1_MGN_%s_%s_h_%s.png' % (base_dir, ex, lambda_min, lambda_max, dx), format='png')
            plt.close(fig)
            
            
            # plot dw about xi
            lambdaa = 1.1* torch.ones((100,))
            lambdaa_cuda = 1.1* torch.ones((100,)).unsqueeze(1).to('cuda')
            cons =  torch.ones_like(lambdaa_cuda)
            xi_norm = torch.linspace(0.05,0.25, 100)
            xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
            # phi_input = torch.cat([xi_plus_eta_norm.unsqueeze(1).to('cuda'), xi_norm_cuda], dim=1)
            k1_NN = (model.phi_MGN(lambdaa_cuda)- model.phi_MGN(cons)).cpu().detach().numpy()
            k2_NN = (model.phi_2(xi_norm_cuda)).cpu().detach().numpy()
            k_NN =  k1_NN*k2_NN
            k1_true = (lambdaa-lambdaa**(-3))
            k2_true = 2*c/xi_norm*g_fun(xi_norm)
            k_true = k1_true*k2_true
            
            
            fontsize = 15
            # plt.plot(xi_norm, k2_NN_normalized, color='darkorange', linestyle='--', linewidth=2)
            plt.plot(xi_norm, k2_NN, color='darkorange', linestyle='--', linewidth=2, label='MGN')
            plt.plot(xi_norm, k2_true, color='k', linewidth=2, label='true')
            plt.legend()
            plt.xlabel(r'$|\xi|$', fontsize=fontsize)
            plt.ylabel(r'$k_2(\xi)$', fontsize=fontsize)
            plt.savefig('%s/%s_k2_%s.png' % (base_dir, ex, act_xi), format='png')
            plt.close()
            
            
            #********************** plot 2d  ****************************
            N=100
            xi_norm = torch.linspace(dx, delta, N)
            # eta_norm = torch.linspace(-0.02,0.1, N)
            # xi_plus_eta_norm = xi_norm+eta_norm
            # lambdaa = xi_plus_eta_norm/xi_norm
            # lambdaa = torch.linspace(lambda_min_data,lambda_max_data, N)
            lambdaa = torch.linspace(lambda_min,lambda_max, N)
            # lambdaa = torch.linspace(0.5,1.7, N)
            [Xi_norm, Lambdaa] = torch.meshgrid(xi_norm, lambdaa)
            Xi_norm_cuda = Xi_norm.reshape(-1,1).to('cuda')   
            Lambdaa_cuda = Lambdaa.reshape(-1,1).to('cuda')
            Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
            # phi_input = torch.cat([xi_plus_eta_norm.unsqueeze(1).to('cuda'), xi_norm_cuda], dim=1)
            # phi_input = lambdaa.unsqueeze(1).to('cuda')
            k_NN = ((model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)) *model.phi_2(Xi_norm_cuda)).reshape(N,N)
            # dw = (model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)).reshape(N,N)
            k_NN = k_NN.cpu().detach().numpy()
            # phi_input = torch.cat([Lambdaa.reshape(-1,1),Xi_norm.reshape(-1,1)], dim=1)
            k_true = ((Lambdaa-Lambdaa**(-3))*2*c/Xi_norm*g_fun(Xi_norm)).numpy()
            err = np.linalg.norm(k_true-k_NN)/np.linalg.norm(k_true)
            print('Relative L2 error of k: %s' % err)
            
            plt.rcParams.update({'font.size': 15}) 
            fig, ax = plt.subplots(figsize = (20,6))
            plt.subplot(1,3,1)
            plt.pcolor(Xi_norm, Lambdaa, k_NN, cmap='Spectral')
            plt.xlabel(r'$|\xi|$')
            plt.ylabel(r'$\lambda$')
            plt.title(r'Learned $k_1^{NN}(\lambda)k_2^{NN}(\xi)$')
            plt.colorbar()
            plt.subplot(1,3,2)
            plt.pcolor(Xi_norm, Lambdaa, k_true, cmap='Spectral')
            plt.title(r'True $k(\lambda,\xi)$')
            plt.xlabel(r'$|\xi|$')
            plt.ylabel(r'$\lambda$')
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.pcolor(Xi_norm, Lambdaa, np.abs(k_NN-k_true), cmap='Spectral')
            plt.title(r'Relative $L^2$ error: %.6f' % err)
            plt.xlabel(r'$|\xi|$')
            plt.ylabel(r'$\lambda$')
            plt.colorbar()
            plt.savefig('%s/%s_k_2d.png' % (base_dir, ex), format='png')
            


if __name__ == "__main__":
    
    main()

