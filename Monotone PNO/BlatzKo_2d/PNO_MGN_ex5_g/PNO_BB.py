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

seed = 1
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
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--layer_info', type=str, default='128_4')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.0)
    # parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    print("PID:", os.getpid())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')
        
    t1 = default_timer()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    DATA_PATH = '%s/../generate_data/analytical_u/DATA/' % current_dir
    ex = 'ex5'
    DATA_NAME = 'BK_2d_%s_ndata_100_Nx_33' % (ex)
    DATA = '%s%s.mat' % (DATA_PATH, DATA_NAME)

    ndata = 100
    ntrain = 80
    nvalid = 10
    ntest = 10
    
    # model and training parameters
    batch_size = 1
    batch_size2 = batch_size
    # layer_info = '64_5_64_3'
    # phi_1_layer = [1, 64, 64, 64, 64, 64, 1]
    # phi_2_layer = [1, 64, 64, 64, 1]
    phi_2_layer = parse_layer_info(args.layer_info)
    
    # model = EGKN(phi_layer).to(device)
    model = E_GCL_GKN(phi_2_layer, nn.ReLU).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    lrs = [args.lr]
    gammas = [args.gamma]
    wds = [args.wd]
    # betas = [1.0, 0.0001]
    betas = [1.0]
    
    epochs = 500
    scheduler_step = 50

    Nx = 33
    Nx_all = 49
    h0 = 2**(-5)
    delta = 0.25
    
    
    # dx = 2**(-4)    # change
    # dx_all = np.array([2**(-4), 2**(-5), 2**(-6), 2**(-7), 2**(-8)])
    dx_all = np.array([2**(-5)])
    for index in range(1):
        dx = dx_all[index]    # change
    
        gap = int(dx/h0)
        m_fact = int(delta/dx)
        s = int((Nx-1)/gap)+1
        S = s+2*m_fact
        n = S**2
        print(f'>> Training Mesh resolution: {S}x{S}')
        
        base_dir = 'Results/%s_gap_%s_ntrain_%s_gamma_%s_wd_%s_lr_%s' % (args.layer_info, gap, ntrain, args.gamma, args.wd, args.lr)
        base_dir = os.path.join(current_dir, base_dir)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        reader = MatReader(DATA)
        
        data_x = reader.read_field('coords').reshape(Nx_all, Nx_all, 2)
        data_u = reader.read_field('displacement')
        data_f = reader.read_field('bodyforce')
        data_x = data_x[::gap,::gap,:].reshape(n, 2)
        data_u = data_u[:,::gap,::gap,:].reshape(-1, n, 2)
        data_f = data_f[:,::gap,::gap,:].reshape(-1, n, 2)
        
        # data_x = reader.read_field('coords').reshape(n, 2)
        # data_u = reader.read_field('displacement').reshape(-1, n, 2)
        # data_f = reader.read_field('bodyforce').reshape(-1, n, 2)
        
        x_train = data_x
        u_train = data_u[:ntrain, :, :]
        f_train = data_f[:ntrain, :, :]

        x_valid = x_train
        u_valid = data_u[-(nvalid+ ntest):-ntest, :, :]
        f_valid = data_f[-(nvalid+ ntest):-ntest, :, :]

        x_test = x_train
        u_test = data_u[-ntest:, :, :]
        f_test = data_f[-ntest:, :, :]

        cond_f = torch.abs(x_train.view(S,S,2)[0,:,1] - 0.5) <= (0.5+1e-10)  # select the data belongs to [0,1]

        # import mesh and dataset
        edge_index_train = {}
        edge_attr_train = {}

        meshgenerator_train = IrregularMeshGenerator(x_train, [S, S])
        edge_index_train = meshgenerator_train.ball_connectivity(float(delta))
        edge_attr_train = meshgenerator_train.attributes(theta=0)

        data_train = []
        for j in range(ntrain):
            data_train.append(Data(x=x_train, u=u_train[j, :, :], f=f_train[j, :, :], edge_index=edge_index_train, edge_attr=edge_attr_train, delta=delta, dx=dx, S=S))

        edge_index_valid = {}
        edge_attr_valid = {}

        meshgenerator_valid = IrregularMeshGenerator(x_valid, [S, S])
        edge_index_valid = meshgenerator_valid.ball_connectivity(float(delta))
        edge_attr_valid = meshgenerator_valid.attributes(theta=0)

        data_valid = []
        for j in range(nvalid):
            data_valid.append(Data(x=x_valid, u=u_valid[j, :, :], f=f_valid[j, :, :], edge_index=edge_index_valid, edge_attr=edge_attr_valid, delta=delta, dx=dx, S=S))


        edge_index_test = {}
        edge_attr_test = {}

        meshgenerator_test = IrregularMeshGenerator(x_test, [S, S])
        edge_index_test = meshgenerator_test.ball_connectivity(float(delta))
        edge_attr_test = meshgenerator_test.attributes(theta=0)
        
        # alpha = 0.5  # order of kernel
        # k1k2w = quadweights(delta=delta, hx=dx, hy=dx, order=alpha+1)
        # k1k2w = torch.from_numpy(k1k2w).to('cuda')

        data_test = []
        for j in range(ntest):
            data_test.append(Data(x=x_test, u=u_test[j, :, :], f=f_test[j, :, :], edge_index=edge_index_test, edge_attr=edge_attr_test, delta=delta, dx=dx, S=S))

        if_compute_lambda = 0
        if if_compute_lambda == 1:
            lambda_max = torch.zeros((ntrain,))  
            lambda_min = torch.zeros((ntrain,))    
            for i in range(ntrain):
                col, row = edge_index_train
                ksi = x_train[col] - x_train[row]
                eta = u_train[i,col] - u_train[i,row]
                ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
                ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
                extension = ksi_plus_eta_norm - ksi_norm
                lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
                # print(torch.mean(lambdaa))
                lambda_max[i] = torch.max(lambdaa) 
                lambda_min[i] = torch.min(lambdaa) 
                
            # lambdaa = compute_lambda(ntrain, x_train, u_train, edge_index_train)
            lambda_max_data = round(torch.max(lambda_max).item(), 2)
            lambda_min_data = round(torch.min(lambda_min).item(), 2)
            print(lambda_min_data, lambda_max_data)    
        print(f'>> grid: {x_train.shape}, edge_index: {edge_index_train.shape}, edge_attr: {edge_attr_train.shape}')
        
        
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
                            optimizer = scheduler(optimizer,
                                                LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
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
                                # crop 2delta layer from computed force
                                out = out.view(batch_size, S, S, 2)
                                out = out[:, cond_f, :, :]
                                out = out[:, :, cond_f, :]
                                out = out.view(-1, 2)
                                #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                                f_gt = batch.f.view(batch_size, -1)
                                f_gt = f_gt.view(batch_size, S, S, 2)
                                f_gt = f_gt[:, cond_f, :, :]
                                f_gt = f_gt[:, :, cond_f, :]
                                f_gt = f_gt.view(-1, 2)

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

                                l2 = myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1))
                                # l2 = torch.norm(out.view(-1) + f_gt.view(-1), 2)/torch.norm(f_gt.view(-1), 2)
                                #l2.backward()

                                optimizer.step()
                                train_loss += loss.item()
                                train_mse += mse.item()
                                train_l2 += l2.item()

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
                                        out = out.view(batch_size, S, S, 2)
                                        out = out[:, cond_f, :, :]
                                        out = out[:, :, cond_f, :]
                                        out = out.view(-1, 2)
                                        #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                                        f_gt = batch.f.view(batch_size, -1)
                                        f_gt = f_gt.view(batch_size, S, S, 2)
                                        f_gt = f_gt[:, cond_f, :, :]
                                        f_gt = f_gt[:, :, cond_f, :]
                                        f_gt = f_gt.view(-1, 2)

                                        valid_l2 += myloss(out.view(batch_size2, -1), f_gt.view(batch_size2, -1)).item()

                                    for batch in test_loader:
                                        batch = batch.to(device)
                                        out = model(batch)
                                        #out = f_normalizer.decode(out)
                                        out = out.reshape(batch_size, -1)
                                        # crop 2delta layer from computed force
                                        out = out.view(batch_size, S, S, 2)
                                        out = out[:, cond_f, :, :]
                                        out = out[:, :, cond_f, :]
                                        out = out.view(-1, 2)
                                        #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                                        f_gt = batch.f.view(batch_size, -1)
                                        f_gt = f_gt.view(batch_size, S, S, 2)
                                        f_gt = f_gt[:, cond_f, :, :]
                                        f_gt = f_gt[:, :, cond_f, :]
                                        f_gt = f_gt.view(-1, 2)

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
                                    
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1):.2f}s, train err: {(train_l2 / ntrain):.4e}, '
                                        f'valid err: {(valid_l2 / nvalid):.4e} , test err: {(test_l2 / ntest):.4e}')
                                else:
                                    early_stop += 1
                                    print(
                                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                                        f'runtime: {(t2 - t1): .2f}s, train err: {(train_l2 / ntrain):.4e} '
                                        f'(best: {best_train_loss:.4e}/{best_valid_loss:.4e})')
                            else:
                                early_stop += 1
                                print(
                                    f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: '
                                    f'{(t2 - t1): .2f}s, train err: {(train_l2 / ntrain):.4e} '
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
                        print(f'>> Best train error: {best_train_loss:.4e}')
                        print(f'>> Best valid error: {best_valid_loss:.4e}')
                        print(f'>> Best test error: {best_test_loss:.4e}')
                        print(f'>> Best epoch: {best_epoch}')
                        print("-" * 100)
                        print("-" * 100)

                        f = open("training_record.txt", "a")
                        f.write(f'{ntrain}, {learning_rate}, {scheduler_gamma}, {weight_decay}, ')
                        f.write(','.join(str(err) for err in bl_train))
                        f.write(',')
                        f.write(','.join(str(err) for err in bl_valid))
                        f.write(',')
                        f.write(','.join(str(err) for err in bl_test))
                        f.write(',')
                        f.write(f', {best_epoch}\n')
                        f.close()

            print('>> Training Completed!!')
            
        else: 
        ################################################################
        # testing
        ################################################################
            model_path = os.path.join(base_dir, 'model.ckpt')
            model.load_state_dict(torch.load(model_path))
            
            model.eval()
            if_compute_error = 1
            if if_compute_error == 1:
                j = 0
                err = 0
                out_plot = None
                f_gt_plot = None
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    
                    out = out.reshape(batch_size, -1)
                    # crop 2delta layer from computed force
                    out = out.view(batch_size, S, S, 2)
                    out = out[:, cond_f, :, :]
                    out = out[:, :, cond_f, :]
                    out = out.cpu().detach().numpy()
                    #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                    f_gt = batch.f.view(batch_size, S, S, 2)
                    f_gt = f_gt[:, cond_f, :, :]
                    f_gt = f_gt[:, :, cond_f, :]
                    f_gt = f_gt.cpu().detach().numpy()
                    
                    # Store the first batch for plotting
                    if j == 0:
                        out_plot = out
                        f_gt_plot = f_gt
                        plot_b(out_plot.reshape(s, s, 2), f_gt_plot.reshape(s, s, 2), base_dir, ex, j)
                    
                    err += np.linalg.norm(out-f_gt)/np.linalg.norm(f_gt)
                    j += 1
                print(f'Error of b: {err/j:.4e}')
            
            
            #******************************************** plot ***************************
            if_plot_g = 1
            if if_plot_g == 1:
                delta = 0.25
                mu = 0.3846
                c = 2*mu/math.pi/delta**2
                # g_fun = lambda x: np.ones_like(x)ß
                # g_fun = lambda x: x
                # g_fun = lambda x: x**2
                # # alpha = 1
                g_fun = lambda x: x-x**(-3)
                k_fun = lambda x: 2*c*torch.exp(-50*x**2)*(delta-torch.abs(x))
                
                # xi_norm = torch.linspace(0.05,0.25, 100)
                # xi_norm_cuda = xi_norm.unsqueeze(1).to('cuda')
                # k_NN = model.phi_2(xi_norm_cuda).detach().cpu().reshape(-1,1)
                # k_true = k_fun(xi_norm).reshape(-1,1)
                # err_k = torch.norm(k_true-k_NN)/torch.norm(k_true)
                # print("relative error for normalized k: %s" % err_k )
                
                lambda_min_data, lambda_max_data = 0.95, 1.34  # h=2**(-5)
                # lambda_min, lambda_max = 0.8, 1.5
                # lambda_min, lambda_max = lambda_min_data, lambda_max_data
                lambdaa = torch.linspace(lambda_min_data, lambda_max_data, 100)
                lambdaa_cuda = lambdaa.unsqueeze(1).to('cuda')
                lambdaa_1_cuda = torch.ones_like(lambdaa_cuda)
                g_NN = (model.phi_MGN(lambdaa_cuda)- model.phi_MGN(lambdaa_1_cuda)).detach().cpu()
                g_true = g_fun(lambdaa)
                
                err_g = torch.norm(g_true-g_NN.reshape(-1,))/torch.norm(g_true)
                print("relative error for normalized g:" f'{err_g:.4e}' )
                
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
                plt.plot(lambdaa, g_true, color='k', linewidth=2.5, label='True $g$')
                plt.plot(lambdaa, g_NN, color='deepskyblue', linestyle='--', linewidth=2.5, label='Learned $g$')
                plt.legend()
                plt.xlabel(r'$\lambda$', fontsize=fontsize)
                plt.ylabel(r'$g(\lambda)$', fontsize=fontsize)
                plt.grid(True)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                plt.savefig('%s/%s_g.png' % (base_dir, ex), format='png', bbox_inches='tight',dpi=300)
                plt.close()
            
            
            #********************** plot 2d  ****************************
            # N=100
            # xi_norm = torch.linspace(dx,delta, N)
            # # eta_norm = torch.linspace(-0.02,0.1, N)
            # # xi_plus_eta_norm = xi_norm+eta_norm
            # # lambdaa = xi_plus_eta_norm/xi_norm
            # # lambdaa = torch.linspace(lambda_min_data,lambda_max_data, N)
            # lambdaa = torch.linspace(lambda_min,lambda_max, N)
            # # lambdaa = torch.linspace(0.5,1.7, N)
            # [Xi_norm, Lambdaa] = torch.meshgrid(xi_norm, lambdaa)
            # Xi_norm_cuda = Xi_norm.reshape(-1,1).to('cuda')   
            # Lambdaa_cuda = Lambdaa.reshape(-1,1).to('cuda')
            # Lambdaa_1_cuda = torch.ones_like(Lambdaa_cuda)
            # # phi_input = torch.cat([xi_plus_eta_norm.unsqueeze(1).to('cuda'), xi_norm_cuda], dim=1)
            # # phi_input = lambdaa.unsqueeze(1).to('cuda')
            # k_NN = ((model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)) *model.phi_2(Xi_norm_cuda)).reshape(N,N)
            # # dw = (model.phi_MGN(Lambdaa_cuda)-model.phi_MGN(Lambdaa_1_cuda)).reshape(N,N)
            # k_NN = k_NN.cpu().detach().numpy()
            # # phi_input = torch.cat([Lambdaa.reshape(-1,1),Xi_norm.reshape(-1,1)], dim=1)
            # k_true = g_fun(Lambdaa)*k_fun(Xi_norm)
            # err = np.linalg.norm(k_true-k_NN)/np.linalg.norm(k_true)
            # print('Relative L2 error of k: %s' % err)
            
            # # Set up the plotting style for 2D heatmap
            # plt.style.use('default')
            # plt.rcParams.update({
            #     'font.size': 12,
            #     'font.family': 'serif',
            #     'axes.linewidth': 1.2,
            #     'axes.spines.top': False,
            #     'axes.spines.right': False,
            #     'xtick.direction': 'in',
            #     'ytick.direction': 'in',
            #     'xtick.major.size': 5,
            #     'ytick.major.size': 5,
            #     'xtick.minor.size': 3,
            #     'ytick.minor.size': 3
            # })
            
            # fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            # fig.suptitle('Kernel Function $k(\\lambda, |\\xi|)$: 2D Visualization', fontsize=18, fontweight='bold', y=0.95)
            
            # # Define consistent color scale for all three plots
            # vmin_all = min(k_NN.min(), k_true.min())
            # vmax_all = max(k_NN.max(), k_true.max())
            # error_max = np.max(np.abs(k_true - k_NN))
            
            # # Plot 1: Neural Network prediction
            # im1 = axes[0].pcolormesh(Xi_norm, Lambdaa, k_NN, cmap='viridis', vmin=vmin_all, vmax=vmax_all, shading='auto')
            # axes[0].set_xlabel(r'$|\xi|$ (Bond Length)', fontsize=14, weight='bold')
            # axes[0].set_ylabel(r'$\lambda$ (Stretch Ratio)', fontsize=14, weight='bold')
            # axes[0].set_title(r'Neural Network: $k_1^{NN}(\lambda) \cdot k_2^{NN}(|\xi|)$', fontsize=16, weight='bold', pad=15)
            # axes[0].set_aspect('equal')
            # cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.8, aspect=20)
            # cbar1.set_label('Kernel Value', fontsize=12, weight='bold')
            # cbar1.ax.tick_params(labelsize=11)
            
            # # Plot 2: Ground truth
            # im2 = axes[1].pcolormesh(Xi_norm, Lambdaa, k_true, cmap='viridis', vmin=vmin_all, vmax=vmax_all, shading='auto')
            # axes[1].set_xlabel(r'$|\xi|$ (Bond Length)', fontsize=14, weight='bold')
            # axes[1].set_ylabel(r'$\lambda$ (Stretch Ratio)', fontsize=14, weight='bold')
            # axes[1].set_title(r'Ground Truth: $k(\lambda, |\xi|)$', fontsize=16, weight='bold', pad=15)
            # axes[1].set_aspect('equal')
            # cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.8, aspect=20)
            # cbar2.set_label('Kernel Value', fontsize=12, weight='bold')
            # cbar2.ax.tick_params(labelsize=11)
            
            # # Plot 3: Error
            # error_data = k_true - k_NN
            # im3 = axes[2].pcolormesh(Xi_norm, Lambdaa, error_data, cmap='RdBu_r', vmin=-error_max, vmax=error_max, shading='auto')
            # axes[2].set_xlabel(r'$|\xi|$ (Bond Length)', fontsize=14, weight='bold')
            # axes[2].set_ylabel(r'$\lambda$ (Stretch Ratio)', fontsize=14, weight='bold')
            # axes[2].set_title(f'Absolute Error (L² Error: {err:.2e})', fontsize=16, weight='bold', pad=15)
            # axes[2].set_aspect('equal')
            # cbar3 = fig.colorbar(im3, ax=axes[2], shrink=0.8, aspect=20)
            # cbar3.set_label('Error', fontsize=12, weight='bold')
            # cbar3.ax.tick_params(labelsize=11)
            
            # # Add training data coverage region to all plots
            # for ax in axes:
            #     ax.axhspan(lambda_min_data, lambda_max_data, alpha=0.2, color='red', label='Training Data Coverage')
            #     ax.axhline(y=lambda_min_data, color='red', linestyle='--', linewidth=2, alpha=0.7)
            #     ax.axhline(y=lambda_max_data, color='red', linestyle='--', linewidth=2, alpha=0.7)
            
            # # Add grid to all plots
            # for ax in axes:
            #     ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            #     ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5)
            #     ax.tick_params(axis='both', which='minor', labelsize=10, width=0.8, length=3)
            
            # # Add legend for training data coverage
            # axes[0].legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True, 
            #               framealpha=0.9, edgecolor='black')
            
            # plt.tight_layout()
            # plt.subplots_adjust(top=0.88, bottom=0.15)
            # plt.savefig('%s/2D_%s_k_2d.png' % (base_dir, ex), format='png', dpi=300, bbox_inches='tight')
            # plt.close()
            
            
            print('>> Testing Completed!!')


if __name__ == "__main__":
    
    main()
