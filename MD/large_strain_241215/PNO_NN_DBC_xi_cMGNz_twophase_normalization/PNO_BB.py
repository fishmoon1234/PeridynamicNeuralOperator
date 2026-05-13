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
import matplotlib.colors as mcolors


seed1 = 1
random.seed(seed1)
torch.manual_seed(seed1)
np.random.seed(seed1)
seed2 = 42
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed2)


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def main():
    parser = argparse.ArgumentParser('Train or Test Arg! ')
    parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    parser.add_argument('--layer_info', type=str, default='64_4_256_4')
    parser.add_argument('--solution_type', type=str, help='Type of solution', default='one_phase')
    parser.add_argument('--test_dataset', type=str, help='Test dataset', default='test')
    parser.add_argument('--beta', type=float, default=1.0)
    args = parser.parse_args()
    
    print(f'PID: {os.getpid()}')
    
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
    # # layer_info = '16_16_512_4'
    # # phi_2_layer = [2, 512, 512, 512, 512, 1]
    # phi_1_layer = [1, 128, 128, 128, 128, 1]
    # phi_2_layer = [2, 256, 256, 256, 256, 1]
    # model = E_GCL_GKN(num_modules, embed_dim, phi_2_layer, nn.ReLU(), nn.Sigmoid()).to(device)
    
    # MGN_params = sum(p.numel() for p in model.phi_MGN.parameters() if p.requires_grad)
    # print(f"Number of parameters for MGN: {MGN_params}")
    
    layer_info = args.layer_info
    phi_1_layer, phi_2_layer = parse_layer_info(layer_info)
    model = E_GCL_GKN(phi_1_layer, phi_2_layer, nn.ReLU(), nn.Sigmoid()).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    
    lrs = [1e-3]
    # lrs = [5e-4]
    # gammas = [0.9, 0.7, 0.5]
    # wds = [0.0, 1e-5, 1e-4]
    
    # lrs = [1e-5]
    gammas = [0.5]
    wds = [0.0]

    epochs = 1000
    scheduler_step = 100
    batch_size = 2
    batch_size_valid = batch_size
    batch_size_test = 1
    
    current_dir = os.path.dirname(os.path.realpath(__file__))

    DATA_NAME = 'cg-md-250212'
    # DATA_NAME = 'cg-md-250212_41'
    # DATA_NAME = 'cg-md-250212_10_60_0.4'
    # DATA_NAME = 'cg-md-250212_10_60_0.4_0.01'
    # DATA_NAME = 'cg-md-250212_0.2'
    DATA = os.path.join(current_dir, "../ProcessData/%s.mat" % DATA_NAME)
    reader = MatReader(DATA)
    data_x = reader.read_field('coords')
    data_mass = reader.read_field('mass')
    data_u = reader.read_field('disps')
    data_f = reader.read_field('forces')
    data_f = -data_f
    
    data_f = data_f/(data_mass.unsqueeze(2)/83.75)
    
    # INDEX_NAME = 'index_b'
    INDEX_NAME = 'cg-md-250212_index_b1_0.02'
    DATA_index = os.path.join(current_dir, "../ProcessData/%s.mat" % INDEX_NAME)
    reader = MatReader(DATA_index)
    index = reader.read_field('index').flatten().int()
    data_u = data_u[index]
    data_f = data_f[index]
    
    INDEX_NAME = 'MD_index_274'
    DATA_index = os.path.join(current_dir, "../ProcessData/%s.mat" % INDEX_NAME)
    reader = MatReader(DATA_index)
    train_index = reader.read_field('train_index').flatten().int()
    valid_index = reader.read_field('valid_index').flatten().int()
    test_index = reader.read_field('test_index').flatten().int()
    
    Nx = 441
    m_fact = 3.01
    S = 21
    s = 15
    n = s ** 2
    cond_f = torch.abs(data_x.view(S,S,2)[0,:,1]) <= (35 + 1e-10)
    
    ###  for solve u ###
    mask_bc = torch.abs(data_x.view(S,S,2)) <= (35 + 1e-10)
    mask_bc = mask_bc[:,:,0] & mask_bc[:,:,1]
    mask_bc = torch.stack((mask_bc, mask_bc), dim= 2)
    mask_bc_logic = mask_bc
    mask_bc = mask_bc.int()
    
    
    # Nx = 1681
    # S = 41
    # s = 29
    # m_fact = 6.01
    # n = s ** 2
    # # cond_f = torch.abs(data_x.view(S,S,2)[0,:,1]) <= (42.5 + 1e-10)
    # cond_f = torch.abs(data_x.view(S,S,2)[0,:,1]) <= (35 + 1e-10)
    
    # normalization
    norm = '1'
    data_x = data_x*0.1
    data_u = data_u*0.1
    data_f = data_f*0.1
    # mass_mean = torch.mean(data_mass)
    # data_f = data_f*100/ 83.75*mass_mean / data_mass.unsqueeze(0).unsqueeze(2) 
    # data_f = data_f/ 83.75 
    
    N_data = data_u.size(0)
    ntrain = 200
    nvalid = 50
    ntest = 20
    
    # # train_index = np.arange(0, 3*ntrain, 3)
    # # start = 0
    # # train_index = list(range(start, start+ntrain))
    # # train_index = np.arange(0, ntrain)
    # train_index = random.sample(range(N_data), ntrain)
    # remaining_numbers = list(set(range(N_data)) - set(train_index))
    # valid_index = random.sample(remaining_numbers, nvalid)
    # remaining_numbers = list(set(range(N_data)) - set(train_index)- set(valid_index))
    # test_index = random.sample(remaining_numbers, ntest)
    # print(np.mean(train_index))
    # print(np.mean(valid_index))
    # print(np.mean(test_index))
    
    u_train = data_u[train_index, :,:]
    u_valid = data_u[valid_index, :,:]
    u_test = data_u[test_index, :,:]
    
    f_train = data_f[train_index, :,:]
    f_valid = data_f[valid_index, :,:]
    f_test = data_f[test_index, :,:]
    
    mass_train = data_mass
    mass_train_mean = torch.mean(data_mass)
    mass_valid = data_mass
    mass_valid_mean = mass_train_mean
    mass_test = data_mass
    mass_test_mean = mass_train_mean
    

    trains = range(ntrain)
    valids = range(nvalid)
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

    data_train = []
    for j in range(ntrain):
        data_train.append(Data(x=data_x, u=u_train[j, :, :], f=f_train[j, :, :], edge_index=edge_index, edge_attr=edge_attr, delta=delta, dx=dx))
    
    
    if_compute_lambda = 0  
    if if_compute_lambda == 1:
        
        lambda_max = torch.zeros((ntrain,))  
        lambda_min = torch.zeros((ntrain,))    
        strain_max = torch.zeros((ntrain,)) 
        strain_mean = torch.zeros((ntrain,))    
        for i in range(ntrain):
            col, row = edge_index
            ksi = data_x[col] - data_x[row]
            eta = u_train[i,col] - u_train[i,row]
            ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
            ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
            extension = ksi_plus_eta_norm - ksi_norm
            lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
            # print(torch.mean(lambdaa))
            lambda_max[i] = torch.max(lambdaa) 
            lambda_min[i] = torch.min(lambdaa)
            strain_max[i] = torch.max(torch.abs(lambdaa-1))
            strain_mean[i] = torch.mean(torch.abs(lambdaa-1))
            
        plt.rcParams.update({'font.size': 15}) 
        fig, ax = plt.subplots(figsize = (6,5))
        fontsize = 15
        ax.plot(range(ntrain), lambda_max, color='darkorange', linewidth=1)
        ax.plot(range(ntrain), lambda_min, color='green', linewidth=1)
        # plt.scatter(range(ntrain), lambda_max, color='darkorange')
        # plt.scatter(range(ntrain), lambda_min, color='green')
        plt.title('min and max of strain')
        plt.tight_layout()
        plt.savefig('%s/%s_lambda.png' % (current_dir, DATA_NAME), format='png')
        plt.close(fig)
        
        # lambdaa = compute_lambda(ntrain, x_train, u_train, edge_index_train)
        lambda_max_data = round(torch.max(lambda_max).item(), 4)
        lambda_min_data = round(torch.min(lambda_min).item(), 4)
        print(lambda_min_data, lambda_max_data) 


    data_valid = []
    for j in valids:
        data_valid.append(Data(x=data_x, u=u_valid[j, :, :], f=f_valid[j, :, :], edge_index=edge_index, edge_attr=edge_attr, delta=delta, dx=dx))


    data_test = []
    for j in tests:
        data_test.append(Data(x=data_x, u=u_test[j, :, :], f=f_test[j, :, :], edge_index=edge_index, edge_attr=edge_attr, delta=delta, dx=dx))

    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data_valid, batch_size=batch_size_valid, shuffle=False)
    test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)

    
    t2 = default_timer()
    print(f'>> Preprocessing completed, time elapsed: {(t2 - t1): .2f}s')
    myloss = LpLoss(size_average=False)
    
    def train_single_layer(model, train_loader, valid_loader, test_loader, model_filename, optimizer, epochs):
        best_epoch = np.zeros(1)
        bl_train_loss, bl_train, bl_valid, bl_test = [], [], [], []
        
        ttrain_loss, ttrain_err, tvalid_err, ttest_err = [], [], [], []
        best_train_loss = best_train_err = best_valid_err = best_test_err = 1e8
        early_stop = 0
        for ep in range(epochs):
            model.train()
            optimizer = scheduler(optimizer,
                                LR_schedule(learning_rate, ep, scheduler_step, scheduler_gamma))
            t1 = default_timer()
            train_err = 0.0
            train_loss = 0.0
            for batch in train_loader:
                batch = batch.to(device)

                optimizer.zero_grad()
                out, L_match = model(batch)
                #out = f_normalizer.decode(out.reshape(batch_size, -1))
                out = out.view(batch_size, -1)
                #crop extended parts from computed force
                out = out.view(batch_size, S, S, 2)
                out = out[:, cond_f, :, :]
                out = out[:, :, cond_f, :]
                # out = out.view(-1,2)
                #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                f_gt = batch.f.view(batch_size, S, S, 2)
                f_gt = f_gt[:, cond_f, :, :]
                f_gt = f_gt[:, :, cond_f, :]

                # mse = F.mse_loss(out.view(-1, 1), f_gt.view(-1, 1))
                #mse.backward()
                train_err_1 = myloss.rel_sqrt(out.view(batch_size, -1), f_gt.view(batch_size, -1))

                train_loss_1 = args.beta*myloss(out.view(batch_size, -1), f_gt.view(batch_size, -1)) + L_match
                # train_loss_1 = torch.norm(out.view(batch_size, -1)- f_gt.view(batch_size, -1))**2
                train_loss_1.backward()

                optimizer.step()
                train_err += train_err_1.item()
                train_loss += train_loss_1.item()
            
            ttrain_err.append([ep, train_err / ntrain])
            ttrain_loss.append([ep, train_loss / ntrain])
            t2 = default_timer()

            model.eval()
            valid_err = 0.0
            test_err = 0.0
            if train_loss / ntrain < best_train_loss:
                with torch.no_grad():
                    for batch in valid_loader:
                        batch = batch.to(device)
                        out, _ = model(batch)
                        #out = f_normalizer.decode(out.reshape(batch_size, -1))
                        # out = out.view(batch_size_valid, -1)
                        # crop extended parts from computed force
                        out = out.view(batch_size_valid, S, S, 2)
                        out = out[:, cond_f, :, :]
                        out = out[:, :, cond_f, :]
                        out = out.view(-1, 2)
                        #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                        # f_gt = batch.f.view(batch_size_valid, -1)
                        f_gt = batch.f.view(batch_size_valid, S, S, 2)
                        f_gt = f_gt[:, cond_f, :, :]
                        f_gt = f_gt[:, :, cond_f, :]

                        # valid_l2 += myloss(out.view(batch_size_valid, -1), f_gt.view(batch_size_valid, -1)).item()
                        valid_err += myloss.rel_sqrt(out.view(batch_size_valid, -1), f_gt.view(batch_size_valid, -1)).item()
                        

                    for batch in test_loader:
                        batch = batch.to(device)
                        out, _ = model(batch)
                        #out = f_normalizer.decode(out.reshape(batch_size, -1))
                        #f_gt = f_normalizer.decode(batch.f.view(batch_size, -1))
                        out = out.view(batch_size_test, S, S, 2)
                        # out[:,:,:,0] = out[:,:,:,0] * mask_f_test
                        # out[:,:,:,1] = out[:,:,:,1] * mask_f_test
                        # out = out.view(batch_size_test, -1)
                        out = out[:, cond_f, :, :]
                        out = out[:, :, cond_f, :]
                        out = out.view(-1, 2)

                        f_gt = batch.f.view(batch_size_test, S, S, 2)
                        f_gt = f_gt[:, cond_f, :, :]
                        f_gt = f_gt[:, :, cond_f, :]

                        test_err += myloss.rel_sqrt(out.view(batch_size_test, -1), f_gt.view(batch_size_test, -1)).item()


                tvalid_err.append([ep, valid_err / nvalid])
                ttest_err.append([ep, test_err / ntest])

                if valid_err / nvalid < best_valid_err:
                    early_stop = 0
                    best_train_loss = train_loss / ntrain
                    best_train_err = train_err / ntrain
                    best_valid_err = valid_err / nvalid
                    best_test_err = test_err / ntest
                    best_epoch = ep
                    torch.save(model.state_dict(), model_filename)
                    
                    print(
                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                        f'runtime: {(t2 - t1):.2f}s, train loss: {(train_loss / ntrain):.4f}, train err: {(train_err / ntrain):.4f}, valid err: {(valid_err / nvalid): .4f},test err: {(test_err / ntest):.4f}, best')
                else:
                    early_stop += 1
                    print(
                        f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                        f'runtime: {(t2 - t1): .2f}s, train loss: {(train_loss / ntrain):.4f}, train err: {(train_err / ntrain): .4f}, valid err: {(valid_err / nvalid): .4f}')
            else:
                early_stop += 1
                print(
                    f'>> epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: '
                    f'{(t2 - t1): .2f}s, train err: {(train_err / ntrain): .4f} '
                    f'(best: {best_train_err: .4f}/{best_valid_err: .4f})')

            if early_stop > 100: break

        bl_train_loss.append(best_train_loss)
        bl_train.append(best_train_err)
        bl_valid.append(best_valid_err)
        bl_test.append(best_test_err)
        with open('%s/train_loss.txt' % (base_dir), 'w') as file:
            np.savetxt(file, ttrain_loss)
        with open('%s/train_err.txt' % (base_dir), 'w') as file:
            np.savetxt(file, ttrain_err)
        with open('%s/valid_err.txt' % (base_dir), 'w') as file:
            np.savetxt(file, tvalid_err)
        with open('%s/test_err.txt' % (base_dir), 'w') as file:
            np.savetxt(file, ttest_err)

        print("-" * 100)
        print("-" * 100)
        print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
        print(f'>> Best train loss: {best_train_loss: .4f}')
        print(f'>> Best train error: {best_train_err: .4f}')
        print(f'>> Best valid error: {best_valid_err: .4f}')
        print(f'>> Best test error: {best_test_err: .4f}')
        print(f'>> Best epoch: {best_epoch}')
        print("-" * 100)
        print("-" * 100)

        f = open("MD_static.txt", "a")
        f.write(f'{seed1}, {seed2}, {DATA_NAME}, {ntrain}, {layer_info}, {learning_rate}, {scheduler_gamma}, {weight_decay}: ')
        f.write(','.join(str(err) for err in bl_train))
        f.write(',')
        f.write(','.join(str(err) for err in bl_valid))
        f.write(',')
        f.write(','.join(str(err) for err in bl_test))
        f.write(',')
        f.write(f'{best_epoch}\n')
        f.close()
    
    ############################################################################################
    #                                        training
    ############################################################################################
    if not args.test:

        for learning_rate in lrs:
            for scheduler_gamma in gammas:
                for weight_decay in wds:
                    print(f'>> ntrain: {ntrain}, lr: {learning_rate}, gamma: {scheduler_gamma}, w_d: {weight_decay}')
                    base_dir = '%s/Results/%s_nvalid%s_%s_lr%s_dr%s_step%s_bs%s_norm%s_z_seed_%s_%s' % (current_dir,INDEX_NAME, nvalid, layer_info, learning_rate , scheduler_gamma, scheduler_step, batch_size, norm, seed1, seed2)

                    if not os.path.exists(base_dir):
                        os.makedirs(base_dir)
                    
                    model_filename = '%s/model.ckpt' % (base_dir)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                        
                    train_single_layer(model, train_loader, valid_loader, test_loader, model_filename, optimizer, epochs)

                    

        print('>> Training Completed!!')
        
    else: 
    ################################################################
    # testing
    ################################################################
        # # base_dir = '%s/Results/cg-md-250212_41_index_b_ntrain200_256_4_512_4_lr0.001_dr0.5_step50_bs2_norm1' % current_dir
        base_dir = '%s/Results/MD_index_274_nvalid50_%s_lr0.001_dr0.5_step100_bs2_norm1_z_seed_1_42' % (current_dir, args.layer_info)
        # # layer_info = '16_7_128_4'
        # # phi_1_layer = [1, 16, 16, 16, 16, 16, 16, 16, 1]
        # # phi_2_layer = [2, 128, 128, 128, 128, 1]
        # # layer_info = '32_10_128_4'
        # # phi_1_layer = [1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1]
        # # # phi_2_layer = [2, 128, 128, 128, 128, 1]
        # # layer_info = '32_10_256_4'
        # # phi_1_layer = [1, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1]
        # # phi_2_layer = [2, 256, 256, 256, 256, 1]
        # # # layer_info = '16_15_256_4'
        # # # phi_1_layer = [1, 16, 16, 16, 16, 16, 16, 16,16, 16, 16, 16, 16, 16, 16, 16, 1]
        # # # phi_2_layer = [2, 256, 256, 256, 256, 1]
        # test_layer = 9  # layer_info -1
        # model = E_GCL_GKN(phi_1_layer, test_layer, phi_2_layer, nn.ReLU(), nn.Sigmoid()).to(device)
        
             
        model_path = os.path.join(base_dir, 'model.ckpt')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        plot_g = 0
        plot_k = 0
        compute_error = 1
        
        def test(data, name, ntest):
            batch_size_test = 1
            test_loader = DataLoader(data, batch_size=batch_size_test, shuffle=False)
            j = 0
            rel_L2_err_test = 0
            err_b_test = np.zeros((ntest,))
            err_u_test = np.zeros((ntest,))
            err_u_test_linear = np.zeros((ntest,))
            for batch in test_loader:
                batch = batch.to(device)
                out, _ = model(batch)
                out = out.view(batch_size_test, -1)
                out = out.view(batch_size_test, S, S, 2)
                out = out[:, cond_f, :, :]
                out = out[:, :, cond_f, :]
                out = out.view(batch_size_test, -1)
                f_gt = batch.f.view(batch_size_test, S, S, 2)
                f_gt = f_gt[:, cond_f, :, :]
                f_gt = f_gt[:, :, cond_f, :]
                f_gt = f_gt.view(batch_size_test, -1)

                # err_test = myloss(out.view(batch_size_test, -1), f_gt.view(batch_size_test, -1)).item()
                # print('err_test:', err_test)
                
                out = out.cpu().detach()
                f_gt = f_gt.cpu().detach()
                # rel_L2_err_test += torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
                err_b_test[j] = torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
                # print(rel_L2_err_test)
                
                
                ###### solve u  ###### 
                def func_linear(u):
                    u = torch.tensor(u, dtype=torch.float32).view(s,s,2)
                    u_new = u_true.clone()
                    u_new[3: s+3, 3: s+3, :] = u
                    
                    u_new = u_new.view(-1, 2)

                    input = Data(x=batch.x, u=u_new, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
                    f = model.linear(input).view(S, S, 2).cpu().detach()
                    # f = model(input).view(S, S, 2).cpu().detach()
                    f_gt = batch.f.squeeze().view(S, S, 2).cpu().detach()
                    y = (f-f_gt)[mask_bc_logic]
                    return y
                               
                def func(u):
                    u = torch.tensor(u, dtype=torch.float32).view(s,s,2)
                    u_new = u_true.clone()
                    u_new[3: s+3, 3: s+3, :] = u
                    
                    u_new = u_new.view(-1, 2)

                    input = Data(x=batch.x, u=u_new, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
                    f, _ = model(input)
                    f = f.view(S, S, 2).cpu().detach()
                    f_gt = batch.f.squeeze().view(S, S, 2).cpu().detach()
                    y = (f-f_gt)[mask_bc_logic]
                    return y
                
                u_true = batch.u.view(S, S, 2).detach().cpu()
                data_u0 = (1 - mask_bc) * u_true
                data_u0 = data_u0[mask_bc_logic]
                u_true_i = u_true[mask_bc_logic]
                if args.solution_type =='one_phase':
                    u, info, ier, msg = fsolve(func, data_u0, xtol=1e-6, full_output=True)
                    u = torch.tensor(u, dtype=torch.float32)
                    err_u_test[j] = torch.norm(u-u_true_i.view(-1,))/torch.norm(u_true_i.view(-1,))
                    print(f"{j}, err b: {err_b_test[j]:.2e}, err u: {err_u_test[j]:.2e}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}\n")
                else:
                    u_linear, info, ier, msg = fsolve(func_linear, data_u0, xtol=1e-6, full_output=True)
                    u = torch.tensor(u_linear, dtype=torch.float32)
                    err_u_test_linear[j] = torch.norm(u-u_true_i.view(-1,))/torch.norm(u_true_i.view(-1,))
                    print('linear:', f"{j}, err b: {err_b_test[j]:.2e}, err u: {err_u_test_linear[j]:.2e}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}")
                    
                    u, info, ier, msg = fsolve(func, u_linear, xtol=1e-6, full_output=True)
                    u = torch.tensor(u, dtype=torch.float32)
                    err_u_test[j] = torch.norm(u-u_true_i.view(-1,))/torch.norm(u_true_i.view(-1,))
                    print('nonlinear:', f"{j}, err b: {err_b_test[j]:.2e}, err u: {err_u_test[j]:.2e}, fsolve flag:{ier}, {np.max(info['fvec'])}, {info['nfev']}")
                    
                
                if_plot = 1
                if j < 10 and if_plot == 1:
                    b_plot = out[0,:].view(s, s, 2)
                    b_true_plot = f_gt[0,:].view(s, s, 2)
                    # plt_b(b_true_plot, b_plot, base_dir, j, b_true_plot-b_plot, 'test')
                    u_plot = u.view(s, s, 2)
                    u_true_plot = u_true_i.view(s, s, 2)
                    plt_b_u( b_plot, b_true_plot, u_plot, u_true_plot, base_dir, j, name)

                
                j +=1
                
            mean_error_b = np.mean(err_b_test)
            mean_error_u = np.mean(err_u_test)
            mean_error_u_linear = np.mean(err_u_test_linear)
            print('Relative L2 error for %s b: %.2e' % (name, mean_error_b) )
            print('Relative L2 error for %s u: %.2e' % (name, mean_error_u) )
            print('Relative L2 error for %s u_linear: %.2e' % (name, mean_error_u_linear) )
            
            # Save error arrays to files
            np.savetxt('%s/err_b_%s.txt' % (base_dir, name), err_b_test)
            np.savetxt('%s/err_u_%s.txt' % (base_dir, name), err_u_test)
            # np.savetxt('%s/err_u_linear_%s.txt' % (base_dir, name), err_u_test_linear)
            print(f'Error arrays saved to {base_dir}/err_b_{name}.txt and {base_dir}/err_u_{name}.txt')
            
            return err_b_test, err_u_test
        
        def plot_errors(err_b, err_u, name, base_dir):
            """
            Plot err_b and err_u in the same figure
            """
            plt.rcParams.update({'font.size': 15})
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot both error arrays
            ax.plot(err_b, 'o-', color='darkorange', linewidth=2, markersize=4, label='err_b_test')
            ax.plot(err_u, 's-', color='forestgreen', linewidth=2, markersize=4, label='err_u_test')
            
            ax.set_xlabel('Test Sample Index')
            ax.set_ylabel('Relative L2 Error')
            ax.set_title(f'Error Comparison for {name} Data')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{base_dir}/error_b_u_{name}.png', format='png', dpi=300)
            plt.close(fig)
            
        
        # err_b_train, err_u_train = test(data_train, 'train', ntrain)
        # err_b_valid, err_u_valid = test(data_valid, 'valid', nvalid)

        if args.test_dataset == 'train':
            err_b_test, err_u_test = test(data_train, 'train', ntrain) 
            plot_errors(err_b_test, err_u_test, 'train', base_dir)
        elif args.test_dataset == 'valid':
            err_b_test, err_u_test = test(data_valid, 'valid', nvalid) 
            plot_errors(err_b_test, err_u_test, 'valid', base_dir)
        elif args.test_dataset == 'test':
            err_b_test, err_u_test = test(data_test, 'test', ntest) 
            plot_errors(err_b_test, err_u_test, 'test', base_dir)
        
        else:
            DATA_NAME = 'cg-md-250212'
            DATA = os.path.join(current_dir, "../ProcessData/%s.mat" % DATA_NAME)
            reader = MatReader(DATA)
            data_x = reader.read_field('coords')
            data_mass = reader.read_field('mass')
            data_u = reader.read_field('disps')
            data_f = reader.read_field('forces')
            data_f = -data_f
            
            data_f = data_f/(data_mass.unsqueeze(2)/83.75)
    
            INDEX_NAME = args.test_dataset
            DATA_index = os.path.join(current_dir, "../ProcessData/%s.mat" % INDEX_NAME)
            reader = MatReader(DATA_index)
            index = reader.read_field('index').flatten().int()
            
            data_u = data_u[index]
            data_f = data_f[index]
            # data_u = data_u[:10]
            # data_f = data_f[:10]
            
            # normalization
            norm = '1'
            data_x = data_x*0.1
            data_u = data_u*0.1
            data_f = data_f*0.1
            # mass_mean = torch.mean(data_mass)
            # data_f = data_f*100/ 83.75*mass_mean / data_mass.unsqueeze(0).unsqueeze(2) 
            # data_f = data_f/ 83.75 
            
            ntest = data_u.size(0)
            tests = range(ntest) 

            edge_index = {}
            edge_attr = {}

            meshgenerator = IrregularMeshGenerator(data_x, [S, S])
            edge_index = meshgenerator.ball_connectivity(float(delta))
            edge_attr = meshgenerator.attributes(theta=0)

            data_test = []
            for j in tests:
                data_test.append(Data(x=data_x, u=data_u[j, :, :], f=data_f[j, :, :], edge_index=edge_index, edge_attr=edge_attr, delta=delta, dx=dx))
            # test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)
        
            err_b_test_large_strain, err_u_test_large_strain = test(data_test, INDEX_NAME, ntest)
            plot_errors(err_b_test_large_strain, err_u_test_large_strain, INDEX_NAME, base_dir)
            
        # myloss = LpLoss(size_average=False)
        # compute_error_train = 1
        # compute_error_valid = 0
        # if compute_error_train == 1:
        #     batch_size = 1
        #     train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
        #     j = 0 
        #     err_train = np.zeros((ntrain,))
        #     rel_L2_err_train = np.zeros((ntrain,))
        #     rel_L2_err_train1 = np.zeros((ntrain,))
        #     u_L2 = np.zeros((ntrain,))
        #     for batch in train_loader:
        #         u = batch.u.view(batch_size, -1)
        #         u_L2[j] = torch.norm(u)
                
        #         batch = batch.to(device)
        #         out = model(batch)
        #         out = out.view(batch_size, -1)
        #         out = out.view(batch_size, S, S, 2)
        #         out = out[:, cond_f, :, :]
        #         out = out[:, :, cond_f, :]
        #         out = out.view(batch_size, -1)
                
        #         f_gt = batch.f.view(batch_size, S, S, 2)
        #         f_gt = f_gt[:, cond_f, :, :]
        #         f_gt = f_gt[:, :, cond_f, :]
        #         f_gt = f_gt.view(batch_size, -1)

        #         err_train[j] = myloss(out, f_gt).item()
                
        #         out = out.cpu().detach()
        #         f_gt = f_gt.cpu().detach()
        #         rel_L2_err_train[j] = torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
        #         rel_L2_err_train1[j] = myloss.rel_sqrt(out.view(batch_size, -1), f_gt.view(batch_size, -1))

        #         # plot_index = [0,1, 12,13,14,15,16,17,18,19,118,119, 133, 139]
                
        #         if_plot = 0
        #         if (rel_L2_err_train[j] >= 0.1) and (if_plot == 1): 
                
        #             b_plot = out[0,:].view(s, s, 2)
        #             b_true_plot = f_gt[0,:].view(s, s, 2)
        #             # plt_b(b_true_plot, b_plot, base_dir, j, rel_L2_err_train[j],'train') 
                    
        #             u_plot = u.view(batch_size, S, S, 2)
        #             u_plot = u_plot[:, cond_f, :, :]
        #             u_plot = u_plot[:, :, cond_f, :]
        #             u_plot = u_plot.view(s, s, 2) 
        #             plt_b_u(u_plot, b_true_plot, b_plot, base_dir, index[train_index[j]].numpy(), rel_L2_err_train[j], 'train_large')  
                
        #         j += 1
                
        #     print('Relative L2 error for train data:', (np.sum(rel_L2_err_train)/ntrain)) 
                
        #     n_plot = N_data
        #     plt.rcParams.update({'font.size': 15}) 
        #     fig, ax1 = plt.subplots(figsize=(10, 5))
        #     # ax1.plot(err_train, color='darkorange')
        #     ax1.plot(rel_L2_err_train[:n_plot], color='darkorange')
        #     ax1.set_xlabel('data index')
        #     ax1.set_ylabel(r'rel $L^2$ err of $b$', color='darkorange')
        #     ax1.tick_params(axis='y', colors='darkorange') 
        #     # ax1.set_xticks(np.linspace(0, n_plot, 11, dtype=int))
        #     ax1.grid(True, axis='x', linestyle='--')

        #     ax2 = ax1.twinx() 
        #     ax2.plot(strain_mean[:n_plot], color='forestgreen')
        #     ax2.set_ylabel(r'mean strain', color='forestgreen')
        #     ax2.tick_params(axis='y', colors='forestgreen')

        #     plt.title('%s, MGN' % DATA_NAME)
        #     plt.tight_layout()
        #     plt.savefig('%s/%s_rel_L2_err_strain.png' % (base_dir, DATA_NAME), format='png')
            
           
        # if compute_error_valid == 1:
        #     j = 0
        #     rel_L2_err_valid = 0
        #     err_valid = np.zeros((nvalid,))
        #     for batch in valid_loader:
        #         batch = batch.to(device)
        #         out = model(batch)
        #         out = out.view(batch_size_valid, -1)
        #         out = out.view(batch_size_valid, S, S, 2)
        #         out = out[:, cond_f, :, :]
        #         out = out[:, :, cond_f, :]
        #         out = out.view(batch_size_valid, -1)
        #         f_gt = batch.f.view(batch_size_valid, S, S, 2)
        #         f_gt = f_gt[:, cond_f, :, :]
        #         f_gt = f_gt[:, :, cond_f, :]

        #         err_valid = myloss(out.view(batch_size_valid, -1), f_gt.view(batch_size_valid, -1)).item()
        #         print('err_valid:', err_valid)
                
        #         out = out.cpu().detach()
        #         f_gt = f_gt.cpu().detach()
        #         rel_L2_err_valid += torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
                
        #         if j<10:
        #             b_plot = out[0,:].view(s, s, 2)
        #             b_true_plot = f_gt[0,:].view(s, s, 2)
        #             plt_b(b_true_plot, b_plot, base_dir, j, 'valid')
                
        #         j +=1
                
        #     print('Relative L2 error for valid data:', (rel_L2_err_valid/nvalid)) 
        
        # compute_error_test = 1    
        # if compute_error_test == 1:
        #     j = 0
        #     rel_L2_err_test = 0
        #     err_b_test = np.zeros((ntest,))
        #     err_u_test = np.zeros((ntest,))
        #     for batch in test_loader:
        #         batch = batch.to(device)
        #         out = model(batch)
        #         out = out.view(batch_size_test, -1)
        #         out = out.view(batch_size_test, S, S, 2)
        #         out = out[:, cond_f, :, :]
        #         out = out[:, :, cond_f, :]
        #         out = out.view(batch_size_test, -1)
        #         f_gt = batch.f.view(batch_size_test, S, S, 2)
        #         f_gt = f_gt[:, cond_f, :, :]
        #         f_gt = f_gt[:, :, cond_f, :]
        #         f_gt = f_gt.view(batch_size_test, -1)

        #         # err_test = myloss(out.view(batch_size_test, -1), f_gt.view(batch_size_test, -1)).item()
        #         # print('err_test:', err_test)
                
        #         out = out.cpu().detach()
        #         f_gt = f_gt.cpu().detach()
        #         # rel_L2_err_test += torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
        #         err_b_test[j] = torch.sum(torch.norm(out-f_gt, 2, 1)/torch.norm(f_gt,2,1))
        #         # print(rel_L2_err_test)
                
                
        #         ###### solve u  ######                
        #         def func(u):
        #             u = torch.tensor(u, dtype=torch.float32).view(s,s,2)
        #             u_new = u_true.clone()
        #             u_new[3: s+3, 3: s+3, :] = u
                    
        #             u_new = u_new.view(-1, 2)

        #             input = Data(x=batch.x, u=u_new, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
        #             f = model.linear(input).view(S, S, 2).cpu().detach()
        #             f_gt = batch.f.squeeze().view(S, S, 2).cpu().detach()
        #             y = (f-f_gt)[mask_bc_logic]
        #             return y
                
        #         u_true = batch.u.view(S, S, 2).detach().cpu()
        #         data_u0 = (1 - mask_bc) * u_true
        #         data_u0 = data_u0[mask_bc_logic]
        #         u, info, ier, msg = fsolve(func, data_u0, xtol=1e-6, full_output=True)
                
                    
        #         u = torch.tensor(u, dtype=torch.float32)
        #         u_true_i = u_true[mask_bc_logic]
        #         err_u_test[j] = torch.norm(u-u_true_i.view(-1,))/torch.norm(u_true_i.view(-1,))
                
        #         if_plot = 1
        #         if j < 5 and if_plot == 1:
        #             b_plot = out[0,:].view(s, s, 2)
        #             b_true_plot = f_gt[0,:].view(s, s, 2)
        #             # plt_b(b_true_plot, b_plot, base_dir, j, b_true_plot-b_plot, 'test')
        #             u_plot = u.view(s, s, 2)
        #             u_true_plot = u_true_i.view(s, s, 2)
        #             plt_b_u( b_plot, b_true_plot, u_plot, u_true_plot, base_dir, j, 'test')
                
        #         j +=1
                
        #     print('Relative L2 error for test data:', (np.mean(err_b_test)) )
        #     print('Relative L2 error for test data u:', (np.mean(err_u_test)) )
        
        
        ################### plot k1(lambda) #################
        if plot_g == 1:
            lambda_min = 0.8
            lambda_max = 1.2
            lambdaa = torch.linspace(lambda_min, lambda_max, 100)
            lambdaa_cuda = lambdaa.unsqueeze(1).to(device)
            lambdaa_1_cuda = torch.ones_like(lambdaa_cuda)
            k1_NN = (model.phi_MGN(lambdaa_cuda)-model.phi_MGN(lambdaa_1_cuda)).cpu().detach().numpy()

            fontsize = 25
            plt.rcParams.update({'font.size': fontsize}) 
            fig, ax = plt.subplots(figsize = (6, 5.5))
            ax.plot(lambdaa, k1_NN, color='deepskyblue', linewidth=3.0, linestyle='--')
            ax.set_xlabel(r'$\lambda$')
            # ax.set_ylabel(r'$g(\lambda)$')
            plt.title(r'$g(\lambda)$')
            
            plt.tight_layout()
            plt.savefig('%s/MD_g.png' % (base_dir), format='png', dpi=300)
            plt.close(fig)
        
        
        ################### plot 1D k2(|xi|) #################
        # Nxi=50
        # xi_norm = torch.linspace(0.001, delta, Nxi)
        # xi_norm_cuda = xi_norm.unsqueeze(1).to(device)
        # k2_NN = (model.phi_2(xi_norm_cuda)).cpu().detach().numpy()
        # plt.rcParams.update({'font.size': 15}) 
        # fig, ax = plt.subplots(figsize = (6,5))
        # fontsize = 15
        # plt.plot(xi_norm, k2_NN, color='darkorange', linewidth=2, linestyle='--')
        # # plt.legend()
        # plt.xlabel(r'$|\xi|$', fontsize=fontsize)
        # plt.ylabel(r'$k_2^{NN}(\xi)$', fontsize=fontsize)
        # plt.tight_layout()
        # plt.savefig('%s/MD_k2.png' % (base_dir), format='png')
        # plt.close()
        
        ################### plot k2(xi) #################
        if plot_k == 1:
            Nxi = 300
            xi = torch.linspace(-delta, delta, Nxi)
            Xi1, Xi2 = torch.meshgrid(xi, xi)
            Xi = torch.cat([Xi1.view(Nxi, Nxi, 1), Xi2.view(Nxi, Nxi, 1)], dim=2).reshape(-1,2)
            # Xi_norm = torch.sqrt(Xi[:, :, 0]**2 + Xi[:, :, 1]**2).to(device)
            Xi_cuda = Xi.to(device)
            # visible = (Xi1 ** 2.0 + Xi2 ** 2.0 < (delta_train + 1e-4) ** 2)
            k2_NN = (model.phi_2(Xi_cuda/delta)).cpu().detach().numpy().reshape(Nxi, Nxi)
            visible = (Xi1 ** 2.0 + Xi2 ** 2.0 < (delta + 1e-4) ** 2)
            k2_NN_masked = np.where(visible, k2_NN, np.nan)
            
            plt.rcParams.update({'font.size': 25}) 
            fig, ax = plt.subplots(figsize = (6,5))
            fontsize = 25
            # img = plt.imshow(k2_NN.reshape(Nxi, Nxi), extent=[-delta, delta, -delta, delta], cmap='plasma', interpolation='spline16')
            original_cmap = plt.get_cmap('Purples')
            start, stop = 0.3, 1.0
            new_colors = original_cmap(np.linspace(start, stop, 256))
            new_cmap = mcolors.LinearSegmentedColormap.from_list('Purples', new_colors)
            img = plt.imshow(k2_NN_masked.reshape(Nxi, Nxi), extent=[-delta, delta, -delta, delta], cmap=new_cmap, interpolation='nearest')
            cbar = plt.colorbar(img)
            plt.title(r'$k(\boldsymbol{\xi})$', fontsize=fontsize)
            plt.xlabel(r'$\xi_1$', fontsize=fontsize)
            plt.ylabel(r'$\xi_2$', fontsize=fontsize)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig('%s/MD_k_2d.png' % (base_dir), format='png', dpi=300)
            plt.close()
            
        print('Done!')


if __name__ == "__main__":
    
    main()
