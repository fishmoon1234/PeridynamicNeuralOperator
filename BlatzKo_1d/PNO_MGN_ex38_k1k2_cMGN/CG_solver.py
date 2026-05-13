import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
from egnn_gcl import E_GCL_GKN
from timeit import default_timer
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import bisect

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def solver_u_tr(data_test_u, model, mask_bc, tol_cg, k_cg_max, tol_ls, i_ls_max, device):
    # one test data

    # Line search subroutine for static solver
    def line_search(u, r, p):
        alpha_0 = 1.0
        #u_old = u
        r_oldd = r
        u = u + alpha_0 * p
        #### INOPD model evaluation
        # u1 = u[:nx*ny]
        # u2 = u[nx*ny:2*nx*ny]
        # u_data = torch.cat([u1,u2], dim=1)
        # input = Data(x=x_test.to(torch.float32), u=u_data, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
        input = Data(x=x_test.to(torch.float32), u=u, ksi=ksi_test).to(device)
        lu_in = model(input)
        # lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
        #####
        
        lu = torch.zeros(u.size()).to(device)
        lu[m_fact:s+m_fact,0] = lu_in
        
        r = -mask_bc*(lu - b)
        i = 1

        alpha_1 = -alpha_0 * torch.dot(r.squeeze(), p.squeeze()) / torch.dot((r - r_oldd).squeeze(), p.squeeze())
        alpha = alpha_0
        while (abs(alpha/alpha_1) > tol_ls) & (i < i_ls_max):
            alpha = -alpha * torch.dot(r.squeeze(), p.squeeze())/torch.dot((r - r_oldd).squeeze(), p.squeeze())
            u = u + alpha * p
            #### INOPD model evaluation
            # u1 = u[:nx * ny]
            # u2 = u[nx * ny:2 * nx * ny]
            # u_data = torch.cat([u1, u2], dim=1)
            input = Data(x=x_test.to(torch.float32), u=u, ksi=ksi_test).to(device)
            lu_in = model(input)
            
            lu = torch.zeros(u.size()).to(device)
            lu[m_fact:s+m_fact,0] = lu_in
            # lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
            ####
            r_oldd = r
            r = -mask_bc*(lu - b)
            #lu = None
            i += 1
        # print(f'line search itr = {i}')

        return u, r

    ###### load the trained INO PD  model

    #model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out).to(device)
    #model.load_state_dict(torch.load(model_filename))
    model.eval()

    ###### Static solver parameters (Polak-Ribiere conjugate gradient)
    # tol_ls = 1e-4
    # i_ls_max = 20

    ###### PD horizon and domain discretization
    error_gt = 0
    error_f = 0
    # ntest_u = len(data_test_u)
    
    # mask_bc = torch.cat([mask1_bc, mask1_bc], dim=0)
    
    # for i_data in range(ntest_u):
    # data_train.append(Data(x=x_train, u0=u0_train[j, :, :], u=u_train[j, :, :], f=f_train[j, :, :], ksi=ksi_train[j, :, :], eta=eta_train[j, :, :]))
    # xuf = data_test_u[i_data]
    xuf = data_test_u
    # x = xuf.x.unsqueeze(1)
    # y = xuf.x[:,1].unsqueeze(1)
    x_test = xuf.x
    # delta = xuf.delta
    # coords = torch.cat([x, y], dim=1)
    # edge_index = xuf.edge_index
    # edge_attr = xuf.edge_attr
    # nx = np.sqrt(x_test.size(0)).astype(int)
    # ny = nx
    ksi_test = xuf.ksi
    # eta = xuf.eta
    # u_test = xuf.u
    
    s, n_ksi = ksi_test.size()
    m_fact = n_ksi//2
    
    ##### Solver Initialization
    # u1_bc = (1 - mask1_bc) * xuf.u[:,0].unsqueeze(1) + mask1_bc * xuf.u0[:,0].unsqueeze(1)
    # u2_bc = (1 - mask1_bc) * xuf.u[:,1].unsqueeze(1) + mask1_bc * xuf.u0[:,1].unsqueeze(1)
    # u1_0 = u1_bc
    # u2_0 = u2_bc
    # u =  torch.cat([u1_0, u2_0], dim=0).to(device)
    # mask_bc = torch.cat([mask1_bc, mask2_bc], dim=0).to(device)
    
    
    u = (1 - mask_bc) * xuf.u + mask_bc *xuf.u0
    
    # u_bc_ave = torch.sum((1 - mask_bc) * xuf.u)/n_ksi
    # one = torch.ones(xuf.u.size()).to(u_bc_ave.device)
    # u = (1 - mask_bc) * xuf.u + mask_bc* u_bc_ave* one

    #### INOPD model evaluation
    # u1 = u[:nx * ny]
    # u2 = u[nx * ny :2 * nx * ny]
    # u_data = torch.cat([u1, u2], dim=1)
    # input = Data(x=x_test.to(torch.float32), u=u_data, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
    input = Data(x=x_test, u=u, ksi=ksi_test)
    #with torch.no_grad():
    lu_in = model(input)
            
    lu = torch.zeros(u.size()).to(device)
    lu[m_fact:s+m_fact,0] = lu_in
    # lu = torch.cat([lu[:,0],lu[:,1]], dim=0).unsqueeze(1)
    ####
    # b1 = body_force1(x ,y)
    # b2 = body_force2(x ,y)
    # b =  torch.cat([b1, b2], dim=0).to(device)
    b = torch.zeros(lu.size()).to(device)

    # r = mask_bc*(lu - b)
    r = lu - b
    p = -r
    k = 0
    err = 1.0
    #with torch.no_grad():
    while (err > tol_cg) & (k < k_cg_max):
        r_old = r
        u_old = u
        u, r = line_search(u, r, p)
        beta = torch.dot(r.squeeze(), (r - r_old).squeeze())/torch.dot(r_old.squeeze(), r_old.squeeze())
        if beta < 0:
            beta = 0.0
        p = -r + beta*p
        k += 1
        err = torch.norm(mask_bc*(u - u_old))/(torch.norm(mask_bc*(u))+1e-9)
        # err = torch.norm((u - u_old))/torch.norm((u))
        #err = torch.norm(r)

        #input.detach()
        #lu.detach()
        #r_old.detach()
        #u_old.detach()
        #print(f'CG itr: {k},   relative l2 error: {err}, residual: {torch.norm(r)}')

    # u_gt = torch.cat([xuf.u[:,0],xuf.u[:,1]], dim=0).unsqueeze(1).to(device)
    u_gt = xuf.u.to(device)
    error_gt += torch.norm(mask_bc*(u - u_gt)) / (torch.norm(mask_bc*u_gt)+1e-9)
    
    print(k)

    #with torch.no_grad():
    
    lu = model(xuf.to(device))
    # lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
    # f_gt = torch.cat([xuf.f[:, 0], xuf.f[:, 1]], dim=0).unsqueeze(1).to(device)
    f_gt = torch.zeros(lu.size()).to(device)
    # error_f += torch.norm(mask_bc * (lu - f_gt)) / torch.norm(mask_bc)
    error_f += torch.norm(lu - f_gt)


        #print(f'l2 error relative to ground truth u: {error_gt/ntest_u}')
        #print(f'l2 error for f: {error_f/ntest_u}')
    return error_gt, error_f, u




def solver_u_val(data_test_u, tol_cg, k_cg_max, model, ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, lx):

############## Boundary Conditions, and Body force:

    def bc_u1(x,y):
        mask_bc_u1 = torch.ones(x.size()).to(device)
        mask_bc_u1[torch.abs(x - 2.2) > 2.2 - 5e-2]  = 0
        mask_bc_u1[torch.abs(y - 2.2) > 2.2 - 5e-2] = 0
        #mask_bc_u1[((x - lx/2) ** 2 + (y - lx/2) ** 2) >  (1.1 ** 2 + 1e-15)] = 0
        #u1_bc = (1 - mask_bc_u1) * xuf.u[:,0].unsqueeze(1) + mask_bc_u1 * 0.0001 * (x - 2.2)
        u1_bc = (1 - mask_bc_u1) * xuf.u[:,0].unsqueeze(1) + mask_bc_u1 * xuf.u0[:,0].unsqueeze(1)
        return u1_bc, mask_bc_u1

    def bc_u2(x,y):
        mask_bc_u2 = torch.ones(x.size()).to(device)
        mask_bc_u2[torch.abs(x - 2.2) > 2.2 - 5e-2]  = 0
        mask_bc_u2[torch.abs(y - 2.2) > 2.2 - 5e-2] = 0
        #mask_bc_u2[((x - lx/2) ** 2 + (y - lx/2) ** 2) >  (1.1 ** 2 + 1e-15)] = 0
        #u2_bc = (1 - mask_bc_u2) * xuf.u[:,1].unsqueeze(1) + mask_bc_u2  * 0.0001 * (y - 2.2)
        u2_bc = (1 - mask_bc_u2) * xuf.u[:,1].unsqueeze(1) + mask_bc_u2 * xuf.u0[:,1].unsqueeze(1)
        return u2_bc, mask_bc_u2

    def body_force1(x,y):
        #mask_domain = torch.ones(x.size())
        #mask_domain[(x ** 2 + y ** 2) < 0.25] = 0
        #bf1 = mask_domain * xuf.f[:,0].unsqueeze(1)
        bf1 = xuf.f[:, 0].unsqueeze(1) * 0.0
        return bf1

    def body_force2(x,y):
        #mask_domain = torch.ones(x.size())
        #mask_domain[(x ** 2 + y ** 2) < 0.25] = 0
        #bf2 = mask_domain * xuf.f[:,1].unsqueeze(1)
        bf2 = xuf.f[:, 1].unsqueeze(1) * 0.0
        return bf2



    # Line search subroutine for static solver
    def line_search(u, r, p):
        alpha_0 = 1.0
        #u_old = u
        r_oldd = r
        u = u + alpha_0 * p
        #### INOPD model evaluation
        u1 = u[:nx*ny]
        u2 = u[nx*ny:2*nx*ny]
        u_data = torch.cat([u1,u2], dim=1)
        input = Data(x=x_test.to(torch.float32), u=u_data, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
        lu, PK = model(input)
        lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
        #####
        r = -mask_bc*(lu - b)
        i = 1

        alpha_1 = -alpha_0 * torch.dot(r.squeeze(), p.squeeze()) / torch.dot((r - r_oldd).squeeze(), p.squeeze())
        alpha = alpha_0
        while (abs(alpha/alpha_1) > tol_ls) & (i < i_ls_max):
            alpha = -alpha * torch.dot(r.squeeze(), p.squeeze())/torch.dot((r - r_oldd).squeeze(), p.squeeze())
            u = u + alpha * p
            #### INOPD model evaluation
            u1 = u[:nx * ny]
            u2 = u[nx * ny:2 * nx * ny]
            u_data = torch.cat([u1, u2], dim=1)
            input = Data(x=x_test.to(torch.float32), u=u_data, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
            lu, PK = model(input)
            lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
            ####
            r_oldd = r
            r = -mask_bc*(lu - b)
            #lu = None
            i += 1
        # print(f'line search itr = {i}')

        return u, r, PK

    ###### load the trained INO PD  model

    #model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out).to(device)
    #model.load_state_dict(torch.load(model_filename))
    model.eval()

    ###### Static solver parameters (Polak-Ribiere conjugate gradient)
    tol_ls = 1e-4
    i_ls_max = 5

    ###### PD horizon and domain discretization

    error_gt = 0
    error_f = 0
    error_pk = 0
    ntest_u = len(data_test_u)
    for i_data in range(ntest_u):
        xuf = data_test_u[i_data]
        x = xuf.x[:,0].unsqueeze(1)
        y = xuf.x[:,1].unsqueeze(1)
        x_test = xuf.x
        delta = xuf.delta
        #x_list = torch.linspace(-lx/2, lx/2, steps=nx)
        #y_list = torch.linspace(-ly/2, ly/2, steps=ny)
        #x, y = torch.meshgrid(x_list, y_list, indexing='xy')
        #x = x.reshape(-1,1)
        #y = y.reshape(-1,1)
        #edge_index = {}
        #edge_attr = {}
        coords = torch.cat([x, y], dim=1)
        #meshgenerator = IrregularMeshGenerator(coords, [nx, ny])
        #edge_index = meshgenerator.ball_connectivity(float(delta))
        #edge_attr = meshgenerator.attributes(theta=0)
        edge_index = xuf.edge_index
        edge_attr = xuf.edge_attr
        nx = np.sqrt(x.size(0)).astype(int)
        ny = nx
        ##### Solver Initialization

        u1_bc, mask1_bc = bc_u1(x, y)
        u2_bc, mask2_bc = bc_u2(x, y)
        u1_0 = u1_bc
        u2_0 = u2_bc
        u =  torch.cat([u1_0, u2_0], dim=0).to(device)
        mask_bc = torch.cat([mask1_bc, mask2_bc], dim=0).to(device)

        #### INOPD model evaluation
        u1 = u[:nx * ny]
        u2 = u[nx * ny :2 * nx * ny]
        u_data = torch.cat([u1, u2], dim=1)
        input = Data(x=x_test.to(torch.float32), u=u_data, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)
        #with torch.no_grad():
        lu, _ = model(input)
        lu = torch.cat([lu[:,0],lu[:,1]], dim=0).unsqueeze(1)
        ####
        b1 = body_force1(x ,y)
        b2 = body_force2(x ,y)
        b =  torch.cat([b1, b2], dim=0).to(device)

        r = mask_bc*(lu - b)
        p = -r
        k = 0
        err = 1.0
        # with torch.no_grad():
        while (err > tol_cg) & (k < k_cg_max):
            r_old = r
            u_old = u
            u, r, PK = line_search(u, r, p)
            beta = torch.dot(r.squeeze(), (r - r_old).squeeze())/torch.dot(r_old.squeeze(), r_old.squeeze())
            if beta < 0:
                beta = 0.0
            p = -r + beta*p
            k += 1
            err = torch.norm(mask_bc*(u - u_old))/torch.norm(mask_bc*(u))
            #err = torch.norm(r)

            #input.detach()
            #lu.detach()
            #r_old.detach()
            #u_old.detach()
            #print(f'CG itr: {k},   relative l2 error: {err}, residual: {torch.norm(r)}')

        u_gt = torch.cat([xuf.u[:,0],xuf.u[:,1]], dim=0).unsqueeze(1).to(device)
        error_gt += torch.norm(mask_bc*(u - u_gt)) / torch.norm(mask_bc*u_gt)


        f_gt = torch.cat([xuf.f[:, 0], xuf.f[:, 1]], dim=0).unsqueeze(1).to(device)
        #with torch.no_grad():
        lu, PK = model(xuf.to(device))
        lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
        error_f += torch.norm(mask_bc * (lu - f_gt)) / torch.norm(mask_bc)

        PK11_gt = xuf.PK[0]
        PK22_gt = xuf.PK[1]
        PK11_ave = -torch.sum(mask1_bc.squeeze() * PK[:, 0, 0])/441.0
        PK22_ave = -torch.sum(mask2_bc.squeeze() * PK[:, 1, 1])/441.0

        #error_pk += torch.sqrt(0.5 * ((PK11_ave - PK11_gt)**2.0/(PK11_gt)**2.0 + (PK22_ave - PK22_gt)**2.0/(PK22_gt)**2.0))
        error_pk += torch.sqrt(0.5 * ((PK11_ave - PK11_gt) ** 2.0 / 0.1684 ** 2.0 + (PK22_ave - PK22_gt) ** 2.0 / 0.1684 ** 2.0))

        #print(f'l2 error relative to ground truth u: {error_gt/ntest_u}')
        #print(f'l2 error for f: {error_f/ntest_u}')
    return error_gt, error_pk
