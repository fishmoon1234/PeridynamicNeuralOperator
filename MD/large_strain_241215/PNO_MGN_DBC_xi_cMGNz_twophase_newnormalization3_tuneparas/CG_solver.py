import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utilities_INO_PD import *
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


# def solver_u_tr(data_test_u, tol_cg, k_cg_max, model, ker_in, ker_width, ker_out, phi_in, phi_width, phi_out, lx):
def solver_u(data_test_u, tol_cg, k_cg_max, model, mask1_bc, tol_ls, i_ls_max, device):


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
        lu = model(input)
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
            lu = model(input)
            lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
            ####
            r_oldd = r
            r = -mask_bc*(lu - b)
            #lu = None
            i += 1

        return u, r

    ###### load the trained INO PD  model

    #model = EGKN(ker_in, ker_width, ker_out, phi_in, phi_width, phi_out).to(device)
    #model.load_state_dict(torch.load(model_filename))
    # model.eval()

    ###### Static solver parameters (Polak-Ribiere conjugate gradient)
    # tol_ls = 1e-4
    # i_ls_max = 20

    ###### PD horizon and domain discretization
    error_u = 0
    error_f = 0
    ntest_u = len(data_test_u)
    mask_bc = torch.cat([mask1_bc, mask1_bc], dim=0)
    
    for i_data in range(ntest_u):
        xuf = data_test_u[i_data]
        x = xuf.x[:,0].unsqueeze(1)
        y = xuf.x[:,1].unsqueeze(1)
        x_test = xuf.x
        delta = xuf.delta
        coords = torch.cat([x, y], dim=1)
        edge_index = xuf.edge_index
        edge_attr = xuf.edge_attr
        nx = np.sqrt(x.size(0)).astype(int)
        ny = nx
        
        ##### Solver Initialization
        # u1_bc, mask1_bc = bc_u1(x, y)
        # u2_bc, mask2_bc = bc_u2(x, y)
        # u1_bc = (1 - mask1_bc) * xuf.u[:,0].unsqueeze(1) + mask1_bc * xuf.u0[:,0].unsqueeze(1)
        # u2_bc = (1 - mask1_bc) * xuf.u[:,1].unsqueeze(1) + mask1_bc * xuf.u0[:,1].unsqueeze(1)
        ### Ave Initialization ###
        # u1_bc_ave = torch.sum((1 - mask1_bc) * xuf.u[:,0].unsqueeze(1))/nx/4
        # u2_bc_ave = torch.sum((1 - mask1_bc) * xuf.u[:,1].unsqueeze(1))/nx/4
        # one = torch.ones(xuf.u[:,0].size()).to(u1_bc_ave.device)
        # u1_bc = (1 - mask1_bc) * xuf.u[:,0].unsqueeze(1) + mask1_bc* u1_bc_ave* one.unsqueeze(1)
        # u2_bc = (1 - mask1_bc) * xuf.u[:,1].unsqueeze(1) + mask1_bc* u2_bc_ave* one.unsqueeze(1)
        
        ### Zero Initialization  ###
        u1_bc = (1 - mask1_bc) * xuf.u[:,0].unsqueeze(1)
        u2_bc = (1 - mask1_bc) * xuf.u[:,1].unsqueeze(1)
        
        u1_0 = u1_bc
        u2_0 = u2_bc
        u =  torch.cat([u1_0, u2_0], dim=0).to(device)
        # mask_bc = torch.cat([mask1_bc, mask2_bc], dim=0).to(device)
        

        #### INOPD model evaluation
        u1 = u[:nx * ny]
        u2 = u[nx * ny :2 * nx * ny]
        u_data = torch.cat([u1, u2], dim=1)
        input = Data(x=x_test.to(torch.float32), u=u_data, edge_index=edge_index, edge_attr=edge_attr, delta=delta).to(device)

        lu = model(input)
        lu = torch.cat([lu[:,0],lu[:,1]], dim=0).unsqueeze(1)
        ###
        # b1 = body_force1(x ,y)
        # b2 = body_force2(x ,y)
        # b =  torch.cat([b1, b2], dim=0).to(device)
        b = xuf.f.view(-1,1).to(device)
        # b = torch.zeros(lu.size()).to(device)

        r = mask_bc*(lu - b)
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
            err = torch.norm(mask_bc*(u - u_old))/torch.norm(mask_bc*(u))
            #err = torch.norm(r)

            #input.detach()
            #lu.detach()
            #r_old.detach()
            #u_old.detach()
            #print(f'CG itr: {k},   relative l2 error: {err}, residual: {torch.norm(r)}')

        u_gt = torch.cat([xuf.u[:,0], xuf.u[:,1]], dim=0).unsqueeze(1).to(device)
        error_u += torch.norm(mask_bc*(u - u_gt)) / (torch.norm(mask_bc*u_gt)+1e-9)
        
        lu = model(xuf.to(device))
        lu = torch.cat([lu[:, 0], lu[:, 1]], dim=0).unsqueeze(1)
        f_gt = torch.cat([xuf.f[:, 0], xuf.f[:, 1]], dim=0).unsqueeze(1).to(device)
        error_f += torch.norm(mask_bc * (lu - f_gt)) / torch.norm(mask_bc * (f_gt))
        # f_gt = torch.zeros(lu.size()).to(device)
        # error_f += torch.norm(mask_bc * (lu - f_gt)) / torch.norm(mask_bc)

    return error_u, error_f, u
