
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import sys
import math
import sklearn.metrics
import scipy
import scipy.integrate as integrate
import time
import torch

np.random.seed(123)

current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, 'BlatzKo_data_1d')
os.makedirs(path, exist_ok=True)

N_data = 400
xmin, xmax = 0, 1
Nx = 193
# Nx = 257
h = (xmax-xmin)/(Nx-1)
xh = np.linspace(xmin, xmax, Nx)
# m_fact = 3
# delta = m_fact*h
delta = 0.25
m_fact = int(delta/h)
Nx_all = Nx+2*m_fact
xh_all = np.linspace(xmin-delta, xmax+delta, Nx_all)

coords = xh_all
displacement = np.zeros((N_data, Nx_all))
bodyforce = np.zeros((N_data, Nx_all))

# model
Pi = np.pi
# u_fun = lambda A, a, b, x: A*np.sin(a*x+b)
mu = 0.3846
c = 2*mu/math.pi/delta**2
# g_fun = lambda x: np.ones_like(x)
# g_fun = lambda x: x**2
# g_fun = lambda x: np.abs(x)*np.exp(-(x)**2)*(delta-np.abs(x))
# g_fun = lambda x: np.abs(x)*np.cos(3*np.pi*np.abs(x))
# g_fun = lambda x: np.abs(x)*np.exp(-50*(x)**2)*(delta-np.abs(x))
# g_fun = lambda x: np.pi*(x - x**(-3)) + np.sin(np.pi*x) + np.sin(2*np.pi*x)
# k_fun = lambda x: np.exp(-50*(x)**2)*(delta-np.abs(x))
g_fun = lambda x: x - x**(-3)
k_fun = lambda x: 2*c*np.cos(np.pi*x)

err1 = np.zeros((N_data,Nx))
err2 = np.zeros((N_data,Nx))
# Lambdaa = np.zeros((N_data,Nx))


nfeq = 40
feq = np.linspace(0, nfeq-1, nfeq).reshape(-1,1)
u_fun = lambda a, b, x: np.sum(a*np.exp(-feq/nfeq)*np.sin(feq*np.pi*x)+ b*np.exp(-feq/nfeq)*np.cos(feq*np.pi*x) , axis=0)

# xh = np.linspace(0,1,11).reshape(1,-1)

# print(u(a,b,xh))

# for n in range(N_data):
n = 0
while n < N_data:
    print(n)
    # maxium = 8*Pi
    # A = np.random.uniform(0.01, 0.02)
    # a = np.round(np.random.uniform(0, maxium), 2)
    # b = np.round(np.random.uniform(0, maxium), 2)
    # u = u_fun(A, a, b, xh_all)
    # maxium = 0.0015
    # maxium = 0.001
    maxium = 0.001
    a = np.random.uniform(-maxium, maxium, nfeq).reshape(-1,1)
    b = np.random.uniform(-maxium, maxium, nfeq).reshape(-1,1)
    # d = np.random.uniform(0.001, 0.01, nfeq).reshape(-1,1)
    # d = np.random.uniform(0.2, 2)
    u = u_fun(a, b, xh_all.reshape(1,-1))
    
    displacement[n,:] = u
    
    for i in range(m_fact, Nx+m_fact):
        xi = i*h-delta
        ksi_norm = lambda ksi: np.abs(ksi)
        eta = lambda ksi: u_fun(a, b, ksi+xi)- u[i]
        ksi_plus_eta = lambda ksi: ksi + eta(ksi)
        ksi_plus_eta_norm = lambda ksi: np.abs(ksi_plus_eta(ksi))
        # lambdaa = lambda ksi: ksi_plus_eta_norm(ksi)/ksi_norm(ksi)
        extension = lambda ksi: ksi_plus_eta_norm(ksi) - ksi_norm(ksi)
        lambdaa  = lambda ksi: 1+ extension(ksi)/(ksi_norm(ksi))
        # f = lambda ksi: ksi_plus_eta(ksi)/ksi_plus_eta_norm(ksi)* 2*c/ksi_norm(ksi)*(lambdaa(ksi)-1/lambdaa(ksi)**3)*g_fun(ksi_norm(ksi))
        f = lambda ksi: ksi_plus_eta(ksi)/ksi_plus_eta_norm(ksi)* g_fun(lambdaa(ksi))*k_fun(ksi_norm(ksi))
        # f = lambda ksi: ksi_plus_eta(ksi)/ksi_plus_eta_norm(ksi)* 2*c/ksi_norm(ksi)*g_fun(ksi_norm(ksi))
        I1, err1[n,i-m_fact] = integrate.quad(f, 0, delta, epsabs=1e-16, epsrel=1e-16)
        I2, err2[n,i-m_fact] = integrate.quad(f, -delta, 0, epsabs=1e-16, epsrel=1e-16)
        # print(err1)
        # print(err2)
        bodyforce[n,i] = I1+I2
        # Lambdaa[n,i-m_fact] = np.max(lambdaa(ksi))
    
    # # print(np.max(Lambdaa[n,:]))    
    print(np.max(err1[n,:]), np.max(err2[n,:]))
    
    
    #######  plot data ###########
    # plt.rcParams.update({'font.size': 15}) 
    # fig, ax = plt.subplots(figsize = (12,6))
    # # fig.suptitle('HGO:P')
    # plt.subplot(1,2,1)
    # plt.plot(xh_all,u)
    # plt.title("u")
    
    # plt.subplot(1,2,2)
    # plt.plot(xh_all,bodyforce[n,:])
    # plt.title("b")
    
    # plt.savefig('%s/data.png' % path, format='png')
    # print("---")
    
    ########### compute lambda  ##############
    # s = Nx
    # S = Nx_all
    # dx = h
    # ndata = 1
    # data_x = torch.from_numpy(coords)
    # data_u = torch.from_numpy(displacement[n,:]).reshape(1,-1)

    # ksi_range = torch.range(-m_fact, m_fact).int()
    # ksi_range = ksi_range[ksi_range != 0]
    # n_ksi = 2*m_fact
    # data_ksi = ksi_range*dx
    # data_ksi = data_ksi.repeat(ndata,s,1)

    # data_eta = torch.zeros((ndata, s, n_ksi))
    # for k in range(s):
    #     data_eta[:,k,:] = (data_u[:,m_fact+k+ksi_range].reshape(-1,1,n_ksi)-data_u[:,m_fact+k].reshape(-1,1,1)).squeeze()
    # ksi_plus_eta_norm = torch.abs(data_ksi+data_eta)
    # ksi_norm = torch.abs(data_ksi)
    # extension = ksi_plus_eta_norm - ksi_norm
    # lambdaa = 1.0 + extension / (ksi_norm + 1e-9)

    # lambda_min_data = torch.min(torch.min(lambdaa, dim=1).values, dim=1).values
    # lambda_max_data = torch.max(torch.max(lambdaa, dim=1).values, dim=1).values
    # print(torch.min(lambda_min_data), torch.max(lambda_max_data))
    
    
    if np.max(err1[n,:])<=1e-10 and np.max(err2[n,:])<=1e-10:
        n = n+1
    

print(np.max(err1))
print(np.max(err2))

data_name = "BK_ex37_ndata_%s_Nx_%s_delta_%s_h_%s.mat" % (N_data, Nx, delta, h)
scipy.io.savemat(os.path.join(path, data_name), {'coords': coords,'displacement': displacement,'bodyforce':bodyforce})
   
