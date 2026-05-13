
# import matplotlib.pyplot as plt
import random
import os
import numpy as np
import sys
import math
import scipy
import scipy.integrate as integrate
import time


def unsorted_segment_mean(data, segment_ids, num_segments):
    
    segment_ids_expanded = segment_ids[:, np.newaxis]
    segment_ids = np.broadcast_to(segment_ids_expanded, (segment_ids.shape[0], data.shape[1]))
    
    result_shape = (num_segments, data.shape[1])
    result = np.zeros(result_shape, dtype=data.dtype) 
    count = np.zeros(result_shape, dtype=data.dtype)

    for i in range(len(segment_ids)):
        result[segment_ids[i]] += data[i]
        count[segment_ids[i]] += 1

    count_clamped = np.clip(count, a_min=1, a_max=None)
    return result / count_clamped

    # segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.shape[1])
    # result = data.new_full(result_shape, 0)  # Init empty result tensor.
    # count = data.new_full(result_shape, 0)
    # result.scatter_add_(0, segment_ids, data)
    # count.scatter_add_(0, segment_ids, np.ones_like(data))
    # return result / count.clamp(min=1)

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.shape[1])
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.shape[1])
    result.scatter_add_(0, segment_ids, data)
    return result

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ndata", type=int, default=10, help="number of data to generate in this job")
parser.add_argument("--seed", type=int, default=123, help="random seed for this job")
parser.add_argument("--job_id", type=int, default=-1, help="Slurm array task id (for naming outputs)")
args = parser.parse_args()

print("PID:", os.getpid())
np.random.seed(args.seed)

N_data = args.ndata

ex = f"ex5_job{args.job_id:02d}" if args.job_id >= 0 else "ex5"

print('PID:', os.getpid())

# np.random.seed(123)
# make new directory
# path = 'BlatzKo_Dir_data_PNO'

current_dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(current_dir, 'DATA')
os.makedirs(path, exist_ok=True)

# ex = 'ex1'

# N_data = 10
xmin, xmax = 0, 1
Nx = 33  # h=2**(-5)
h = (xmax-xmin)/(Nx-1)
delta = 1/4
m_fact = 8
# delta = m_fact*h
Nx_all = Nx+2*m_fact
xh_all = np.linspace(xmin-delta, xmax+delta, Nx_all)
[Yh_all, Xh_all] = np.meshgrid(xh_all, xh_all)
grid = np.concatenate((Xh_all.reshape(-1,1), Yh_all.reshape(-1,1)), axis=1)

# pwd = sklearn.metrics.pairwise_distances(grid)
# edge_index = np.vstack(np.where((pwd <= delta) & (pwd > 1e-10)))
# n_edges = edge_index.shape[1]
# row, col = edge_index

coords = grid
displacement = np.zeros((N_data, Nx_all, Nx_all, 2))
bodyforce = np.zeros((N_data, Nx_all, Nx_all, 2))

# # model
# u1_fun = lambda a1, a2, x1, x2: np.sin(a1*x1)*np.sin(a2*x2)
# u2_fun = lambda a1, a2, x1, x2: np.cos(a1*x1)*np.cos(a2*x2)
Pi = np.pi

# Define frequency parameters
nfeq = 10
feq = np.linspace(0, nfeq-1, nfeq).reshape(-1,1)
decay = np.exp(-feq / nfeq)
x0 = 0.49
C = 0.01

u1_fun = lambda a, b, x1, x2: (
    np.sum(
        a.reshape(-1, 1, 1)*decay.reshape(-1, 1, 1)*np.sin(feq.reshape(-1, 1, 1)*np.pi*np.asarray(x1)) +
        b.reshape(-1, 1, 1)*decay.reshape(-1, 1, 1)*np.cos(feq.reshape(-1, 1, 1)*np.pi*np.asarray(x2)),
        axis=0
    )
    + C * (np.asarray(x1) >= x0).astype(float)
)

u2_fun = lambda a, b, x1, x2: (
    np.sum(
        a.reshape(-1, 1, 1)*decay.reshape(-1, 1, 1)*np.sin(feq.reshape(-1, 1, 1)*np.pi*np.asarray(x1)) +
        b.reshape(-1, 1, 1)*decay.reshape(-1, 1, 1)*np.cos(feq.reshape(-1, 1, 1)*np.pi*np.asarray(x2)),
        axis=0
    )
)
mu = 0.3846
c = 2*mu/math.pi/delta**2
# g_fun = lambda x: np.ones_like(x)
# g_fun = lambda x: x**2
g_fun = lambda x: x-x**(-3)
k_fun = lambda x: 2*c*np.exp(-50*x**2)*(delta-np.abs(x))


N_theta = 51
theta = np.linspace(0, 2*math.pi, N_theta)
dtheta = 2*np.pi/(N_theta-1)
err = np.zeros((N_data,))

for n in range(N_data):
    start_time = time.time()  # Record start time for this iteration
    # Generate random coefficients for u1_fun and u2_fun
    maxium = 0.001
    a1 = np.random.uniform(-maxium, maxium, nfeq)  # coefficients for sin terms in u1
    b1 = np.random.uniform(-maxium, maxium, nfeq)  # coefficients for cos terms in u1
    a2 = np.random.uniform(-maxium, maxium, nfeq)  # coefficients for sin terms in u2
    b2 = np.random.uniform(-maxium, maxium, nfeq)  # coefficients for cos terms in u2
    
    u1 = u1_fun(a1, b1, Xh_all, Yh_all)
    u2 = u2_fun(a2, b2, Xh_all, Yh_all)
    
    displacement[n,:,:,0] = u1
    displacement[n,:,:,1] = u2
    
    I_1 = np.zeros((N_theta,))
    I_2 = np.zeros((N_theta,))
    err_max = 0
    
    start_time_in = time.time()
    for i in range(m_fact, Nx+m_fact):
        for j in range(m_fact, Nx+m_fact):
            # print(j)
            # start_time = time.time()
            xij_1, xij_2 = i*h-delta, j*h-delta
            # ksi_1 = lambda x: x-xij_1
            # ksi_2 = lambda x: x-xij_2
            ksi_norm = lambda xi1, xi2: np.sqrt(xi1**2+xi2**2)
            eta_1 = lambda xi1, xi2: u1_fun(a1, b1, xij_1+xi1, xij_2+xi2)- u1[i,j]
            eta_2 = lambda xi1, xi2: u2_fun(a2, b2, xij_1+xi1, xij_2+xi2)- u2[i,j]
            ksi_plus_eta_1 = lambda xi1, xi2: xi1 + eta_1(xi1,xi2)
            ksi_plus_eta_2 = lambda xi1, xi2: xi2 + eta_2(xi1,xi2)
            ksi_plus_eta_norm = lambda xi1, xi2: np.sqrt(ksi_plus_eta_1(xi1, xi2)**2+ksi_plus_eta_2(xi1, xi2)**2)
            lambdaa = lambda xi1, xi2: ksi_plus_eta_norm(xi1, xi2)/ksi_norm(xi1, xi2)
            f_1 = lambda xi1, xi2: ksi_plus_eta_1(xi1, xi2)/ksi_plus_eta_norm(xi1, xi2)*g_fun(lambdaa(xi1,xi2))*k_fun(ksi_norm(xi1,xi2))
            f_1_polar = lambda r, theta: r*f_1(r*np.cos(theta), r*np.sin(theta))
            f_2 = lambda x1, x2: ksi_plus_eta_2(x1,x2)/ksi_plus_eta_norm(x1,x2)*g_fun(lambdaa(x1,x2))*k_fun(ksi_norm(x1,x2))
            f_2_polar = lambda r, theta: r*f_2(r*np.cos(theta), r*np.sin(theta))
            # print(time.time() -start_time)
            # start_time = time.time()
            for k in range(N_theta):
                fk_1 = lambda r: f_1_polar(r,theta[k])
                I_1[k], err1 = integrate.quad(fk_1, 0, delta, epsabs=1e-8, epsrel=1e-8)
                fk_2 = lambda r: f_2_polar(r,theta[k])
                I_2[k], err2 = integrate.quad(fk_2, 0, delta, epsabs=1e-8, epsrel=1e-8)
                err_max = max(err_max, err1, err2)
            bodyforce[n,i,j,0] =  dtheta/2*(I_1[0]+2*np.sum(I_1[1:-1])+I_1[-1]) 
            bodyforce[n,i,j,1] =  dtheta/2*(I_2[0]+2*np.sum(I_2[1:-1])+I_2[-1])
            # print(err)
            # print(time.time() -start_time)
        print(i, time.time() - start_time_in)
            
    err[n] = err_max
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print(f"{n}: err_max: {err_max}, time: {elapsed_time:.2f} s")
    
        
    if err_max <= 1e-8:
        n = n+1
    
print(np.max(err))
    
scipy.io.savemat(os.path.join(path, "BK_2d_%s_ndata_%s_Nx_%s.mat" % (ex, N_data, Nx)), {'coords': coords,'displacement': displacement,'bodyforce':bodyforce})
        
    

'''
#u_plot = np.loadtxt('./HGO_Dir_data_INOPD/displacement.txt')
test_ux = np.zeros((21, 21))
test_uy = np.zeros((21, 21))

for xi in range(0, 21):
    for xj in range(0, 21):
        x_cords = xi * 1.0 / 20
        y_cords = xj * 1.0 / 20
        test_ux[xi, xj] = u([x_cords, y_cords])[0]
        test_uy[xi, xj] = u([x_cords, y_cords])[1]

plt.imshow(micro)
plt.colorbar()
plt.show()
plt.imshow(test_ux)
plt.colorbar()
plt.show()
plt.imshow(test_uy)
plt.colorbar()
plt.show()
'''

# u = np.loadtxt('./BlatzKo_Dir_data_PNO/displacement.txt')
# x = np.loadtxt('./BlatzKo_Dir_data_PNO/coords.txt')
# n = 21
# u_grad = get_u_gradient(u,x,n)
# u_grad_norm = np.linalg.norm(u_grad, axis=[2,3])
# print(np.max(u_grad_norm))
   
