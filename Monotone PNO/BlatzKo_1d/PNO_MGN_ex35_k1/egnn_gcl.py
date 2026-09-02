from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import math
from utilities_INO_PD import *

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class MGN(nn.Module):
    def __init__(self, phi_1_layer, act_fn=nn.Softplus()):

        super().__init__()
        
        self.act_fn = act_fn
        self.phi_n_layers = len(phi_1_layer) - 1
        phi_in = phi_1_layer[0]
        phi_width = phi_1_layer[1]

        self.W = nn.Parameter(torch.empty(phi_in, phi_width))  # Weight matrix
        self.V = nn.Parameter(torch.empty(phi_in, phi_width))  # Second weight matrix
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(phi_width)) for _ in range(self.phi_n_layers + 1)])  # Biases for z_0 to z_L
        self.bias_last = nn.Parameter(torch.zeros(phi_in))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)

    def forward(self, input):
        z = torch.matmul(input, self.W) + self.biases[0]  # z_0
        for l in range(1, self.phi_n_layers + 1):
            z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z) +z

        phi_out = torch.matmul(self.act_fn(z), self.W.t()) + torch.matmul(input, torch.matmul(self.V, self.V.t())) + self.bias_last
        return phi_out
    
    
class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, phi_1_layer, act_fn=nn.Softplus(), normalize=False,
                 coords_agg='mean', residual=True):
        super().__init__()

        # self.act_fun_xi = act_fun_xi
        self.act_fn = act_fn
        self.phi_MGN = MGN(phi_1_layer, act_fn)
        # self.phi_1 = DenseNet(phi_1_layer, torch.nn.ReLU)
        # self.phi_2 = DenseNet(phi_2_layer, torch.nn.Tanh)
        # self.phi_2 = DenseNet(phi_2_layer, self.act_fun_xi)
        # self.phi_2 = DenseNet(phi_2_layer, torch.nn.GELU)

        # self.phi_n_layers = len(phi_1_layer) - 1
        # phi_in = phi_1_layer[0]
        # phi_width = phi_1_layer[1]
        # self.W = nn.Parameter(torch.empty(phi_in, phi_width))  # Weight matrix
        # self.V = nn.Parameter(torch.empty(phi_in, phi_width))  # Second weight matrix
        # self.biases = nn.ParameterList(
        #     [nn.Parameter(torch.zeros(phi_width)) for _ in range(self.phi_n_layers + 1)])  # Biases for z_0 to z_L
        # self.bias_last = nn.Parameter(torch.zeros(phi_in))
        # # self.phi_layers = phi_layers
        # # Initialize weights
        # nn.init.xavier_uniform_(self.W)
        # nn.init.xavier_uniform_(self.V)
        
        # self.reset_parameters()

    # def phi_MGN(self, input):
    #     z = torch.matmul(input, self.W) + self.biases[0]  # z_0
    #     for l in range(1, self.phi_n_layers + 1):
    #         z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z)
    #     phi_out = torch.matmul(self.act_fn(z), self.W.t()) + torch.matmul(input, torch.matmul(self.V, self.V.t())) + self.bias_last
    #     return phi_out

    # def reset_parameters(self):
    #     # reset(self.phi_1)
    #     reset(self.phi_2)
    
    
    def forward(self, data):
        u, ksi, eta = data.u, data.ksi, data.eta
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.abs(ksi)
        ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
        # ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
        extension = ksi_plus_eta_norm - ksi_norm
        lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
        n_ksi = ksi.size(1)
        
        phi_NN1_1 = self.phi_MGN(torch.ones_like(lambdaa).reshape(-1,1)).reshape(-1,n_ksi)
        phi_NN1 = self.phi_MGN(lambdaa.reshape(-1,1)).reshape(-1,n_ksi)
        # phi_NN1 = lambdaa-lambdaa**(-3)
        # phi_NN2 = self.phi_2(ksi_norm.reshape(-1,1)).reshape(-1,n_ksi)
        delta = 0.25
        phi_NN2 = torch.exp(-50*ksi_norm**2)*(delta-torch.abs(ksi_norm))
        # phi_NN2 = 2*c*torch.sin(3*np.pi*ksi_norm)
        phi_NN = (phi_NN1 -phi_NN1_1) *phi_NN2
        # phi_NN = phi_NN1 *phi_NN2
        # phi_NN = (phi_NN1 -phi_NN1_1)
        
        # force = torch.mean(kernel * phi * bond_dir, axis = -1)
        h = ksi[0,1]-ksi[0,0]
        # integrand = kernel * (phi-phi_1) * bond_dir
        # integrand = phi_NN * bond_dir
        integrand = phi_NN * bond_dir
        # weights = (torch.tensor([1/2,1,1,1,1,1,1/2])).repeat(ksi.size(0),1).to('cuda')
        # weights = torch.ones((n_ksi,))
        # weights[[0,-1]] = torch.tensor([1/2,1/2])
        # weights = weights.repeat(ksi.size(0),1).to('cuda')
        # force = h*torch.sum(integrand*weights, axis = -1)
        force = h*torch.sum(integrand[:,1:], axis = -1)
        
        # force = unsorted_segment_mean(kernel * phi * bond_dir, row, num_segments=u.size(0))
        
        return force
    

    def nonlinear(self, data):
        u, ksi = data.u, data.ksi
        # row, col = edge_index
        # ksi = x[col] - x[row]
        # eta = u[col] - u[row]
        ndata = u.size(0)
        s, n_ksi = ksi.size()
        eta = torch.zeros((s, n_ksi)).to(u.device)
        m_fact = n_ksi//2
        ksi_range = torch.range(-m_fact, m_fact).int()
        ksi_range = ksi_range[ksi_range != 0]
        for i in range(s):
            eta[i,:] = (u[m_fact+i+ksi_range]-u[m_fact+i]).squeeze()
            
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.abs(ksi)
        ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
        # ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
        extension = ksi_plus_eta_norm - ksi_norm
        lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)

        n_ksi = ksi.size(1)

        phi_NN1_1 = self.phi_MGN(torch.ones_like(lambdaa).reshape(-1,1)).reshape(-1,n_ksi)
        phi_NN1 = self.phi_MGN(lambdaa.reshape(-1,1)).reshape(-1,n_ksi)
        # phi_NN1 = lambdaa-lambdaa**(-3)
        # phi_NN2 = self.phi_2(ksi_norm.reshape(-1,1)).reshape(-1,n_ksi)
        delta = 0.25
        phi_NN2 = torch.exp(-50*ksi_norm**2)*(delta-torch.abs(ksi_norm))
        phi_NN = (phi_NN1 -phi_NN1_1) *phi_NN2
        # phi_NN = phi_NN1 *phi_NN2
        # phi_NN = (phi_NN1 -phi_NN1_1)

        # force = torch.mean(kernel * phi * bond_dir, axis = -1)
        h = ksi[0,1]-ksi[0,0]
        # integrand = kernel * (phi-phi_1) * bond_dir
        # integrand = phi_NN * bond_dir
        integrand = phi_NN * bond_dir
        # weights = (torch.tensor([1/2,1,1,1,1,1,1/2])).repeat(ksi.size(0),1).to('cuda')
        # weights = torch.ones((n_ksi,))
        # weights[[0,-1]] = torch.tensor([1/2,1/2])
        # weights = weights.repeat(ksi.size(0),1).to('cuda')
        # force = h*torch.sum(integrand*weights, axis = -1)
        force = h*torch.sum(integrand[:,1:], axis = -1)

        # force = unsorted_segment_mean(kernel * phi * bond_dir, row, num_segments=u.size(0))

        return force
    
    def linear(self, data):
        u, ksi = data.u, data.ksi
        ndata = u.size(0)
        s, n_ksi = ksi.size()
        eta = torch.zeros((s, n_ksi)).to(u.device)
        m_fact = n_ksi//2
        ksi_range = torch.range(-m_fact, m_fact).int()
        ksi_range = ksi_range[ksi_range != 0]
        for i in range(s):
            eta[i,:] = (u[m_fact+i+ksi_range]-u[m_fact+i]).squeeze()
            
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.abs(ksi)
        ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
        # ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
        # extension = ksi_plus_eta_norm - ksi_norm
        # lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
        lambdaa = 1.0 + ksi*eta / (ksi_norm + 1e-9)**2
        # bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
        bond_dir = ksi / (ksi_norm + 1e-9)
        
        extension = ksi_plus_eta_norm - ksi_norm
        lambdaa1 = 1.0 + extension / (ksi_norm + 1e-9)
        err = lambdaa1 - lambdaa
        max_index = torch.argmax(err)
        max_position = (max_index // err.size(1), max_index % err.size(1))

        n_ksi = ksi.size(1)


        phi_NN1_1 = self.phi_MGN(torch.ones_like(lambdaa).reshape(-1,1)).reshape(-1,n_ksi)
        phi_NN1 = self.phi_MGN(lambdaa.reshape(-1,1)).reshape(-1,n_ksi)
        # phi_NN1 = lambdaa-lambdaa**(-3)
        # phi_NN2 = self.phi_2(ksi_norm.reshape(-1,1)).reshape(-1,n_ksi)
        delta = 0.25
        phi_NN2 = torch.exp(-50*ksi_norm**2)*(delta-torch.abs(ksi_norm))
        phi_NN = (phi_NN1 -phi_NN1_1) *phi_NN2

        # force = torch.mean(kernel * phi * bond_dir, axis = -1)
        h = ksi[0,1]-ksi[0,0]
        integrand = phi_NN * bond_dir
        # weights = (torch.tensor([1/2,1,1,1,1,1,1/2])).repeat(ksi.size(0),1).to('cuda')
        # weights = torch.ones((n_ksi,))
        # weights[[0,-1]] = torch.tensor([1/2,1/2])
        # weights = weights.repeat(ksi.size(0),1).to('cuda')
        # force = h*torch.sum(integrand*weights, axis = -1)
        force = h*torch.sum(integrand[:,1:], axis = -1)

        # force = unsorted_segment_mean(kernel * phi * bond_dir, row, num_segments=u.size(0))

        return force
    

def Integral_w(data):
    u, ksi, eta = data.u, data.ksi, data.eta
    ksi_plus_eta = ksi + eta
    ksi_norm = torch.abs(ksi)
    ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
    # ksi_plus_eta_norm = torch.norm(ksi+eta, dim=1).unsqueeze(1)
    extension = ksi_plus_eta_norm - ksi_norm
    lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
    bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
    
    delta = 0.25
    mu = 0.3846
    c = 2*mu/math.pi/delta**2
    # g_fun = lambda x: np.ones_like(x)
    g_fun = lambda x: x
    dw_exact = (lambdaa-lambdaa**(-3))*2*c/(ksi_norm+1e-9)*g_fun(ksi_norm)
    h = ksi[0,1]-ksi[0,0]
    n_ksi = ksi.size(1)
    integrand = dw_exact * bond_dir
    # weights = (torch.tensor([1/2,1,1,1,1,1,1/2])).repeat(ksi.size(0),1).to('cuda')
    weights = torch.ones((n_ksi,))
    weights[[0,-1]] = torch.tensor([1/2,1/2])
    weights = weights.repeat(ksi.size(0),1).to('cuda')
    force = h*torch.sum(integrand*weights, axis = -1)
    
    return force 
    

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
