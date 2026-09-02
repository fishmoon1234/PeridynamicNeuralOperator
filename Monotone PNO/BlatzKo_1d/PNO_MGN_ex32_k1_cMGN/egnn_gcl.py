from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import math
from utilities_INO_PD import *
import torch.nn.functional as F

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class mGradNet_C(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, activation):
        super().__init__()
       
        # self.current_layers = layer_start
        self.num_layers = num_layers
        self.nonlinearity = nn.ModuleList([activation for i in range(num_layers+1)])
        # self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)
        self.W = nn.Parameter(self.init_W(embed_dim, in_dim), requires_grad=True)
        
        self.bias = nn.ParameterList([nn.Parameter(self.init_b(embed_dim, ), requires_grad=True) for i in range(num_layers+2)])
        self.bias[0] = nn.Parameter(self.init_b(embed_dim, ), requires_grad=True)
        self.bias[-1] = nn.Parameter(self.init_b(in_dim, ), requires_grad=True)

        self.beta = nn.ParameterList([nn.Parameter(torch.rand(embed_dim,), requires_grad=True) for i in range(num_layers+1)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.rand(embed_dim,), requires_grad=True) for i in range(num_layers+1)])
    
    def init_W(self, embed_dim, in_dim):
        W = torch.empty(embed_dim, in_dim)
        nn.init.xavier_uniform_(W)
        return W
    
    def init_b(self, *shape):
        b = torch.empty(*shape)
        nn.init.xavier_uniform_(b.unsqueeze(0))
        return b
    
    # def add_layer(self):
    #     if self.current_layers < self.num_layers:
    #         self.current_layers += 1

    def forward(self, x):
        z = F.softplus(self.beta[0]).view(1,-1) * F.linear(x, self.W, self.bias[0])
        for i in range(self.num_layers):
        # for i in range(self.current_layers - 1):
            skip = F.softplus(self.beta[i+1]).view(1,-1) * F.linear(x, self.W, self.bias[i+1])
            z = skip + F.softplus(self.alpha[i]).view(1,-1) * self.nonlinearity[i](z)

        z = F.softplus(self.alpha[-1]).view(1,-1) * self.nonlinearity[-1](z)
        z = F.linear(z, self.W.T, self.bias[-1])

        return z
    
    
class mGradNet_C_z(nn.Module):
    def __init__(self, num_layers, layer_start, in_dim, embed_dim, activation):
        super().__init__()
        
        self.current_layers = layer_start
        self.num_layers = num_layers
        self.nonlinearity = activation
        
        self.W = nn.Parameter(self.init_W(embed_dim, in_dim), requires_grad=True)
        self.bias_shared = nn.Parameter(self.init_b(embed_dim), requires_grad=True)
        self.beta_shared = nn.Parameter(torch.rand(embed_dim), requires_grad=True)
        self.alpha_shared = nn.Parameter(torch.rand(embed_dim), requires_grad=True)
        
        self.beta_0 = nn.Parameter(torch.rand(embed_dim), requires_grad=True)
        self.alpha_out = nn.Parameter(torch.rand(embed_dim), requires_grad=True)
        self.bias_0 = nn.Parameter(self.init_b(embed_dim), requires_grad=True)
        self.bias_out = nn.Parameter(self.init_b(in_dim), requires_grad=True)
        
    def init_W(self, embed_dim, in_dim):
        W = torch.empty(embed_dim, in_dim)
        nn.init.xavier_uniform_(W)
        return W
    
    def init_b(self, dim):
        b = torch.empty(dim)
        nn.init.xavier_uniform_(b.unsqueeze(0))
        return b
    
    def add_layer(self, add_num_layers):
        if self.current_layers + add_num_layers <= self.num_layers:
            self.current_layers += add_num_layers
            # self.beta_shared.data /= 2
            # self.alpha_shared.data /= 2
            # self.bias_shared.data /= 2
    
    def forward(self, x):
        coef = 1. / self.current_layers
        z = F.softplus(self.beta_0).view(1, -1) * F.linear(x, self.W, self.bias_0)
        
        for _ in range(self.current_layers):
            skip = F.softplus(self.beta_shared).view(1, -1) * F.linear(x, self.W, self.bias_shared)
            z = z + coef * (skip + F.softplus(self.alpha_shared).view(1, -1) * self.nonlinearity(z))
        
        z = F.softplus(self.alpha_out).view(1, -1) * self.nonlinearity(z)
        z = F.linear(z, self.W.T, self.bias_out)
        
        return z

class MGN(nn.Module):
    def __init__(self, phi_1_layer, current_layers=1, act_fn=nn.Softplus()):
        super().__init__()
            
        self.act_fn = act_fn
        # self.phi_n_layers = len(phi_1_layer) - 2
        self.phi_n_layers = len(phi_1_layer) - 3
        self.current_layers = current_layers  # Number of currently active layers

        phi_in = phi_1_layer[0]
        phi_width = phi_1_layer[1]
        phi_out = phi_1_layer[-1]

        # Initialize weight matrices
        self.W = nn.Parameter(torch.empty(phi_in, phi_width))
        self.V = nn.Parameter(torch.empty(phi_in, phi_width))

        # ParameterList to store biases
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(phi_width)) for _ in range(self.phi_n_layers + 1)]
        )
        self.bias_last = nn.Parameter(torch.zeros(phi_out))

        # Xavier initialization
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
    
    def add_layer(self):
        """Dynamically add a new layer, keep all previous layers trainable."""
        if self.current_layers < self.phi_n_layers:
            self.current_layers += 1
            print(f"Added layer {self.current_layers}. All layers are trainable.")

    def forward(self, input):
        # Initial hidden layer computation (z_0)
        z = torch.matmul(input, self.W) + self.biases[0]
        
        # Forward pass through all active layers
        for l in range(1, self.current_layers + 1):
            # z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z)
            z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z) +z
        
        # Compute the final output
        phi_out = (
            torch.matmul(self.act_fn(z), self.W.t()) +
            torch.matmul(input, torch.matmul(self.V, self.V.t())) +
            self.bias_last
        )
        return phi_out
    
    
class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, num_layers, in_dim, embed_dim, act_fn=nn.Sigmoid(), normalize=False,
                 coords_agg='mean', residual=True):
        super().__init__()

        self.act_fn = act_fn
        # self.phi_MGN = MGN(phi_1_layer, current_layers, act_fn)
        # self.phi_MGN = mGradNet_C_z(num_layers, current_layers, in_dim, embed_dim, self.act_fn)
        self.phi_MGN = mGradNet_C(num_layers, in_dim, embed_dim, self.act_fn)
    
    
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
        mu = 0.3846
        c = 2*mu/math.pi/delta**2
        phi_NN2 = 2*c*torch.exp(-50*ksi_norm**2)*(delta-torch.abs(ksi_norm))
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
        mu = 0.3846
        c = 2*mu/math.pi/delta**2
        phi_NN2 = 2*c*torch.exp(-50*ksi_norm**2)*(delta-torch.abs(ksi_norm))
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

        n_ksi = ksi.size(1)


        phi_NN1_1 = self.phi_MGN(torch.ones_like(lambdaa).reshape(-1,1)).reshape(-1,n_ksi)
        phi_NN1 = self.phi_MGN(lambdaa.reshape(-1,1)).reshape(-1,n_ksi)
        # phi_NN1 = lambdaa-lambdaa**(-3)
        # phi_NN2 = self.phi_2(ksi_norm.reshape(-1,1)).reshape(-1,n_ksi)
        delta = 0.25
        mu = 0.3846
        c = 2*mu/math.pi/delta**2
        phi_NN2 = 2*c*torch.exp(-50*ksi_norm**2)*(delta-torch.abs(ksi_norm))
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
