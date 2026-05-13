from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import math
from utilities_INO_PD import *

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, phi_1_layer, act_fun_xi, act_fn=nn.Softplus(), normalize=False, coords_agg='mean', residual=True):
        super().__init__()

        # self.k1k2w = k1k2w
        self.act_fun_xi = act_fun_xi
        self.act_fn = act_fn
        # self.phi_1 = DenseNet(phi_1_layer, torch.nn.ReLU)
        # self.phi_2 = DenseNet(phi_2_layer, torch.nn.Tanh)
        # self.phi_2 = DenseNet(phi_2_layer, self.act_fun_xi)

        self.phi_n_layers = len(phi_1_layer) - 1
        phi_in = phi_1_layer[0]
        phi_width = phi_1_layer[1]
        self.W = nn.Parameter(torch.empty(phi_in, phi_width))  # Weight matrix
        self.V = nn.Parameter(torch.empty(phi_in, phi_width))  # Second weight matrix
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(phi_width)) for _ in range(self.phi_n_layers + 1)])  # Biases for z_0 to z_L
        self.bias_last = nn.Parameter(torch.zeros(phi_in))
        # self.phi_layers = phi_layers
        # Initialize weights
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)
        

    def phi_MGN(self, input):
        z = torch.matmul(input, self.W) + self.biases[0]  # z_0
        for l in range(1, self.phi_n_layers + 1):
            z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z)
        phi_out = torch.matmul(self.act_fn(z), self.W.t()) + torch.matmul(input, torch.matmul(self.V, self.V.t())) + self.bias_last
        return phi_out
    
    def forward(self, data):
        x, u, edge_index, dx = data.x, data.u, data.edge_index, data.dx
        row, col = edge_index
        ksi = x[col] - x[row]
        eta = u[col] - u[row]
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
        ksi_plus_eta_norm = torch.norm(ksi_plus_eta, dim=1).unsqueeze(1)
        extension = ksi_plus_eta_norm - ksi_norm
        lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
        lambdaa_1 = torch.ones_like(lambdaa)
        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
        
        delta = 0.25
        mu = 0.3846
        c = 2*mu/math.pi/delta**2
        # g_fun = lambda x: x-x**(-3)
        k_fun = lambda x: 2*c*torch.exp(-50*x**2)*(delta-torch.abs(x))

        # kernel = 1.0/(ksi_norm + 1e-9)
        
        # k1k2 = self.k1k2w[:, :2]
        # matches = (torch.round(ksi.unsqueeze(1)*1000) == torch.round(k1k2*1000)).all(dim=2)
        # matching_indices = matches.nonzero(as_tuple=True)[1]
        # wij = self.k1k2w[matching_indices, 2]
        # wij = wij.repeat(2,1).permute(1,0)
        
        # phi_NN =  self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1)
        # phi_NN = (self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1))* self.phi_2(ksi_norm)
        # phi_NN = (self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1))* k_fun(ksi_norm)
        # phi_NN = g_fun(lambdaa)* self.phi_2(ksi_norm)
        phi_NN = (self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1))* k_fun(ksi_norm)
        force = dx[0]**2*unsorted_segment_sum(phi_NN * bond_dir, row, num_segments=x.size(0))
        # force = unsorted_segment_sum(kernel* phi_NN * bond_dir * wij, row, num_segments=x.size(0))
        
        return force
    

    
class MGN(torch.nn.Module):
    def __init__(self, phi_layer, act_fn=nn.Softplus(), normalize=False, coords_agg='mean', residual=True):
        super(MGN, self).__init__()

        self.act_fn = act_fn
        #layer = nn.Linear(phi_width, phi_out)
        #torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_n_layers = len(phi_layer) - 1
        phi_in = phi_layer[0]
        phi_width = phi_layer[1]
        self.W = nn.Parameter(torch.empty(phi_in, phi_width))  # Weight matrix
        self.V = nn.Parameter(torch.empty(phi_in, phi_width))  # Second weight matrix
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(phi_width)) for _ in range(self.phi_n_layers + 1)])  # Biases for z_0 to z_L
        self.bias_last = nn.Parameter(torch.zeros(phi_in))
        # self.phi_layers = phi_layers
        # Initialize weights
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V)


    def phi_MGN(self, input):
        z = torch.matmul(input, self.W) + self.biases[0]  # z_0
        for l in range(1, self.phi_n_layers + 1):
            z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z)
        phi_out = torch.matmul(self.act_fn(z), self.W.t()) + torch.matmul(input, torch.matmul(self.V, self.V.t())) + self.bias_last
        return phi_out
    

    def forward(self, x):
        y = self.phi_MGN(x)
        return y


# class EGKN(torch.nn.Module):
#     def __init__(self, phi_layer, act_fn=nn.Softplus()):
#         super().__init__()

#         # kernel = DenseNet([ker_in, ker_width//2, ker_width, ker_out], torch.nn.ReLU)
#         # kernel_NN = DenseNet(kernel_layer, torch.nn.ReLU)
#         self.egkn_conv = E_GCL_GKN(phi_layer, act_fn=act_fn)

#     def forward(self, data):
#         # coords, u, edge_index, delta = data.x, data.u, data.edge_index, data.delta
#         # force = self.egkn_conv(coords, u, edge_index, delta)
#         u, ksi, eta = data.u, data.ksi, data.eta
#         force = self.egkn_conv(u, ksi, eta)

#         return force
    
class EGKN(nn.Module):
# class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, phi_layer, act_fn=nn.Softplus(), normalize=False, coords_agg='mean', residual=True):
        super().__init__()

        # self.act_fn = act_fn
        # self.phi_NN = DenseNet(phi_layer, torch.nn.ReLU)
        self.phi_MGN = MGN(phi_layer, nn.Softplus())
    
    def reset_parameters(self):
        # reset(self.phi_NN)
        reset(self.phi_MGN)

    # def forward(self, u, ksi, eta):
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
        
        # fix kernel
        kernel = (ksi_norm + 1e-9)
        
        # not fix kernel
        # kernel = self.kernel_NN(ksi_norm.reshape(-1,1)).reshape(-1,n_ksi)
        
        phi = self.phi_MGN(lambdaa.reshape(-1,1)).reshape(-1,n_ksi)
        phi_1 = self.phi_MGN(torch.ones_like(lambdaa).reshape(-1,1)).reshape(-1,n_ksi)
        # phi = self.phi_NN(lambdaa.reshape(-1,1)).reshape(-1,n_ksi)
        # phi_1 = self.phi_NN(torch.ones_like(lambdaa).reshape(-1,1)).reshape(-1,n_ksi)
        
        # force = torch.mean(kernel * phi * bond_dir, axis = -1)
        h = ksi[0,1]-ksi[0,0]
        integrand = kernel * (phi-phi_1) * bond_dir
        # integrand = (phi-phi_1) * bond_dir
        # integrand = phi * bond_dir
        # weights = (torch.tensor([1/2,1,1,1,1,1,1/2])).repeat(ksi.size(0),1).to('cuda')
        weights = torch.ones((n_ksi,))
        weights[[0,-1]] = torch.tensor([1/2,1/2])
        weights = weights.repeat(ksi.size(0),1).to('cuda')
        force = h*torch.sum(integrand*weights, axis = -1)
        # force = h*torch.sum(integrand, axis = -1)
        
        # force = unsorted_segment_mean(kernel * phi * bond_dir, row, num_segments=u.size(0))
        
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
