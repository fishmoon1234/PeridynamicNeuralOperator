from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import math
from utilities_INO_PD import *
import torch.nn.functional as F 

torch.manual_seed(1)
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

        # self.bias = nn.ParameterList([nn.Parameter(self.init_b(embed_dim, embed_dim), requires_grad=True) for i in range(num_layers+1)])
        # self.bias[0] = nn.Parameter(self.init_b(embed_dim, in_dim), requires_grad=True)
        # self.bias[-1] = nn.Parameter(self.init_b(in_dim, embed_dim), requires_grad=True)
        
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

    def forward(self, x):
        z = F.softplus(self.beta[0]).view(1,-1) * F.linear(x, self.W, self.bias[0])
        for i in range(self.num_layers):
        # for i in range(self.current_layers - 1):
            skip = F.softplus(self.beta[i+1]).view(1,-1) * F.linear(x, self.W, self.bias[i+1])
            # z = skip + F.softplus(self.alpha[i]).view(1,-1) * self.nonlinearity[i](z)
            z += skip + F.softplus(self.alpha[i]).view(1,-1) * self.nonlinearity[i](z)

        z = F.softplus(self.alpha[-1]).view(1,-1) * self.nonlinearity[-1](z)
        z = F.linear(z, self.W.T, self.bias[-1])

        return z
 
class GNM_Module(nn.Module):
    def __init__(self, in_dim, embed_dim, activation):
        super().__init__()

        self.beta = nn.Parameter(torch.rand(1), requires_grad=True)
        self.W = nn.Parameter(self.init_W(embed_dim, in_dim), requires_grad=True)
        # self.b = nn.Parameter(self.init_b(embed_dim, in_dim), requires_grad=True)
        self.b = nn.Parameter(self.init_b(embed_dim, ), requires_grad=True)
        self.act = activation()
        
    def init_W(self, embed_dim, in_dim):
        W = torch.empty(embed_dim, in_dim)
        nn.init.xavier_uniform_(W)
        return W

    def init_b(self, *shape):
        b = torch.empty(*shape)
        nn.init.xavier_uniform_(b.unsqueeze(0))
        return b

    def forward(self, x):

        z = F.linear(x, weight = self.W, bias=self.b)
        z = self.act(z * F.softplus(self.beta))
        z = F.linear(z, weight=self.W.T)

        return z
    
 
class mGradNet_M(nn.Module):
    def __init__(self, num_modules, current_layers, in_dim, embed_dim, activation):
        super().__init__()

        self.num_modules = num_modules
        self.current_layers = current_layers
        self.mmgn_modules = nn.ModuleList([GNM_Module(in_dim, embed_dim, activation) for i in range(num_modules)])
        self.alpha = nn.Parameter(torch.rand(num_modules,), requires_grad=True)
        # self.bias = nn.Parameter(GNM_Module.init_b(in_dim, embed_dim), requires_grad=True)
        self.bias = nn.Parameter(self.init_b(in_dim, ), requires_grad=True)
        
    def init_W(self, embed_dim, in_dim):
        W = torch.empty(embed_dim, in_dim)
        nn.init.xavier_uniform_(W)
        return W
    
    def init_b(self, *shape):
        b = torch.empty(*shape)
        nn.init.xavier_uniform_(b.unsqueeze(0))
        return b
    
    def add_layer(self):
        if self.current_layers < self.num_modules:
            self.current_layers += 1

    def forward(self, x):

        z = 0
        for i in range(self.current_layers):
            out = self.mmgn_modules[i](x)
            z += F.softplus(self.alpha[i]) * out 
        z += self.bias
        return z
    
       
class DenseNet(nn.Module):
    def __init__(self, phi_1_layer, act_fn=nn.ReLU()):
        super().__init__()

        self.act_fn = act_fn
        self.phi_n_layers = len(phi_1_layer) - 2  # Calculate number of hidden layers
        self.current_layers = self.phi_n_layers  # Number of currently active layers

        phi_in = phi_1_layer[0]
        phi_width = phi_1_layer[1]
        phi_out = phi_1_layer[-1]

        # List of Linear layers for each intermediate layer
        self.layers = nn.ModuleList()

        # Initialize the first linear layer
        self.layers.append(nn.Linear(phi_in, phi_width))
        # Initialize intermediate layers (if any)
        for _ in range(self.phi_n_layers - 1):
            self.layers.append(nn.Linear(phi_width, phi_width))

        # Initialize the last linear layer
        self.layers.append(nn.Linear(phi_width, phi_out))

        # Bias term for the last layer
        self.bias_last = nn.Parameter(torch.zeros(phi_out))

    def add_layer(self):
        """Dynamically add a new layer, keep all previous layers trainable."""
        if self.current_layers < self.phi_n_layers:
            # self.layers.insert(self.current_layers, nn.Linear(self.layers[self.current_layers - 1].out_features, self.layers[0].in_features))
            self.current_layers += 1
            print(f"Added layer {self.current_layers}. All layers are trainable.")


    def forward(self, input):
        # Forward pass through all active layers
        z = input
        for l in range(self.current_layers):
            z = self.act_fn(self.layers[l](z))  # Apply linear transf ormation + nonlinearity

        # Compute the final output
        # phi_out = self.layers[self.current_layers - 1](z) + self.bias_last
        phi_out = self.layers[self.phi_n_layers](z) + self.bias_last
        # phi_out = self.layers[self.phi_n_layers](z)
        return phi_out
    
    
class MGN(nn.Module):
    def __init__(self, phi_1_layer, current_layer, act_fn=nn.Softplus()):
        super().__init__()
            
        self.act_fn = act_fn
        self.phi_n_layers = len(phi_1_layer) - 3
        self.current_layers = current_layer  # Number of currently active layers

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
            z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z) + z
        
        # Compute the final output
        y = (
            torch.matmul(self.act_fn(z), self.W.t()) +
            torch.matmul(input, torch.matmul(self.V, self.V.t())) +
            self.bias_last
        )
        return y
    
class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, phi_1_layer, phi_2_layer, act_fun_xi, act_fn, normalize=False, coords_agg='mean', residual=True):
        super().__init__()

        # self.k1k2w = k1k2w
        self.act_fun_xi = act_fun_xi
        self.act_fn = act_fn
        self.phi_MGN = DenseNet(phi_1_layer,  act_fn)
        # self.phi_2 = DenseNet(phi_2_layer, torch.nn.Tanh)
        # self.phi_MGN = MGN(phi_1_layer, current_layer, act_fn)
        # self.phi_MGN = mGradNet_M(num_modules=4, in_dim=1, embed_dim=7, activation=lambda : nn.Softmax(dim=-1))
        # self.phi_MGN = mGradNet_M(num_modules, current_layers, 1, embed_dim, activation=lambda : nn.Softmax(dim=-1))
        # self.phi_MGN = mGradNet_C(num_layers, 1, embed_dim, self.act_fn)
        self.phi_2 = DenseNet(phi_2_layer, self.act_fun_xi)
        
        
        self.reset_parameters()

    # def phi_MGN(self, input):
    #     z = torch.matmul(input, self.W) + self.biases[0]  # z_0
    #     for l in range(1, self.phi_n_layers + 1):
    #         z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z)
    #     phi_out = torch.matmul(self.act_fn(z), self.W.t()) + torch.matmul(input, torch.matmul(self.V, self.V.t())) + self.bias_last
    #     return phi_out

    def reset_parameters(self):
        # reset(self.phi_1)
        reset(self.phi_2)
    
    def forward(self, data):
        x, u, edge_index, delta = data.x, data.u, data.edge_index, data.delta
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
        dx = torch.abs(x[1, 1] - x[1,0]) 

        # kernel = 1.0/(ksi_norm + 1e-9)
        
        # k1k2 = self.k1k2w[:, :2]
        # matches = (torch.round(ksi.unsqueeze(1)*1000) == torch.round(k1k2*1000)).all(dim=2)
        # matching_indices = matches.nonzero(as_tuple=True)[1]
        # wij = self.k1k2w[matching_indices, 2]
        # wij = wij.repeat(2,1).permute(1,0)
        
        # phi_NN =  self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1)
        # phi_NN = self.phi_MGN(lambdaa/delta)* self.phi_2(ksi/delta)
        # phi_NN = self.phi_MGN(lambdaa)* self.phi_2(ksi)
        # phi_NN = self.phi_MGN(ksi_plus_eta_norm)* self.phi_2(ksi)
        # phi_NN = (self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1))* self.phi_2(ksi_norm)
        # phi_NN = (self.phi_1(lambdaa)-self.phi_1(lambdaa_1))* self.phi_2(ksi)
        g_NN = (self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1))
        k_NN = self.phi_2(ksi/delta)
        phi_NN = g_NN*k_NN
        force = dx**2*unsorted_segment_sum(phi_NN * bond_dir, row, num_segments=x.size(0))
        
        rms_g = torch.sqrt((g_NN**2).mean())
        rms_k = torch.sqrt((k_NN**2).mean())
        L_match = (torch.log(rms_g) - torch.log(rms_k))**2
        
        return force, L_match
    
    
    def linear(self, data):
        x, u, edge_index, delta = data.x, data.u, data.edge_index, data.delta
        row, col = edge_index
        ksi = x[col] - x[row]
        eta = u[col] - u[row]
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
        ksi_plus_eta_norm = torch.norm(ksi_plus_eta, dim=1).unsqueeze(1)
        extension = ksi_plus_eta_norm - ksi_norm
        
        # lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
        # bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
        lambdaa = 1.0 + torch.sum(ksi*eta, dim=-1).view(-1,1) / (ksi_norm + 1e-9)**2
        bond_dir = ksi / (ksi_norm + 1e-9)
        
        lambdaa_1 = torch.ones_like(lambdaa)
        dx = torch.abs(x[1, 1] - x[1,0]) 
        
        phi_NN = (self.phi_MGN(lambdaa)-self.phi_MGN(lambdaa_1))* self.phi_2(ksi)
        force = dx**2*unsorted_segment_sum(phi_NN * bond_dir, row, num_segments=x.size(0))
        
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
