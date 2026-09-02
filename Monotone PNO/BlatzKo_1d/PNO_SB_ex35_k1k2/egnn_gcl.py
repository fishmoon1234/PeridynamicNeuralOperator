from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import numpy as np


# class EGKN(nn.Module):
#     def __init__(self, ker_layers, phi_layers, act_fn=nn.ReLU()):
#         """
#         ker_layers: list[int], e.g., [ker_in, ..., ker_out]
#         phi_layers: list[int], e.g., [phi_in, ..., phi_out]
#         ang_layers: list[int], e.g., [2, ..., 1]
#         """
#         super().__init__()
#         kernel = MLP(ker_layers, act_fn=act_fn)
#         self.egkn_conv = E_GCL_GKN(phi_layers, kernel, act_fn=act_fn)

#     def forward(self, data):
#         # coords = data.x       # [num_nodes, coord_dim]
#         # u = data.u            # [num_nodes, u_dim]
#         # edge_index = data.edge_index  # [2, num_edges]
#         # delta = data.delta    # [num_edges, delta_dim]
#         ksi, eta = data.ksi, data.eta
        
#         force = self.egkn_conv(ksi, eta)
#         return force

class MLP(nn.Module):
    def __init__(self, layer_sizes, act_fn=nn.ReLU()):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_fn)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class E_GCL_GKN(nn.Module):
    def __init__(self, phi_layers, kernel_layers, act_fn=nn.ReLU()):
        super().__init__()

        self.kernel =  MLP(kernel_layers, act_fn=act_fn)
        self.act_fn = act_fn

        self.phi_mlp = MLP(phi_layers, act_fn=act_fn)

    def forward(self, data):
        ksi, eta = data.ksi, data.eta
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.abs(ksi)
        ksi_plus_eta_norm = torch.abs(ksi_plus_eta)
        extension = (ksi_plus_eta_norm - ksi_norm)
        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
        delta = 0.25
        dx = ksi[0,1]-ksi[0,0]
        n_ksi = ksi.size(1)
        
        omega = self.kernel(ksi.reshape(-1,1) / delta).reshape(-1,n_ksi)
        
        numer = (omega*extension*ksi_norm).reshape(-1,n_ksi)
        denom = (omega*ksi_norm**2).reshape(-1,n_ksi)
        dilatation = torch.sum(numer[:,1:], axis = -1)/(torch.sum(denom[:,1:], axis = -1)+1e-8)
        # Expand dilatation from [2570] to [2570, n_ksi] by repeating n_ksi times
        dilatation = dilatation.unsqueeze(1).expand(-1, n_ksi)

        # h, m = self.node_feat(omega, ksi_plus_eta_norm, ksi_norm, row, x)
        t_input = torch.cat([omega.reshape(-1,1), dilatation.reshape(-1,1), extension.reshape(-1,1) / delta, ksi_norm.reshape(-1,1) / delta], dim=1)
        integrand = self.phi_mlp(t_input).reshape(-1,n_ksi)*bond_dir
        force = dx*torch.sum(integrand[:,1:], axis = -1)
        
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