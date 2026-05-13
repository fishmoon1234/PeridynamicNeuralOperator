from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import numpy as np


class EGKN(nn.Module):
    def __init__(self, ker_layers, phi_layers, act_fn=nn.ReLU()):
        """
        ker_layers: list[int], e.g., [ker_in, ..., ker_out]
        phi_layers: list[int], e.g., [phi_in, ..., phi_out]
        ang_layers: list[int], e.g., [2, ..., 1]
        """
        super().__init__()
        kernel = MLP(ker_layers, act_fn=act_fn)
        self.egkn_conv = E_GCL_GKN(phi_layers, kernel, act_fn=act_fn)

    def forward(self, data):
        coords = data.x       # [num_nodes, coord_dim]
        u = data.u            # [num_nodes, u_dim]
        edge_index = data.edge_index  # [2, num_edges]
        delta = data.delta    # [num_edges, delta_dim]
        
        force = self.egkn_conv(coords, u, edge_index, delta)
        return force

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
    def __init__(self, phi_layers, kernel, act_fn=nn.ReLU()):
        super().__init__()

        self.kernel = kernel
        self.act_fn = act_fn

        self.phi_mlp = MLP(phi_layers, act_fn=act_fn)


    def node_feat(self, omega_xxp, ksi_plus_eta_norm, ksi_norm, row, x):
        dv = torch.abs(x[1, 1] - x[0, 1]) ** 2
        # dv = torch.abs(x[1, 0] - x[0, 0]) ** 2
        m = unsorted_segment_sum(omega_xxp * ksi_norm ** 2, row, num_segments=x.size(0)) * dv
        h = unsorted_segment_sum(omega_xxp * ksi_norm * (ksi_plus_eta_norm - ksi_norm), row, num_segments=x.size(0)) * dv / m
        return h, m

    def force_integral(self, omega_xxp, omega_xpx, delta, m, h, edge_index, ksi_plus_eta, ksi_plus_eta_norm, ksi_norm, x, u):
        row, col = edge_index
        extension = (ksi_plus_eta_norm - ksi_norm)
        dv = torch.abs(x[1, 1] - x[0, 1]) ** 2
        # dv = torch.abs(x[1, 0] - x[0, 0]) ** 2

        tx_arg = torch.cat([omega_xxp / m[row] * delta ** 4, h[row], extension / delta, ksi_norm / delta], dim=1)
        tx = self.phi_mlp(tx_arg) * (extension / delta)
        txp_arg = torch.cat([omega_xpx / m[col] * delta ** 4, h[col], extension / delta, ksi_norm / delta], dim=1)
        txp = self.phi_mlp(txp_arg) * (extension / delta)
        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)

        force = unsorted_segment_sum((tx + txp) * bond_dir , row, num_segments=x.size(0)) * dv

        return force

    def stress_integral(self, omega_xxp, delta, m, h, edge_index, ksi_plus_eta, ksi_plus_eta_norm, ksi_norm, x):
        row, col = edge_index
        extension = (ksi_plus_eta_norm - ksi_norm)
        dv = torch.abs(x[1, 1] - x[0, 1]) ** 2
        # dv = torch.abs(x[1, 0] - x[0, 0]) ** 2

        tx_arg = torch.cat([omega_xxp / m[row] * delta ** 4, h[row], extension / delta, ksi_norm / delta], dim=1)
        tx = self.phi_mlp(tx_arg) * (extension / delta)

        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)

        ksi = x[col] - x[row]
        # First Piola-Kirchhoff Stress
        PK_tensor = torch.zeros(x.size(0), 2, 2).to(delta.device)
        PK_tensor[:, 0, 0] = (unsorted_segment_sum(tx * bond_dir[:, 0].unsqueeze(1) * ksi[:, 0].unsqueeze(1), row,
                                     num_segments=x.size(0)) * dv).squeeze()
        PK_tensor[:, 0, 1] = (unsorted_segment_sum(tx * bond_dir[:, 0].unsqueeze(1) * ksi[:, 1].unsqueeze(1), row,
                                     num_segments=x.size(0)) * dv).squeeze()
        PK_tensor[:, 1, 0] = (unsorted_segment_sum(tx * bond_dir[:, 1].unsqueeze(1) * ksi[:, 0].unsqueeze(1), row,
                                     num_segments=x.size(0)) * dv).squeeze()
        PK_tensor[:, 1, 1] = (unsorted_segment_sum(tx * bond_dir[:, 1].unsqueeze(1) * ksi[:, 1].unsqueeze(1), row,
                                     num_segments=x.size(0)) * dv).squeeze()

        return PK_tensor

    def forward(self, x, u, edge_index, delta):
        row, col = edge_index
        ksi = x[col] - x[row]
        eta = u[col] - u[row]
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
        ksi_plus_eta_norm = torch.norm(ksi_plus_eta, dim=1).unsqueeze(1)
        
        # enforce radial symmetry of the kernel, as a good basemodel to train heteroPNO:
        mask_0 = torch.abs(ksi[:,0] - 0.0) < 1e-6
        arg_xxp_1 = torch.abs(ksi[:, 0])
        arg_xxp_2 = mask_0 * torch.abs(ksi[:, 1]) + (~mask_0) * torch.sign(ksi[:, 0]) * ksi[:, 1]
        arg_xpx_1 = torch.abs(-ksi[:, 0])
        arg_xpx_2 = mask_0 * torch.abs(-ksi[:, 1]) + (~mask_0) * torch.sign(-ksi[:, 0]) * (-ksi[:, 1])
        
        arg_xxp = torch.cat([arg_xxp_1.unsqueeze(1), arg_xxp_2.unsqueeze(1)], dim=1)
        arg_xpx = torch.cat([arg_xpx_1.unsqueeze(1), arg_xpx_2.unsqueeze(1)], dim=1)
        
        omega_xxp = self.kernel(arg_xxp / delta)
        omega_xpx = self.kernel(arg_xpx / delta)

        h, m = self.node_feat(omega_xxp, ksi_plus_eta_norm, ksi_norm, row, x)
        force = self.force_integral(omega_xxp, omega_xpx, delta, m, h, edge_index, ksi_plus_eta, ksi_plus_eta_norm, ksi_norm, x, u)
        PK_tensor = self.stress_integral(omega_xxp, delta, m, h, edge_index, ksi_plus_eta, ksi_plus_eta_norm, ksi_norm, x)
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
