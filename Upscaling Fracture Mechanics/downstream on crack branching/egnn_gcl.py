from torch import nn
import torch
from torch_geometric.nn.inits import reset, uniform
import math
import numpy as np
from utilities_INO_PD import *
from quad_rule_2d import quadweights_2d

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class LinearlyControlledNet(nn.Module):
    """
    Linearly controlled network for learning g function.
    Combines a bounded nonlinear part with a linear trend.
    """
    def __init__(self, g_layer, act_fun):
        super().__init__()
        # Nonlinear part (Bounded)
        self.nonlinear_net = nn.Sequential(DenseNet(g_layer, act_fun), act_fun())
        self.scale = nn.Parameter(torch.tensor(1.0))  # Allow the nonlinear part to scale
        
        # Linear part (Linear Trend)
        self.linear_trend = nn.Linear(1, 1) 

    def forward(self, x):
        # g(x) = Scale * Tanh(Net(x)) + (wx + b)
        bounded_part = self.scale * self.nonlinear_net(x)
        linear_part = self.linear_trend(x)
        return bounded_part + linear_part


class E_GCL_GKN(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    """

    def __init__(self, phi_2_layer, g_layer, act_fun_xi, act_fn=nn.Softplus(), normalize=False, coords_agg='mean', residual=True, init_alpha=1.0, validate_quadrature=False, use_singular=True, integration='quadrature'):
        super().__init__()

        # self.k1k2w = k1k2w
        self.act_fun_xi = act_fun_xi
        self.act_fn = act_fn
        self.use_singular = use_singular  # False: phi = g*k (no singular); True: phi = g*k*|xi|^{-alpha}
        if integration not in ('quadrature', 'riemann'):
            raise ValueError("integration must be 'quadrature' or 'riemann', got " + str(integration))
        self.integration = integration
        # self.phi_1 = DenseNet(phi_1_layer, torch.nn.ReLU)
        # self.phi_2 = DenseNet(phi_2_layer, torch.nn.Tanh)
        self.k = DenseNet(phi_2_layer, self.act_fun_xi)
        # Keep module name g_MGN for compatibility with plotting and checkpoints.
        # g(lambda)-g(1) is enforced to be sign-consistent by construction:
        # sign(lambda-1) * positive_magnitude(|lambda-1|).
        self.g_MGN = DenseNet(g_layer, self.act_fun_xi)
        # alpha is constrained to be strictly positive through softplus reparameterization.
        self.alpha_raw = nn.Parameter(torch.tensor(float(init_alpha)))
        self._alpha_min = 1e-6
        self.softplus = nn.Softplus()
        self.validate_quadrature = validate_quadrature
        self._validation_done = False 

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
        

    def phi_MGN(self, input):
        z = torch.matmul(input, self.W) + self.biases[0]  # z_0
        for l in range(1, self.phi_n_layers + 1):
            z = torch.matmul(input, self.W) + self.biases[l] + self.act_fn(z)
        phi_out = torch.matmul(self.act_fn(z), self.W.t()) + torch.matmul(input, torch.matmul(self.V, self.V.t())) + self.bias_last
        return phi_out
    
    
    def get_alpha(self):
        return self.softplus(self.alpha_raw) + self._alpha_min

    def signed_g_difference(self, lambdaa):
        delta_lambda = lambdaa.reshape(-1, 1) - 1.0
        sign = torch.sign(delta_lambda)
        # Use delta_lambda directly (not abs) so the network can learn
        # different magnitudes for tension (λ>1) vs compression (λ<1).
        magnitude = self.softplus(self.g_MGN(delta_lambda))
        return sign * magnitude

    def g_sign_violation_rate(self, lambda_vals):
        g_vals = self.signed_g_difference(lambda_vals)
        below_mask = lambda_vals < 1.0
        above_mask = lambda_vals > 1.0
        violation = torch.zeros_like(g_vals, dtype=torch.bool)
        if below_mask.any():
            violation = violation | (below_mask & (g_vals >= 0.0))
        if above_mask.any():
            violation = violation | (above_mask & (g_vals <= 0.0))
        active = below_mask | above_mask
        if active.any():
            return violation[active].float().mean()
        return torch.tensor(0.0, device=lambda_vals.device)

    def forward(self, data, quad_points_weights, edge_weight_indices):
        x, u, edge_index, dx = data.x, data.u, data.edge_index, data.dx
        row, col = edge_index
        ksi = x[col] - x[row]
        eta = u[col] - u[row]
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.norm(ksi, dim=1).unsqueeze(1)
        ksi_plus_eta_norm = torch.norm(ksi_plus_eta, dim=1).unsqueeze(1)
        extension = ksi_plus_eta_norm - ksi_norm
        lambdaa = 1.0 + extension / (ksi_norm + 1e-9)
        bond_dir = ksi_plus_eta / (ksi_plus_eta_norm + 1e-9)
        delta = data.delta
        delta_val = delta[0] if torch.is_tensor(delta) else delta
        
        # Hard-constraint for g(lambda)-g(1):
        # lambda<1 -> negative, lambda>1 -> positive by construction.
        g_NN = self.signed_g_difference(lambdaa).reshape(-1, 1)
        # k(ksi): input is 2D ksi [n_edges, 2], normalized by delta
        ksi_2d = ksi / (delta_val + 1e-12)
        k_NN = self.k(ksi_2d)
        if self.use_singular:
            alpha_eff = self.get_alpha()
            phi_NN = g_NN * k_NN * (ksi_norm / (delta_val + 1e-12))**(-alpha_eff)
        else:
            phi_NN = g_NN * k_NN  # no singular term: do not multiply by |xi|^{-alpha}
        
        # Integral weights.
        # 'quadrature': moment-matching weights from quadweights_2d (singular kernel-aware).
        #   int f*r^{-alpha} dxi ~ sum_j f_j * r_j^{-alpha} * w_j ; phi_NN already carries r^{-alpha}.
        # 'riemann': uniform dx**2 weights. int f dxi ~ sum_j f_j * dx**2 ; phi_NN still carries
        #   r^{-alpha} when use_singular=True, so this is the naive uniform discretization of the
        #   same singular integral (used here as a baseline against quadrature).
        if self.integration == 'quadrature':
            quad_weights = quad_points_weights[:, 2].to(x.device)
            edge_weights = quad_weights[edge_weight_indices.to(x.device)]
            weighted_integrand = phi_NN * bond_dir * edge_weights.unsqueeze(1)  # [n_edges, 2]
        else:  # riemann
            dx_scalar = dx[0] if torch.is_tensor(dx) else dx
            weighted_integrand = phi_NN * bond_dir * (dx_scalar ** 2)  # [n_edges, 2]
        force = unsorted_segment_sum(weighted_integrand, row, num_segments=x.size(0))
        
        # L_match: align scale of g and k (same as PNO_uniqueness_architecture_new_f_mix)
        rms_g = torch.sqrt((g_NN**2).mean() + 1e-12)
        rms_k = torch.sqrt((k_NN**2).mean() + 1e-12)
        L_match = (torch.log(rms_g) - torch.log(rms_k))**2
        
        # Compare with Riemann sum for validation (only if enabled and not done yet)
        if self.validate_quadrature and not self._validation_done:
            # Riemann sum: force_riemann = dx^2 * sum(phi_NN * bond_dir)
            integrand_riemann = phi_NN * bond_dir  # [n_edges, 2]
            force_riemann = dx[0]**2 * unsorted_segment_sum(integrand_riemann, row, num_segments=x.size(0))
            
            # Compute error statistics
            error = torch.norm(force - force_riemann, dim=1)
            rel_error = error / (torch.norm(force_riemann, dim=1) + 1e-10)
            max_error = torch.max(error).item()
            mean_error = torch.mean(error).item()
            max_rel_error = torch.max(rel_error).item()
            mean_rel_error = torch.mean(rel_error).item()
            
            # Also compute per-node statistics
            abs_diff = torch.abs(force - force_riemann)
            max_abs_diff = torch.max(abs_diff).item()
            mean_abs_diff = torch.mean(abs_diff).item()
            
            print(f"\n{'='*60}")
            print(f"Quadrature Rule Validation (vs Riemann Sum)")
            print(f"{'='*60}")
            print(f"Force shape: {force.shape}")
            print(f"Absolute Error Statistics:")
            print(f"  Max absolute error: {max_error:.6e}")
            print(f"  Mean absolute error: {mean_error:.6e}")
            print(f"  Max per-component error: {max_abs_diff:.6e}")
            print(f"  Mean per-component error: {mean_abs_diff:.6e}")
            print(f"Relative Error Statistics:")
            print(f"  Max relative error: {max_rel_error:.6e}")
            print(f"  Mean relative error: {mean_rel_error:.6e}")
            print(f"{'='*60}\n")
            
            self._validation_done = True
        
        return force, L_match
    

    
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
