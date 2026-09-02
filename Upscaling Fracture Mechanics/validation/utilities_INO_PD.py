import torch
import numpy as np
import scipy.io
import csv
import json
import os
import re
# import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import math
import matplotlib.pyplot as plt
from quad_rule_2d import quad_rule_2d, quadweights_2d

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)


def get_best_init_ckpt_from_record(record_path, results_base_dir, ntrain, error_index=1):
    """
    Read training_record.txt, find the entry with the smallest error (by default
    the second value in losss, i.e. validation error), and return the path to the
    corresponding model.ckpt in Results if it exists.

    record_path: path to training_record.txt
    results_base_dir: directory containing Results/ (e.g. current_dir)
    ntrain: ntrain used to build the Results subdir name (record does not store ntrain)
    error_index: which loss to use for ranking (0=train, 1=valid, 2=test). Default 1 (validation).

    Returns: path to model.ckpt or None if not found.
    """
    if not os.path.isfile(record_path):
        return None
    # Format: Data:xxx, act:ReLU, alpha_0:0.0, layer:128_4_128_4, lrs:[0.001], lr:0.99, ...
    #         lrs_alpha:0.02, lr_alpha: 0.998, w_d:0.001,losss: 1.73e-01, 1.78e-01, 1.75e-01,, best_ep=...
    losss_re = re.compile(r'losss:\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)')
    entries = []
    with open(record_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('Data:'):
                continue
            # Parse key:value pairs (split by ", " but Data: value may not have comma until " act:")
            parts = line.split(', ')
            data_name = act = layer = None
            alpha_0_val = lr = lrs_alpha = lr_alpha = w_d = None
            lrs_val = None
            loss_train = loss_valid = loss_test = None
            for i, p in enumerate(parts):
                if p.startswith('Data:'):
                    data_name = p[5:].strip()
                elif p.startswith('act:'):
                    act = p[4:].strip()
                elif p.startswith('alpha_0:'):
                    try:
                        alpha_0_val = float(p[8:].strip())
                    except ValueError:
                        pass
                elif p.startswith('layer:'):
                    layer = p[6:].strip()
                elif p.startswith('lrs:'):
                    s = p[4:].strip().replace('[', '').replace(']', '')
                    try:
                        lrs_val = float(s)
                    except ValueError:
                        pass
                elif p.startswith('lr:') and not p.startswith('lr_alpha:'):
                    try:
                        lr = float(p[3:].strip())
                    except ValueError:
                        pass
                elif p.startswith('lrs_alpha:'):
                    try:
                        lrs_alpha = float(p[10:].strip())
                    except ValueError:
                        pass
                elif p.startswith('lr_alpha:'):
                    try:
                        lr_alpha = float(p[9:].strip())
                    except ValueError:
                        pass
                elif p.startswith('w_d:'):
                    try:
                        w_d = float(p[4:].strip())
                    except ValueError:
                        pass
                elif p.startswith('losss:'):
                    m = losss_re.search(line)
                    if m:
                        loss_train = float(m.group(1))
                        loss_valid = float(m.group(2))
                        loss_test = float(m.group(3))
                    break
            if data_name is None or layer is None or act is None or lrs_val is None or lr is None:
                continue
            if loss_train is None or loss_valid is None or loss_test is None:
                continue
            if lrs_alpha is None:
                lrs_alpha = 0.01
            if lr_alpha is None:
                lr_alpha = 0.998
            if alpha_0_val is None:
                alpha_0_val = 0.0
            if w_d is None:
                w_d = 0.001
            err_val = (loss_train, loss_valid, loss_test)[error_index]
            entries.append({
                'data_name': data_name, 'act': act, 'alpha_0': alpha_0_val, 'layer': layer,
                'lrs': lrs_val, 'lr': lr, 'lrs_alpha': lrs_alpha, 'lr_alpha': lr_alpha, 'w_d': w_d,
                'error': err_val, 'loss_train': loss_train, 'loss_valid': loss_valid, 'loss_test': loss_test
            })
    if len(entries) < 1:
        return None
    # Sort by error (ascending), take the smallest
    entries.sort(key=lambda x: x['error'])
    best = entries[0]
    # Build Results subdir name (same pattern as in main)
    subdir = 'Results/%s_k%s_g%s_%s_ntrain%s_wd%s_lrs%s_lr%s_lrs_alpha%s_lr_alpha%s_alpha_0%s' % (
        best['data_name'], best['layer'], best['layer'], best['act'], ntrain,
        best['w_d'], best['lrs'], best['lr'], best['lrs_alpha'], best['lr_alpha'], best['alpha_0'])
    # Try stage1_no_singular first, then without suffix
    for suffix in ['_stage1_no_singular', '']:
        ckpt = os.path.join(results_base_dir, subdir + suffix, 'model.ckpt')
        if os.path.isfile(ckpt):
            return ckpt
    return None


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def sanitize_token(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def load_wave_number_metadata(csv_path, decimals=6):
    """Load one stable wave number per file from the per-sample CSV report."""
    per_file = {}
    with open(csv_path, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_index = int(row['file_index'])
            wave_number = round(float(row['estimated_wave_number']), decimals)
            timestep = int(row['timestep'])
            entry = per_file.setdefault(
                file_index,
                {
                    'file_index': file_index,
                    'file_name': row['file_name'],
                    'wave_number': wave_number,
                    'timesteps': [],
                },
            )
            if abs(entry['wave_number'] - wave_number) > 10 ** (-decimals):
                raise ValueError(
                    f"Inconsistent wave numbers for file {file_index}: "
                    f"{entry['wave_number']} vs {wave_number}"
                )
            entry['timesteps'].append(timestep)

    for entry in per_file.values():
        entry['timesteps'] = sorted(entry['timesteps'])
        entry['num_timesteps'] = len(entry['timesteps'])

    return dict(sorted(per_file.items()))


def _build_group_names(num_groups):
    default_names = ['low', 'mid', 'high']
    if num_groups <= len(default_names):
        return default_names[:num_groups]
    extra = [f'group{i + 1}' for i in range(len(default_names), num_groups)]
    return default_names + extra


def _partition_counts_balanced(sorted_items, num_groups):
    """Partition sorted (wave_number, count) items into contiguous balanced groups."""
    counts = [count for _, count in sorted_items]
    target = float(sum(counts)) / float(num_groups)
    best = {'objective': None, 'cuts': None}

    def recurse(start_index, groups_left, current_cuts):
        if groups_left == 1:
            segments = []
            prev = 0
            for cut in current_cuts + [len(counts)]:
                segments.append(sum(counts[prev:cut]))
                prev = cut
            deviations = [abs(segment - target) for segment in segments]
            objective = (max(deviations), sum(deviations), current_cuts)
            if best['objective'] is None or objective < best['objective']:
                best['objective'] = objective
                best['cuts'] = list(current_cuts)
            return

        min_cut = start_index + 1
        max_cut = len(counts) - groups_left + 1
        for cut in range(min_cut, max_cut + 1):
            recurse(cut, groups_left - 1, current_cuts + [cut])

    recurse(0, num_groups, [])

    partitions = []
    prev = 0
    for cut in best['cuts'] + [len(sorted_items)]:
        partitions.append(sorted_items[prev:cut])
        prev = cut
    return partitions


def build_wave_number_groups(per_file_metadata, num_groups=3):
    """Create contiguous low/mid/high wave-number groups with balanced file counts."""
    counts_by_wave = {}
    for entry in per_file_metadata.values():
        counts_by_wave.setdefault(entry['wave_number'], []).append(entry)

    sorted_items = sorted((wave, len(entries)) for wave, entries in counts_by_wave.items())
    partitions = _partition_counts_balanced(sorted_items, num_groups)
    group_names = _build_group_names(num_groups)
    groups = []

    for group_name, partition in zip(group_names, partitions):
        wave_numbers = [wave for wave, _ in partition]
        files = []
        for wave in wave_numbers:
            files.extend(sorted(counts_by_wave[wave], key=lambda item: item['file_index']))
        groups.append(
            {
                'group_name': group_name,
                'wave_numbers': wave_numbers,
                'wave_number_min': min(wave_numbers),
                'wave_number_max': max(wave_numbers),
                'file_indices': [entry['file_index'] for entry in files],
                'file_names': [entry['file_name'] for entry in files],
                'files': files,
                'file_count': len(files),
                'sample_count': int(sum(entry['num_timesteps'] for entry in files)),
            }
        )

    return groups


def export_wave_group_manifest(groups, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_md = os.path.join(output_dir, 'wave_group_summary.md')
    assignments_md = os.path.join(output_dir, 'wave_group_assignments.md')
    manifest_json = os.path.join(output_dir, 'wave_group_manifest.json')

    summary_lines = [
        '# Wave-Number Group Summary',
        '',
        '| Group | Wave-number min | Wave-number max | File count | Sample count | Wave numbers |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for group in groups:
        summary_lines.append(
            '| '
            f"{group['group_name']} | {group['wave_number_min']:.6f} | {group['wave_number_max']:.6f} | "
            f"{group['file_count']} | {group['sample_count']} | "
            f"{';'.join(f'{value:.6f}' for value in group['wave_numbers'])} |"
        )
    with open(summary_md, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(summary_lines) + '\n')

    assignment_lines = [
        '# Wave-Number Group Assignments',
        '',
        '| Group | File index | File name | Wave number |',
        '| --- | ---: | --- | ---: |',
    ]
    for group in groups:
        for file_entry in group['files']:
            assignment_lines.append(
                '| '
                f"{group['group_name']} | {file_entry['file_index']} | {file_entry['file_name']} | {file_entry['wave_number']:.6f} |"
            )
    with open(assignments_md, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(assignment_lines) + '\n')

    with open(manifest_json, 'w', encoding='utf-8') as handle:
        json.dump(groups, handle, indent=2)

    return {
        'summary_md': summary_md,
        'assignments_md': assignments_md,
        'manifest_json': manifest_json,
    }


def write_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)


def read_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def read_loss_history(loss_path):
    if not os.path.isfile(loss_path):
        return np.empty((0, 2), dtype=float)
    history = np.loadtxt(loss_path)
    if history.size == 0:
        return np.empty((0, 2), dtype=float)
    history = np.atleast_2d(history)
    return history


def collect_result_summaries(results_dir):
    summaries = []
    if not os.path.isdir(results_dir):
        return summaries
    for entry in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, entry, 'summary.json')
        if os.path.isfile(summary_path):
            summary = read_json(summary_path)
            summary['result_dir'] = os.path.join(results_dir, entry)
            summaries.append(summary)
    return summaries


def select_best_summaries_by_group(summaries):
    best = {}
    for summary in summaries:
        group_name = summary['group_name']
        current = best.get(group_name)
        if current is None or summary['best_valid_loss'] < current['best_valid_loss']:
            best[group_name] = summary
    return dict(sorted(best.items()))


def generate_quadrature_ordered_connectivity(x, delta, dx, S, tolerance=1e-8, interior_mask=None):
    """
    Generate edge connectivity for interior nodes following quadrature weights order.
    
    For each interior node, find all neighbors within distance delta, in quadrature order.
    Build edge_index (row = center, col = neighbor) and return interior_node_mask.
    Only nodes marked as interior get outgoing edges; others get force = 0.
    
    Parameters:
    -----------
    x : torch.Tensor
        Node coordinates [n_nodes, 2]
    delta : float
        Integration radius
    dx : float
        Grid spacing
    S : int
        Mesh size (S x S)
    tolerance : float
        Tolerance for matching coordinates (default: 1e-8)
    interior_mask : torch.Tensor or None
        If provided, boolean mask [n_nodes] or [S,S] defining which nodes are "interior"
        (centers for building edges). Must match the evaluation region (e.g. from cond_f).
        If None, interior = nodes with coords in [x_min+delta, x_max-delta] (at least
        delta from full-grid boundary), which can be a smaller set than the evaluation region.
    
    Returns:
    --------
    edge_index : torch.Tensor
        Edge indices [2, n_edges] where edge_index[0] is source (interior nodes) 
        and edge_index[1] is target (neighbors in quadrature order)
    interior_node_mask : torch.Tensor
        Boolean mask indicating which nodes are interior [n_nodes]
    """
    # Generate quadrature points order (same as quadweights_2d)
    # Ensure delta and dx are Python floats
    if isinstance(delta, torch.Tensor):
        delta = delta.item() if delta.numel() == 1 else float(delta.cpu().numpy())
    if isinstance(dx, torch.Tensor):
        dx = dx.item() if dx.numel() == 1 else float(dx.cpu().numpy())
    delta = float(delta)
    dx = float(dx)
    
    a, b = -delta, delta
    center = (a + b) / 2.0
    # Symmetric grid anchored on multiples of dx around 0 (kept identical to
    # the construction in quadweights_2d so that ewi indexing stays aligned).
    m = int(round(delta / dx))
    x_grid = np.arange(-m, m + 1) * dx
    y_grid = np.arange(-m, m + 1) * dx
    
    # Generate quadrature points in order (excluding origin)
    quad_points_relative = []
    for x_val in x_grid:
        for y_val in y_grid:
            dx_rel = x_val - center
            dy_rel = y_val - center
            r = np.sqrt(dx_rel**2 + dy_rel**2)
            # Exclude center point (r=0) - this is the origin
            if r <= delta and r > 1e-10:
                quad_points_relative.append([x_val, y_val])
    
    quad_points_relative = np.array(quad_points_relative)  # [n_quad_points, 2]
    
    # Identify interior nodes: use provided mask (e.g. from cond_f) so integration region
    # matches evaluation region; otherwise "at least delta from full-grid boundary".
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    
    # Ensure delta and tolerance are Python floats
    if isinstance(delta, torch.Tensor):
        delta = delta.item() if delta.numel() == 1 else float(delta.cpu().numpy())
    delta = float(delta)
    tolerance = float(tolerance)
    
    if interior_mask is not None:
        if isinstance(interior_mask, torch.Tensor):
            interior_mask = interior_mask.cpu().numpy()
        if interior_mask.ndim == 2:
            interior_mask = interior_mask.ravel()
        interior_mask = interior_mask.astype(bool)
        if interior_mask.shape[0] != x_np.shape[0]:
            raise ValueError("interior_mask length must be n_nodes (or S*S).")
    else:
        x_min = float(x_np[:, 0].min())
        x_max = float(x_np[:, 0].max())
        y_min = float(x_np[:, 1].min())
        y_max = float(x_np[:, 1].max())
        interior_mask = (
            (x_np[:, 0] >= x_min + delta - tolerance) &
            (x_np[:, 0] <= x_max - delta + tolerance) &
            (x_np[:, 1] >= y_min + delta - tolerance) &
            (x_np[:, 1] <= y_max - delta + tolerance)
        )
    interior_indices = np.where(interior_mask)[0]
    
    # Build edge_index following quadrature order
    edge_rows = []
    edge_cols = []
    
    # Use a more lenient tolerance for uniform grid matching
    match_tolerance = max(tolerance, dx * 0.1)  # At least 10% of grid spacing
    
    for interior_idx in interior_indices:
        center_node = x_np[interior_idx]  # [2]
        
        # For each quadrature point (relative to origin), find corresponding neighbor
        for quad_point_rel in quad_points_relative:
            # Absolute position of neighbor
            neighbor_pos = center_node + quad_point_rel  # [2]
            
            # Find the node closest to this position
            dists = np.sum((x_np - neighbor_pos) ** 2, axis=1)
            closest_idx = np.argmin(dists)
            closest_dist = np.sqrt(dists[closest_idx])
            
            # Check if within tolerance (for uniform grid, should be very close)
            # Also verify the actual distance from center is within delta
            actual_dist = np.sqrt(np.sum(quad_point_rel ** 2))
            if closest_dist < match_tolerance and closest_idx != interior_idx and actual_dist <= delta + tolerance:
                edge_rows.append(interior_idx)
                edge_cols.append(closest_idx)
    
    edge_index = torch.tensor([edge_rows, edge_cols], dtype=torch.long)
    interior_node_mask = torch.tensor(interior_mask, dtype=torch.bool)
    
    return edge_index, interior_node_mask


def compute_quadrature_weights_simple(delta, dx, alpha, p_order, device):
    """
    Compute quadrature weights for 2D integration in original quadrature order.
    
    Parameters:
    -----------
    delta : float
        Integration radius
    dx : float
        Grid spacing
    alpha : float
        Singularity parameter for quad_rule_2d
    p_order : int
        Polynomial order for quadrature
    device : torch.device
        Device to place tensors on
    
    Returns:
    --------
    quad_points_weights : torch.Tensor
        Quadrature weights [n_quad_points, 3] with columns [x, y, weight]
        Points are ordered as in quadweights_2d (x_grid outer loop, y_grid inner loop)
    """
    # Compute quadrature weights for 2D integration
    a, b = -delta, delta
    points_weights_np, _ = quadweights_2d(a, b, delta, dx, float(alpha), p_order)
    quad_points_weights = torch.from_numpy(points_weights_np).float().to(device)
    
    return quad_points_weights


def compute_and_reorder_quadrature_weights(x_train, delta, dx, alpha, p_order, S, device, tolerance=1e-8, validate_quad_match=False, interior_mask=None):
    """
    Compute quadrature weights and create edge weight indices for quadrature-ordered connectivity.

    Node--quadrature correspondence (must match quad_rule_2d.quadweights_2d):
    - quadweights_2d(a=-delta, b=delta, h=dx) builds points in [a,b]x[a,b] with step dx,
      order: for x in x_grid (outer), for y in y_grid (inner), append (x,y) if r in (0, delta].
    - generate_quadrature_ordered_connectivity uses the SAME a,b,dx and SAME loop order to build
      quad_points_relative. So the j-th quadrature point and the j-th relative offset used for
      each interior node's neighbors are the same (x,y).
    - For each edge e, ksi = x[col]-x[row] equals (up to grid snapping) the quadrature point
      position. We map e to the quadrature index by matching ksi to quad_points_weights[:, :2].

    Parameters:
    -----------
    x_train : torch.Tensor
        Training node coordinates [n_nodes, 2]
    delta : float
        Integration radius
    dx : float
        Grid spacing (same as h in quadweights_2d)
    alpha : float
        Singularity parameter for quad_rule_2d
    p_order : int
        Polynomial order for quadrature
    S : int
        Mesh size (S x S)
    device : torch.device
        Device to place tensors on
    tolerance : float
        Base tolerance for matching; effective match uses max(tolerance, dx*0.15)
    validate_quad_match : bool
        If True, check that max |ksi - quad_point| is small and print a short report
    interior_mask : torch.Tensor or None
        If provided, boolean mask for interior nodes (same as passed to generate_quadrature_ordered_connectivity).
        If None, interior is computed from full-grid boundary (at least delta from boundary).

    Returns:
    --------
    quad_points_weights : torch.Tensor
        Quadrature weights [n_quad_points, 3] with columns [x, y, weight]
    edge_weight_indices : torch.Tensor
        Index mapping from edges to quadrature weights [n_edges]
    interior_mask : torch.Tensor
        Boolean mask of interior nodes
    """
    # Compute quadrature weights in original order (same grid/order as generate_quadrature_ordered_connectivity)
    quad_points_weights = compute_quadrature_weights_simple(delta, dx, alpha, p_order, device)

    # Generate edge_index using quadrature-ordered connectivity (same dx, same loop order)
    edge_index, interior_mask = generate_quadrature_ordered_connectivity(
        x_train, delta, dx, S, tolerance, interior_mask=interior_mask
    )

    edge_index = edge_index.to(device)

    x_train_device = x_train.to(device) if isinstance(x_train, torch.Tensor) else torch.tensor(x_train, device=device)
    row, col = edge_index
    ksi = x_train_device[col] - x_train_device[row]  # [n_edges, 2], same frame as quad points

    quad_points_rel = quad_points_weights[:, :2]  # [n_quad_points, 2], relative to origin

    # Match tolerance: allow for grid snapping (neighbor may be within ~0.1*dx of exact quad point)
    match_tol = max(float(tolerance), float(dx) * 0.15)

    edge_weight_indices = torch.zeros(ksi.size(0), dtype=torch.long, device=device)
    max_dist_sq = 0.0

    for i in range(ksi.size(0)):
        dists = torch.sum((quad_points_rel - ksi[i]) ** 2, dim=1)
        best_match_idx = torch.argmin(dists).item()
        best_dist = torch.sqrt(dists[best_match_idx]).item()
        if best_dist * best_dist > max_dist_sq:
            max_dist_sq = best_dist * best_dist
        edge_weight_indices[i] = best_match_idx

    if validate_quad_match:
        max_dist = np.sqrt(max_dist_sq)
        n_quad = quad_points_rel.size(0)
        print(f"  [quad match] max |ksi - quad_point| = {max_dist:.2e}, dx = {dx:.2e}, n_quad = {n_quad}")
        if max_dist > match_tol:
            print(f"  [quad match] warning: max dist > match_tol ({match_tol:.2e}); check grid/order consistency.")

    return quad_points_weights, edge_weight_indices, interior_mask


def plot_u_b(data_x, u_plot, b_pred_plot, b_true_plot, base_dir, ex, n):
    """
    Combined plot with 2 rows and 4 columns:
    Column 1: u1, u2
    Column 2: true b1, true b2
    Column 3: predicted b1, predicted b2
    Column 4: absolute error of b1, absolute error of b2
    """
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    # Extract coordinate ranges from data_x
    x1_min = data_x[:, :, 0].min()
    x1_max = data_x[:, :, 0].max()
    x2_min = data_x[:, :, 1].min()
    x2_max = data_x[:, :, 1].max()
    extent = [x1_min, x1_max, x2_min, x2_max]
    
    # Loop over two components (u1/b1 and u2/b2)
    for i in range(2):
        component_idx = i + 1  # 1 or 2 for display
        
        # Get min/max values for current component
        vmin_u = u_plot[:, :, i].min()
        vmax_u = u_plot[:, :, i].max()
        vmin_b = min(b_pred_plot[:, :, i].min(), b_true_plot[:, :, i].min())
        vmax_b = max(b_pred_plot[:, :, i].max(), b_true_plot[:, :, i].max())
        
        # Calculate error
        error = b_true_plot[:, :, i] - b_pred_plot[:, :, i]
        vmax_error = np.max(np.abs(error))
        
        # Define plot data for each column
        plot_data = [
            (u_plot[:, :, i], vmin_u, vmax_u, f'True $u_{component_idx}$', 'viridis'),
            (b_true_plot[:, :, i], vmin_b, vmax_b, f'True $b_{component_idx}$', 'viridis'),
            (b_pred_plot[:, :, i], vmin_b, vmax_b, f'Predicted $b_{component_idx}$', 'viridis'),
            (error, -vmax_error, vmax_error, f'Absolute Error of $b_{component_idx}$', 'viridis')
        ]
        
        # Plot each column
        for j, (data, vmin, vmax, title, cmap) in enumerate(plot_data):
            im = axes[i, j].imshow(data, vmin=vmin, vmax=vmax, aspect='equal', 
                                  origin='lower', extent=extent, cmap=cmap)
            axes[i, j].set_title(title, pad=15)
            axes[i, j].set_xlabel('$x_1$')
            axes[i, j].set_ylabel('$x_2$')
            fig.colorbar(im, ax=axes[i, j], shrink=0.85, aspect=20)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  # Adjust vertical spacing between rows
    plt.show()
    plt.savefig('%s/2D_%s_u_b_%s.png' % (base_dir, ex, n), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    


def plot_b(data_x, out_plot, f_gt_plot, base_dir, ex, n):
    
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 13))
    # fig.suptitle('Displacement Field Comparison: Predicted vs Ground Truth', fontsize=16, fontweight='bold', y=0.95)
    
    # Extract coordinate ranges from data_x
    x1_min = data_x[:, :, 0].min()
    x1_max = data_x[:, :, 0].max()
    x2_min = data_x[:, :, 1].min()
    x2_max = data_x[:, :, 1].max()
    extent = [x1_min, x1_max, x2_min, x2_max]
    
    # Define colormap and normalization for consistent scaling
    vmin_b0 = min(out_plot[:,:,0].min(), f_gt_plot[:,:,0].min())
    vmax_b0 = max(out_plot[:,:,0].max(), f_gt_plot[:,:,0].max())
    vmin_b1 = min(out_plot[:,:,1].min(), f_gt_plot[:,:,1].min())
    vmax_b1 = max(out_plot[:,:,1].max(), f_gt_plot[:,:,1].max())
    
    # Plot b0 (x-component)
    im0 = axes[0, 0].imshow(out_plot[:,:,0], vmin=vmin_b0, vmax=vmax_b0, aspect='equal', origin='lower', extent=extent)
    axes[0, 0].set_title('Predicted $b_1$', fontweight='bold', pad=15)
    axes[0, 0].set_xlabel('$x_1$')
    axes[0, 0].set_ylabel('$x_2$')
    cbar0 = fig.colorbar(im0, ax=axes[0,0], shrink=0.7, aspect=20)
    
    im1 = axes[0,1].imshow(f_gt_plot[:,:,0], vmin=vmin_b0, vmax=vmax_b0, aspect='equal', origin='lower', extent=extent)
    axes[0,1].set_title('True $b_1$', fontweight='bold', pad=15)
    axes[0,1].set_xlabel('$x_1$')
    axes[0,1].set_ylabel('$x_2$')
    cbar1 = fig.colorbar(im1, ax=axes[0,1], shrink=0.7, aspect=20)
    
    error_b0 = f_gt_plot[:,:,0] - out_plot[:,:,0]
    vmax_error_b0 = np.max(np.abs(error_b0))
    im2 = axes[0,2].imshow(error_b0, vmin=-vmax_error_b0, vmax=vmax_error_b0, aspect='equal', origin='lower', extent=extent)
    axes[0,2].set_title('Absolute Error of $b_1$', fontweight='bold', pad=15)
    axes[0,2].set_xlabel('$x_1$')
    axes[0,2].set_ylabel('$x_2$')
    cbar2 = fig.colorbar(im2, ax=axes[0,2], shrink=0.7, aspect=20)
    
    # Plot b1 (y-component)
    im3 = axes[1, 0].imshow(out_plot[:,:,1], vmin=vmin_b1, vmax=vmax_b1, aspect='equal', origin='lower', extent=extent)
    axes[1, 0].set_title('Predicted $b_2$', fontweight='bold', pad=15)
    axes[1, 0].set_xlabel('$x_1$')
    axes[1, 0].set_ylabel('$x_2$')
    cbar3 = fig.colorbar(im3, ax=axes[1,0], shrink=0.7, aspect=20)
    
    im4 = axes[1,1].imshow(f_gt_plot[:,:,1], vmin=vmin_b1, vmax=vmax_b1, aspect='equal', origin='lower', extent=extent)
    axes[1,1].set_title('True $b_2$', fontweight='bold', pad=15)
    axes[1,1].set_xlabel('$x_1$')
    axes[1,1].set_ylabel('$x_2$')
    cbar4 = fig.colorbar(im4, ax=axes[1,1], shrink=0.7, aspect=20)
    
    error_b1 = f_gt_plot[:,:,1] - out_plot[:,:,1]
    vmax_error_b1 = np.max(np.abs(error_b1))
    im5 = axes[1,2].imshow(error_b1, vmin=-vmax_error_b1, vmax=vmax_error_b1, aspect='equal', origin='lower', extent=extent)
    axes[1,2].set_title('Absolute Error of $b_2$', fontweight='bold', pad=15)
    axes[1,2].set_xlabel('$x_1$')
    axes[1,2].set_ylabel('$x_2$')
    cbar5 = fig.colorbar(im5, ax=axes[1,2], shrink=0.7, aspect=20)
    
    # # Add error statistics as text
    # l2_error_b0 = np.linalg.norm(error_b0) / np.linalg.norm(f_gt_plot[:,:,0])
    # l2_error_b1 = np.linalg.norm(error_b1) / np.linalg.norm(f_gt_plot[:,:,1])
    
    # Add text box with error statistics
    # textstr = f'Relative $L^2$ Error:\n$b_0$: {l2_error_b0:.2e}\n$b_1$: {l2_error_b1:.2e}'
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # fig.text(0.02, 0.02, textstr, fontsize=11, verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15)
    plt.show()
    plt.savefig('%s/MD_fracture_%s_b%s.png' % (base_dir, ex, n), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    

def parse_layer_info(layer_info, input_dim=1):
    """
    Parse layer string like '64_4' into [input_dim, width, ..., width, 1].
    For g (lambda): input_dim=1. For k(ksi) with 2D ksi: input_dim=2.
    """
    parts = layer_info.split('_')
    if len(parts) != 2:
        raise ValueError(f"layer_info must have 2 parts separated by '_', got: {layer_info}")
    ker_width = int(parts[0])
    ker_layers = int(parts[1])
    ker_layers_list = [input_dim] + [ker_width] * ker_layers + [1]
    return ker_layers_list


def parse_layer_info_4parts(layer_info):
    """
    Same as PNO_uniqueness_architecture_new_f_mix: g_width, g_layers, k_width, k_layers.
    Returns g_layers_list = [1, g_width, ..., 1], k_layers_list = [2, k_width, ..., 1].
    For MPNOinit, use k_layer_mpno = [2] + k_layers_list[1:-1] + [1] (k(ksi) takes 2D ksi).
    """
    parts = layer_info.split('_')
    if len(parts) != 4:
        raise ValueError(f"layer_info must have 4 parts (g_width_g_layers_k_width_k_layers), got: {layer_info}")
    g_width = int(parts[0])
    g_layers = int(parts[1])
    k_width = int(parts[2])
    k_layers = int(parts[3])
    g_layers_list = [1] + [g_width] * g_layers + [1]
    k_layers_list = [2] + [k_width] * k_layers + [1]
    return g_layers_list, k_layers_list

class MatReader:
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super().__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = scipy.io.loadmat(self.file_path)
            # self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class UnitGaussianNormalizer:
    def __init__(self, x, eps=1e-5):

        self.mean = torch.mean(x, 0).view(-1)
        self.std = torch.std(x, 0).view(-1)

        self.eps = eps

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = x.view(s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = x.size()
        x = x.view(s[0], -1)
        x = (x * std) + mean
        x = x.view(s)
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class GaussianNormalizer:
    def __init__(self, x, eps=1e-9):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a * x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b) / self.a
        x = x.view(s)
        return x


class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        # torch.norm is deprecated
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1) + 1e-9

        # diff_norms = torch.linalg.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1))
        # y_norms = torch.linalg.norm(y.reshape(num_examples, -1))

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms ** 2.0 / y_norms ** 2.0)
            else:
                return torch.sum(diff_norms ** 2.0 / y_norms ** 2.0)

        return diff_norms / y_norms

    def rel_sqrt(self, x, y):
        """Relative L2 error (same as PNO_BB for comparable train/valid/test err)."""
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1) + 1e-9
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class SquareMeshGenerator:
    def __init__(self, real_space, mesh_size):

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]
            # xx.ravel() is equiv to xx.reshape(-1)
            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    # the 1st col of edge_attr is the edge length for equivariance
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                # edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                #edge_attr = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                #                            *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
                edge_attr = None
            else:
                edge_attr = np.zeros((self.n_edges, 2))
                dist = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                                       *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                   dtype=float)

                # ang: angle based on local coordinate axis of
                # (x,y)=(self.grid[1,0]-self.grid[0,0],self.grid[1,1]-self.grid[0,1])
                ang = np.fromiter(map(lambda x1, y1, x2, y2: math.atan2(y2 - y1, x2 - x1) -
                                                             math.atan2(self.grid[1, 1] - self.grid[0, 1],
                                                                        self.grid[1, 0] - self.grid[0, 0]),
                                      *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                  dtype=float)
                # enforce angle in the range [0, 2*pi]
                for ii in range(self.n_edges):
                    if ang[ii] < 0: ang[ii] += 2 * math.pi
                    edge_attr[ii, 0] = dist[ii] * np.cos(ang[ii])
                    edge_attr[ii, 1] = dist[ii] * np.sin(ang[ii])
                # edge_attr[:, 2] = theta[self.edge_index[0]]
                # edge_attr[:, 3] = theta[self.edge_index[1]]
        else:
            # xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            #xy = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
            #                                *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
            xy = None
            if theta is None:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
                edge_attr = f(xy)
            else:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])
                edge_attr = f(xy, theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class IrregularMeshGenerator:
    def __init__(self, grid, mesh_size):
        self.n, self.d = grid.shape
        self.s = mesh_size[0]
        assert len(mesh_size) == self.d
        self.grid = grid

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        # self.edge_index = np.vstack(np.where(pwd <= r ))
        self.edge_index = np.vstack(np.where((pwd <= r) & (pwd > 1e-10) ))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    # the 1st col of edge_attr is the edge length for equivariance
    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                # edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                #edge_attr = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                #                            *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
                edge_attr = None
            else:
                edge_attr = np.zeros((self.n_edges, 2))
                dist = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
                                       *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                   dtype=float)

                # ang: angle based on local coordinate axis of
                # (x,y)=(self.grid[1,0]-self.grid[0,0],self.grid[1,1]-self.grid[0,1])
                ang = np.fromiter(map(lambda x1, y1, x2, y2: math.atan2(y2 - y1, x2 - x1) -
                                                             math.atan2(self.grid[1, 1] - self.grid[0, 1],
                                                                        self.grid[1, 0] - self.grid[0, 0]),
                                      *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T),
                                  dtype=float)
                # enforce angle in the range [0, 2*pi]
                for ii in range(self.n_edges):
                    if ang[ii] < 0: ang[ii] += 2 * math.pi
                    edge_attr[ii, 0] = dist[ii] * np.cos(ang[ii])
                    edge_attr[ii, 1] = dist[ii] * np.sin(ang[ii])
                # edge_attr[:, 2] = theta[self.edge_index[0]]
                # edge_attr[:, 3] = theta[self.edge_index[1]]
        else:
            # xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            #xy = np.fromiter(map(lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2),
            #                                *self.grid[self.edge_index.T].reshape((self.n_edges, -1)).T), dtype=float)
            xy = None
            if theta is None:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
                edge_attr = f(xy)
            else:
                # edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])
                edge_attr = f(xy, theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
            self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    # def get_boundary(self):
    #     s = self.s
    #     n = self.n
    #     boundary1 = np.array(range(0, s))
    #     boundary2 = np.array(range(n - s, n))
    #     boundary3 = np.array(range(s, n, s))
    #     boundary4 = np.array(range(2 * s - 1, n, s))
    #     self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])
    #
    # def boundary_connectivity2d(self, stride=1):
    #
    #     boundary = self.boundary[::stride]
    #     boundary_size = len(boundary)
    #     vertice1 = np.array(range(self.n))
    #     vertice1 = np.repeat(vertice1, boundary_size)
    #     vertice2 = np.tile(boundary, self.n)
    #     self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
    #     self.n_edges_boundary = self.edge_index_boundary.shape[1]
    #     return torch.tensor(self.edge_index_boundary, dtype=torch.long)
    #
    # def attributes_boundary(self, f=None, theta=None):
    #     # if self.edge_index_boundary == None:
    #     #     self.boundary_connectivity2d()
    #     if f is None:
    #         if theta is None:
    #             edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         else:
    #             edge_attr_boundary = np.zeros((self.n_edges_boundary, 3*self.d))
    #             edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #             edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
    #             edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
    #     else:
    #         xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
    #         if theta is None:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
    #         else:
    #             edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])
    #
    #     return torch.tensor(edge_attr_boundary, dtype=torch.float)


class RandomGridSplitter(object):
    def __init__(self, grid, resolution, m=200, l=2, radius=0.25):
        super(RandomGridSplitter, self).__init__()

        self.grid = grid
        self.resolution = resolution
        self.n = resolution ** 2
        self.m = m
        self.l = l
        self.radius = radius

        assert self.n % self.m == 0
        self.num = self.n // self.m

    def get_data(self, theta):

        data = []
        for i in range(self.l):
            perm = torch.randperm(self.n)
            perm = perm.reshape(self.num, self.m)

            for j in range(self.num):
                idx = perm[j, :].reshape(-1, )
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                X = torch.cat([grid_sample, theta_sample], dim=1)

                pwd = sklearn.metrics.pairwise_distances(grid_sample)
                edge_index = np.vstack(np.where(pwd <= self.radius))
                n_edges = edge_index.shape[1]
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                edge_attr = np.zeros((n_edges, 6))
                a = theta_sample[:, 0]
                edge_attr[:, :4] = grid_sample[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4] = a[edge_index[0]]
                edge_attr[:, 5] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.num * self.l // batch_size2

        out = torch.zeros(self.n, )
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i].reshape(batch_size2, self.m)
            for j in range(batch_size2):
                pred_ij = pred_i[j, :].reshape(-1, )
                idx = split_idx_i[j, :].reshape(-1, )
                out[idx] = pred_ij

        out = out / self.l

        # out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        # out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


class DownsampleGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(DownsampleGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution ** 2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def ball_connectivity(self, grid):
        pwd = sklearn.metrics.pairwise_distances(grid)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
                theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges = self.ball_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1, )
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1, )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1, )
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges = self.ball_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 4 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
        edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                    sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r ** 2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


class TorusGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, edge_features=1):
        super(TorusGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution, 2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution % 2 == 1:
            self.s = int(((resolution - 1) / r) + 1)
        else:
            self.s = int(resolution / r)
        self.r = r
        self.n = resolution ** 2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def pairwise_difference(self, grid1, grid2):
        n = grid1.shape[0]
        x1 = grid1[:, 0]
        y1 = grid1[:, 1]
        x2 = grid2[:, 0]
        y2 = grid2[:, 1]

        X1 = np.tile(x1.reshape(n, 1), [1, n])
        X2 = np.tile(x2.reshape(1, n), [n, 1])
        X_diff = X1 - X2
        Y1 = np.tile(y1.reshape(n, 1), [1, n])
        Y2 = np.tile(y2.reshape(1, n), [n, 1])
        Y_diff = Y1 - Y2

        return X_diff, Y_diff

    def torus_connectivity(self, grid):
        pwd0 = sklearn.metrics.pairwise_distances(grid, grid)
        X_diff0, Y_diff0 = self.pairwise_difference(grid, grid)

        grid1 = grid
        grid1[:, 0] = grid[:, 0] + 1
        pwd1 = sklearn.metrics.pairwise_distances(grid, grid1)
        X_diff1, Y_diff1 = self.pairwise_difference(grid, grid1)

        grid2 = grid
        grid2[:, 1] = grid[:, 1] + 1
        pwd2 = sklearn.metrics.pairwise_distances(grid, grid2)
        X_diff2, Y_diff2 = self.pairwise_difference(grid, grid2)

        grid3 = grid
        grid3[:, :] = grid[:, :] + 1
        pwd3 = sklearn.metrics.pairwise_distances(grid, grid3)
        X_diff3, Y_diff3 = self.pairwise_difference(grid, grid3)

        grid4 = grid
        grid4[:, 0] = grid[:, 0] + 1
        grid4[:, 1] = grid[:, 1] - 1
        pwd4 = sklearn.metrics.pairwise_distances(grid, grid4)
        X_diff4, Y_diff4 = self.pairwise_difference(grid, grid4)

        PWD = np.stack([pwd0, pwd1, pwd2, pwd3, pwd4], axis=2)
        X_DIFF = np.stack([X_diff0, X_diff1, X_diff2, X_diff3, X_diff4], axis=2)
        Y_DIFF = np.stack([Y_diff0, Y_diff1, Y_diff2, Y_diff3, Y_diff4], axis=2)
        pwd = np.min(PWD, axis=2)
        pwd_index = np.argmin(PWD, axis=2)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        pwd_index = pwd_index[np.where(pwd <= self.radius)]
        PWD_index = (np.where(pwd <= self.radius)[0], np.where(pwd <= self.radius)[1], pwd_index)
        distance = PWD[PWD_index]
        X_difference = X_DIFF[PWD_index]
        Y_difference = Y_DIFF[PWD_index]
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges, distance, X_difference, Y_difference

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
                theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n, -1)[idx]
                theta_sample = theta.reshape(self.n, -1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample], dim=0)
                theta_split = torch.cat([theta_sub, theta_sample], dim=0)
                X = torch.cat([grid_split, theta_split], dim=1)

                edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, 0] = X_difference.reshape(n_edges, )
                edge_attr[:, 1] = Y_difference.reshape(n_edges, )
                edge_attr[:, 2] = distance.reshape(n_edges, )
                edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y, connectivity='ball'):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1, )
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1, )
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1, )
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, 0] = X_difference.reshape(n_edges, )
        edge_attr[:, 1] = Y_difference.reshape(n_edges, )
        edge_attr[:, 2] = distance.reshape(n_edges, )
        edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                    sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r ** 2 // batch_size2

        out = torch.zeros((self.resolution, self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j, :]
                x, y = split_idx_i[j]
                if self.resolution % 2 == 1:
                    if x == 0:
                        nx = self.s
                    else:
                        nx = self.s - 1
                    if y == 0:
                        ny = self.s
                    else:
                        ny = self.s - 1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx, ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1, )


def downsample(data, grid_size, l):
    data = data.reshape(-1, grid_size, grid_size)
    data = data[:, ::l, ::l]
    data = data.reshape(-1, (grid_size // l) ** 2)
    return data


def grid(n_x, n_y):
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if (y != n_y - 1):
                edge_index.append((i, i + n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i + n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y + 1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge_aug_full(n_x, n_y, r, a):
    n = n_x * n_y

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []

    for i1 in range(n):
        x1 = grid[i1]
        for i2 in range(n):
            x2 = grid[i2]

            d = np.linalg.norm(x1 - x2)

            if (d <= r):
                a1 = a[i1]
                a2 = a[i2]
                edge_index.append((i1, i2))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i2, i1))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def multi_grid(depth, n_x, n_y, grid, params):
    edge_index_global = []
    edge_attr_global = []
    X_global = []
    num_nodes = 0

    # build connected graph
    for l in range(depth):
        h_x_l = n_x // (2 ** l)
        h_y_l = n_y // (2 ** l)
        n_l = h_x_l * h_y_l

        a = downsample(params, n_x, (2 ** l))
        if grid == 'grid':
            X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)
        elif grid == 'grid_edge':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)
        elif grid == 'grid_edge_aug':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)

        # update index
        edge_index_inner = edge_index_inner + num_nodes
        edge_index_global.append(edge_index_inner)
        edge_attr_global.append(edge_attr_inner)

        # construct X
        # if (is_high):
        #     X = torch.cat([torch.zeros(n_l, l * 2), X, torch.zeros(n_l, (depth - 1 - l) * 2)], dim=1)
        # else:
        #     X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
        #     X = torch.cat([X, X_l], dim=1)
        X_global.append(X)

        # construct edges
        index1 = torch.tensor(range(n_l), dtype=torch.long)
        index1 = index1 + num_nodes
        num_nodes += n_l

        # #construct inter-graph edge
        if l != depth - 1:
            index2 = np.array(range(n_l // 4)).reshape(h_x_l // 2, h_y_l // 2)  # torch.repeat is different from numpy
            index2 = index2.repeat(2, axis=0).repeat(2, axis=1)
            index2 = torch.tensor(index2).reshape(-1)
            index2 = index2 + num_nodes
            index2 = torch.tensor(index2, dtype=torch.long)

            edge_index_inter1 = torch.cat([index1, index2], dim=-1).reshape(2, -1)
            edge_index_inter2 = torch.cat([index2, index1], dim=-1).reshape(2, -1)
            edge_index_inter = torch.cat([edge_index_inter1, edge_index_inter2], dim=1)

            edge_attr_inter1 = torch.tensor((0, 0, 1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter2 = torch.tensor((0, 0, -1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter = torch.cat([edge_attr_inter1, edge_attr_inter2], dim=0)

            edge_index_global.append(edge_index_inter)
            edge_attr_global.append(edge_attr_inter)

    X = torch.cat(X_global, dim=0)
    edge_index = torch.cat(edge_index_global, dim=1)
    edge_attr = torch.cat(edge_attr_global, dim=0)
    mask_index = torch.tensor(range(n_x * n_y), dtype=torch.long)
    # print('create multi_grid with size:', X.shape,  edge_index.shape, edge_attr.shape, mask_index.shape)

    return (X, edge_index, edge_attr, mask_index, num_nodes)
