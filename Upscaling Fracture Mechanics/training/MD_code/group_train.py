#!/usr/bin/env python3
"""Train PNO models on low/mid/high Fourier wave-number groups."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from egnn_gcl import E_GCL_GKN
from utilities_INO_PD import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--group-name', type=str, required=True, choices=['low', 'mid', 'high'])
    parser.add_argument('--run-tag', type=str, default='')
    parser.add_argument('--num-groups', type=int, default=3)
    parser.add_argument('--wave-csv', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--reports-dir', type=str, default='reports')
    parser.add_argument('--results-dir', type=str, default='Results')
    parser.add_argument('--k_layer', type=str, default='128_4')
    parser.add_argument('--g_layer', type=str, default='128_4')
    parser.add_argument('--layer_info', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lrs', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.995)
    parser.add_argument('--wds', type=float, default=1e-3)
    parser.add_argument('--alpha_0', type=float, default=0.5)
    parser.add_argument('--lrs_alpha', type=float, default=0.005)
    parser.add_argument('--lr_alpha', type=float, default=0.998)
    parser.add_argument('--act', type=str, default='ReLU')
    parser.add_argument('--ntrain', type=int, default=200)
    parser.add_argument('--nvalid', type=int, default=50)
    parser.add_argument('--ntest', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--beta', type=float, default=100.0)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--integration', type=str, default='quadrature', choices=['quadrature', 'riemann'])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_activation(name: str):
    if name == 'ReLU':
        return torch.nn.ReLU
    if name == 'GELU':
        return torch.nn.GELU
    if name == 'Tanh':
        return torch.nn.Tanh
    if name == 'Softplus':
        return torch.nn.Softplus
    raise ValueError(f'Unsupported activation: {name}')


def build_model(args: argparse.Namespace, device: torch.device):
    act_fun = get_activation(args.act)
    if args.layer_info is not None:
        g_layer, k_layer = parse_layer_info_4parts(args.layer_info)
        layer_str = args.layer_info
    else:
        g_layer = parse_layer_info(args.g_layer)
        k_layer = parse_layer_info(args.k_layer, input_dim=2)
        layer_str = f'{args.k_layer}_{args.g_layer}'
    model = E_GCL_GKN(
        k_layer,
        g_layer,
        act_fun,
        init_alpha=args.alpha_0,
        validate_quadrature=False,
        use_singular=True,
        integration=args.integration,
    ).to(device)
    return model, layer_str


def resolve_paths(current_dir: Path, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    package_dir = current_dir.parent
    wave_csv = Path(args.wave_csv) if args.wave_csv else package_dir / 'data' / 'simple_wavenumber_per_sample.csv'
    data_dir = Path(args.data_dir) if args.data_dir else package_dir / 'data'
    reports_dir = current_dir / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    return wave_csv.resolve(), data_dir.resolve(), reports_dir.resolve()


def resolve_split_sizes(total_samples: int, requested_train: int, requested_valid: int, requested_test: int) -> tuple[int, int, int]:
    requested = np.array([requested_train, requested_valid, requested_test], dtype=float)
    if requested.sum() <= total_samples:
        return requested_train, requested_valid, requested_test
    if total_samples < 3:
        raise ValueError(f'Need at least 3 samples, got {total_samples}')

    ratios = requested / requested.sum()
    counts = np.floor(ratios * total_samples).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > total_samples:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < total_samples:
        counts[np.argmin(counts)] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def load_group_spec(groups: list[dict[str, object]], group_name: str) -> dict[str, object]:
    for group in groups:
        if group['group_name'] == group_name:
            return group
    raise KeyError(f'Unknown group: {group_name}')


def load_group_dataset(group_spec: dict[str, object], data_dir: Path):
    coords_reference = None
    disps_list = []
    forces_list = []
    file_paths = []

    for file_name in group_spec['file_names']:
        file_path = data_dir / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f'Missing data file: {file_path}')
        reader = MatReader(str(file_path))
        coords = reader.read_field('coords').float()
        disps = reader.read_field('disps').float()
        forces = reader.read_field('forces').float()
        if coords_reference is None:
            coords_reference = coords
        elif not torch.allclose(coords_reference, coords):
            raise ValueError(f'Coordinate mismatch detected in {file_name}')
        disps_list.append(disps)
        forces_list.append(forces)
        file_paths.append(str(file_path))

    if coords_reference is None:
        raise ValueError('No files found for the selected group')

    data_u = torch.cat(disps_list, dim=0)
    data_f = torch.cat(forces_list, dim=0)
    return coords_reference, data_u, data_f, file_paths


def prepare_normalized_data(data_x: torch.Tensor, data_u: torch.Tensor, data_f: torch.Tensor):
    m_fact = 3.01
    S = 21
    cond_f = torch.abs(data_x.view(S, S, 2)[0, :, 0]) <= (35 + 1e-10)
    import os as _os
    data_x = data_x * float(_os.environ.get("SC_XU",0.1))
    data_u = data_u * float(_os.environ.get("SC_XU",0.1))
    data_f = -data_f * float(_os.environ.get("SC_F",0.1))
    dx = float((data_x[1, 0] - data_x[0, 0]).item())
    delta = float(m_fact * dx)
    interior_mask = (cond_f.unsqueeze(1) & cond_f.unsqueeze(0)).flatten()
    return data_x, data_u, data_f, cond_f, dx, delta, interior_mask, S


def build_datasets(data_x, data_u, data_f, split_indices, edge_index, delta, dx, S):
    datasets = {}
    for split_name, indices in split_indices.items():
        datasets[split_name] = [
            Data(
                x=data_x,
                u=data_u[index, :, :],
                f=data_f[index, :, :],
                edge_index=edge_index,
                edge_attr=None,
                delta=delta,
                dx=dx,
                S=S,
            )
            for index in indices
        ]
    return datasets


def crop_force_field(field: torch.Tensor, cond_f: torch.Tensor) -> torch.Tensor:
    field = field[:, cond_f, :, :]
    field = field[:, :, cond_f, :]
    return field


def evaluate_model(model, loader, quad_points_weights, edge_weight_indices, cond_f, myloss, batch_size, S, device):
    total_l2 = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch, quad_points_weights=quad_points_weights, edge_weight_indices=edge_weight_indices)
            out = out.view(batch_size, S, S, 2)
            out = crop_force_field(out, cond_f)
            f_gt = batch.f.view(batch_size, S, S, 2)
            f_gt = crop_force_field(f_gt, cond_f)
            total_l2 += myloss.rel_sqrt(out.view(batch_size, -1), f_gt.view(batch_size, -1)).item()
    return total_l2 / max(len(loader.dataset), 1)


def plot_kernel_curves(model, dx, delta, base_dir: Path, tag: str, lambda_min=0.5, lambda_max=1.5):
    device = next(model.parameters()).device
    xi_norm = torch.linspace(dx, delta, 200, device=device)
    xi_2d = torch.stack([xi_norm, torch.zeros_like(xi_norm)], dim=1)
    k_vals = model.k(xi_2d)
    k0 = model.k(torch.zeros(1, 2, device=device))
    k_vals = (k_vals / (k0 + 1e-9)).detach().cpu().reshape(-1)
    xi_cpu = xi_norm.detach().cpu().reshape(-1)
    learned_alpha = float(model.get_alpha().item())
    kernel_vals = k_vals * torch.pow(xi_cpu, -learned_alpha)

    lambda_vals = torch.linspace(lambda_min, lambda_max, 200, device=device).reshape(-1, 1)
    g_vals = model.signed_g_difference(lambda_vals)
    g_vals = g_vals.detach().cpu().reshape(-1)
    lambda_cpu = lambda_vals.detach().cpu().reshape(-1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].plot(xi_cpu, k_vals, linewidth=2.2, label='k(xi) along (r, 0)')
    axes[0].set_xlabel(r'$|\xi|$')
    axes[0].set_ylabel(r'$k(\xi)$')
    axes[0].set_title('Learned k')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(lambda_cpu, g_vals, linewidth=2.2, color='darkorange', label='g(lambda) - g(1)')
    axes[1].set_xlabel(r'$\lambda$')
    axes[1].set_ylabel(r'$g(\lambda) - g(1)$')
    axes[1].set_title('Learned g')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(xi_cpu, kernel_vals, linewidth=2.2, color='seagreen', label=r'$k(\xi)|\xi|^{-\alpha}$')
    axes[2].set_xlabel(r'$|\xi|$')
    axes[2].set_ylabel(r'$k(\xi)|\xi|^{-\alpha}$')
    axes[2].set_title(f'Kernel, alpha={learned_alpha:.4f}')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(base_dir / f'{tag}_g_k_kernel.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def compute_g_sign_diagnostics(model, device: torch.device, lambda_min=0.5, lambda_max=1.5, n_points=400) -> dict[str, float]:
    lambda_vals = torch.linspace(lambda_min, lambda_max, n_points, device=device).reshape(-1, 1)
    with torch.no_grad():
        g_vals = model.signed_g_difference(lambda_vals)
        below = lambda_vals < 1.0
        above = lambda_vals > 1.0
        neg_viol = (g_vals[below] >= 0.0).float().mean().item() if below.any() else 0.0
        pos_viol = (g_vals[above] <= 0.0).float().mean().item() if above.any() else 0.0
        return {
            'g_sign_violation_ratio_low': float(neg_viol),
            'g_sign_violation_ratio_high': float(pos_viol),
            'g_sign_violation_ratio_total': float(model.g_sign_violation_rate(lambda_vals).item()),
        }


def append_training_record(record_path: Path, summary: dict[str, object]) -> None:
    with record_path.open('a', encoding='utf-8') as handle:
        handle.write(
            'Group:{group_name}, Data:{data_tag}, act:{act}, alpha_0:{alpha_0}, layer:{layer_str}, '
            'lrs:[{lrs}], lr:{lr}, lrs_alpha:{lrs_alpha}, lr_alpha:{lr_alpha}, w_d:{wds}, '
            'losss: {best_train_loss:.4e}, {best_valid_loss:.4e}, {best_test_loss:.4e}, '
            'best_ep={best_epoch}, alpha={best_alpha:.4e}\n'.format(**summary)
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = Path(__file__).resolve().parent
    wave_csv, data_dir, reports_dir = resolve_paths(current_dir, args)

    metadata = load_wave_number_metadata(str(wave_csv))
    groups = build_wave_number_groups(metadata, num_groups=args.num_groups)
    group_report_dir = reports_dir / 'wave_groups'
    export_wave_group_manifest(groups, str(group_report_dir))
    group_spec = load_group_spec(groups, args.group_name)

    data_x, data_u, data_f, file_paths = load_group_dataset(group_spec, data_dir)
    data_x, data_u, data_f, cond_f, dx, delta, interior_mask, S = prepare_normalized_data(data_x, data_u, data_f)
    total_samples = int(data_u.size(0))
    ntrain, nvalid, ntest = resolve_split_sizes(total_samples, args.ntrain, args.nvalid, args.ntest)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(total_samples)
    train_indices = torch.from_numpy(indices[:ntrain]).long()
    valid_indices = torch.from_numpy(indices[ntrain:ntrain + nvalid]).long()
    test_indices = torch.from_numpy(indices[ntrain + nvalid:ntrain + nvalid + ntest]).long()

    model, layer_str = build_model(args, device)
    total_params = sum(param.numel() for param in model.parameters())

    quad_points_weights, edge_weight_indices, _ = compute_and_reorder_quadrature_weights(
        x_train=data_x,
        delta=delta,
        dx=dx,
        alpha=args.alpha_0,
        p_order=5,
        S=S,
        device=device,
        validate_quad_match=True,
        interior_mask=interior_mask,
    )
    edge_index, _ = generate_quadrature_ordered_connectivity(data_x, delta, dx, S, interior_mask=interior_mask)

    split_indices = {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices,
    }
    datasets = build_datasets(data_x, data_u, data_f, split_indices, edge_index, delta, dx, S)
    train_loader = DataLoader(datasets['train'], batch_size=1, shuffle=True)
    valid_loader = DataLoader(datasets['valid'], batch_size=1, shuffle=False)
    test_loader = DataLoader(datasets['test'], batch_size=1, shuffle=False)

    run_tag = sanitize_token(args.run_tag) if args.run_tag else ''
    result_name = (
        f"{run_tag + '_' if run_tag else ''}group_{sanitize_token(args.group_name)}_w{group_spec['wave_number_min']:.3f}_{group_spec['wave_number_max']:.3f}_"
        f"k{sanitize_token(args.k_layer)}_g{sanitize_token(args.g_layer)}_{sanitize_token(args.act)}_"
        f"ntrain{ntrain}_wd{args.wds}_lrs{args.lrs}_lr{args.lr}_"
        f"lrs_alpha{args.lrs_alpha}_lr_alpha{args.lr_alpha}_alpha0{args.alpha_0}"
    )
    base_dir = current_dir / args.results_dir / sanitize_token(result_name)
    base_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_dir / 'summary.json'
    model_path = base_dir / 'model.ckpt'

    if args.test:
        if not model_path.is_file():
            raise FileNotFoundError(f'Missing checkpoint: {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        plot_kernel_curves(model, dx, delta, base_dir, f'{args.group_name}_best')
        return

    myloss = LpLoss(size_average=False)
    optimizer = torch.optim.AdamW(
        [
            {'params': [param for name, param in model.named_parameters() if name != 'alpha_raw'], 'lr': args.lrs, 'weight_decay': args.wds},
            {'params': [model.alpha_raw], 'lr': args.lrs_alpha, 'weight_decay': args.wds},
        ]
    )
    lr_main = [args.lr, 0.998]
    lr_alpha = [args.lr_alpha, 0.999]
    lambda_fn_main = lambda epoch: lr_main[0] ** epoch if epoch < args.epochs * 0.3 else lr_main[0] ** (args.epochs * 0.3) * lr_main[1] ** (epoch - args.epochs * 0.3)
    lambda_fn_alpha = lambda epoch: lr_alpha[0] ** epoch if epoch < args.epochs * 0.3 else lr_alpha[0] ** (args.epochs * 0.3) * lr_alpha[1] ** (epoch - args.epochs * 0.3)

    ttrain, tvalid, ttest = [], [], []
    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_test_loss = float('inf')
    best_epoch = -1
    best_alpha = float(model.get_alpha().item())
    early_stop = 0
    train_start = default_timer()

    for epoch in range(args.epochs):
        model.train()
        optimizer.param_groups[0]['lr'] = args.lrs * lambda_fn_main(epoch)
        optimizer.param_groups[1]['lr'] = args.lrs_alpha * lambda_fn_alpha(epoch)
        if args.integration == 'quadrature' and epoch % 50 == 0 and epoch > 0:
            quad_points_weights = compute_quadrature_weights_simple(delta, dx, model.get_alpha().item(), 5, device)

        epoch_train_err = 0.0
        epoch_train_loss = 0.0
        epoch_t1 = default_timer()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out, l_match = model(batch, quad_points_weights=quad_points_weights, edge_weight_indices=edge_weight_indices)
            out = out.view(1, S, S, 2)
            out = crop_force_field(out, cond_f)
            f_gt = batch.f.view(1, S, S, 2)
            f_gt = crop_force_field(f_gt, cond_f)
            loss = args.beta * myloss(out.view(1, -1), f_gt.view(1, -1)) + l_match
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_train_err += myloss.rel_sqrt(out.view(1, -1), f_gt.view(1, -1)).item()

        train_err = epoch_train_err / ntrain
        train_loss = epoch_train_loss / ntrain
        ttrain.append([epoch, train_err])
        model.eval()
        valid_err = evaluate_model(model, valid_loader, quad_points_weights, edge_weight_indices, cond_f, myloss, 1, S, device)
        test_err = evaluate_model(model, test_loader, quad_points_weights, edge_weight_indices, cond_f, myloss, 1, S, device)
        tvalid.append([epoch, valid_err])
        ttest.append([epoch, test_err])

        epoch_t2 = default_timer()
        if valid_err < best_valid_loss:
            best_train_loss = train_err
            best_valid_loss = valid_err
            best_test_loss = test_err
            best_epoch = epoch
            best_alpha = float(model.get_alpha().item())
            torch.save(model.state_dict(), model_path)
            early_stop = 0
            print(
                f'>> group={args.group_name}, epoch [{epoch + 1:>{len(str(args.epochs))}d}/{args.epochs}], '
                f'runtime: {(epoch_t2 - epoch_t1):.2f}s, loss: {train_loss:.4e}, train err: {train_err:.4e}, '
                f'valid err: {valid_err:.4e}, test err: {test_err:.4e}, alpha: {best_alpha:.4e}'
            )
        else:
            early_stop += 1
            print(
                f'>> group={args.group_name}, epoch [{epoch + 1:>{len(str(args.epochs))}d}/{args.epochs}], '
                f'runtime: {(epoch_t2 - epoch_t1):.2f}s, loss: {train_loss:.4e}, train err: {train_err:.4e}, '
                f'best valid: {best_valid_loss:.4e}, best alpha: {best_alpha:.4e}'
            )
        if early_stop > args.patience:
            break

    np.savetxt(base_dir / 'loss_train.txt', np.asarray(ttrain))
    np.savetxt(base_dir / 'loss_valid.txt', np.asarray(tvalid))
    np.savetxt(base_dir / 'loss_test.txt', np.asarray(ttest))

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    plot_kernel_curves(model, dx, delta, base_dir, f'{args.group_name}_best')
    g_sign_diag = compute_g_sign_diagnostics(model, device=device)

    elapsed = default_timer() - train_start
    summary = {
        'group_name': args.group_name,
        'run_tag': str(args.run_tag),
        'group_wave_number_min': float(group_spec['wave_number_min']),
        'group_wave_number_max': float(group_spec['wave_number_max']),
        'group_wave_numbers': [float(value) for value in group_spec['wave_numbers']],
        'group_file_count': int(group_spec['file_count']),
        'group_sample_count': int(group_spec['sample_count']),
        'group_file_indices': [int(value) for value in group_spec['file_indices']],
        'group_file_names': list(group_spec['file_names']),
        'data_tag': 'periodic_wave_group',
        'layer_str': layer_str,
        'k_layer': args.k_layer,
        'g_layer': args.g_layer,
        'layer_info': args.layer_info,
        'act': args.act,
        'alpha_0': float(args.alpha_0),
        'lrs': float(args.lrs),
        'lr': float(args.lr),
        'lrs_alpha': float(args.lrs_alpha),
        'lr_alpha': float(args.lr_alpha),
        'wds': float(args.wds),
        'beta': float(args.beta),
        'epochs_requested': int(args.epochs),
        'epochs_ran': int(len(ttrain)),
        'ntrain': int(ntrain),
        'nvalid': int(nvalid),
        'ntest': int(ntest),
        'total_samples': int(total_samples),
        'best_train_loss': float(best_train_loss),
        'best_valid_loss': float(best_valid_loss),
        'best_test_loss': float(best_test_loss),
        'best_epoch': int(best_epoch),
        'best_alpha': float(best_alpha),
        'final_alpha': float(model.get_alpha().item()),
        'seed': int(args.seed),
        'integration': str(args.integration),
        'dx': float(dx),
        'delta': float(delta),
        'result_dir_name': base_dir.name,
        'result_dir': str(base_dir),
        'wave_csv': str(wave_csv),
        'data_dir': str(data_dir),
        'source_files': file_paths,
        'total_params': int(total_params),
        'runtime_seconds': float(elapsed),
    }
    summary.update(g_sign_diag)
    write_json(summary, summary_path)
    append_training_record(base_dir.parent / 'training_record.txt', summary)
    print(f'>> Training complete for group {args.group_name}. Best valid loss: {best_valid_loss:.4e}')


if __name__ == '__main__':
    main()
