#!/usr/bin/env python3
"""Pipeline check: load the archived pre-trained nrm_n256_lr3e4
checkpoint into THIS training code (same data, split, model build, and
error evaluation as group_train.py) and reproduce the summary.json
relative L2 errors: train/valid/test = 0.110266 / 0.080245 / 0.079429.

Note: valid/test are expected to match to float precision (they were
recorded by post-epoch eval-mode passes of the selected checkpoint);
the archived train number was accumulated DURING epoch 433 while the
weights were being updated batch-by-batch, so a post-hoc train-set
evaluation of the frozen checkpoint may differ slightly.

Usage:  python MD_code/verify_pretrained.py [--ckpt PATH]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["SC_XU"] = "1.0"
os.environ["SC_F"] = "1.0"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import group_train as GT
from torch_geometric.loader import DataLoader

DEFAULT_CKPT = (HERE.parent.parent / "models" / "nrm_n256_lr3e4"
                / "model.ckpt")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
cli = ap.parse_args()

args = argparse.Namespace(
    seed=100, num_groups=3, group_name="low",
    ntrain=200, nvalid=50, ntest=20,
    k_layer="256_4", g_layer="256_4", layer_info=None,
    act="ReLU", alpha_0=0.4, integration="riemann",
    wave_csv=str(HERE.parent / "MD_data" / "simple_wavenumber_per_sample.csv"),
    data_dir=str(HERE.parent / "MD_data"),
    reports_dir="results/reports",
)

GT.set_seed(args.seed)
device = torch.device("cpu")
wave_csv, data_dir, reports_dir = GT.resolve_paths(HERE, args)
metadata = GT.load_wave_number_metadata(str(wave_csv))
groups = GT.build_wave_number_groups(metadata, num_groups=args.num_groups)
group_spec = GT.load_group_spec(groups, args.group_name)
data_x, data_u, data_f, _ = GT.load_group_dataset(group_spec, data_dir)
(data_x, data_u, data_f, cond_f, dx, delta,
 interior_mask, S) = GT.prepare_normalized_data(data_x, data_u, data_f)
print(f"dx={dx:g}  delta={delta:g}  samples={data_u.size(0)}")

total = int(data_u.size(0))
ntrain, nvalid, ntest = GT.resolve_split_sizes(total, args.ntrain,
                                               args.nvalid, args.ntest)
rng = np.random.default_rng(args.seed)
idx = rng.permutation(total)
split = {"train": torch.from_numpy(idx[:ntrain]).long(),
         "valid": torch.from_numpy(idx[ntrain:ntrain + nvalid]).long(),
         "test": torch.from_numpy(idx[ntrain + nvalid:
                                      ntrain + nvalid + ntest]).long()}

model, layer_str = GT.build_model(args, device)
print(f"model layers={layer_str}  params="
      f"{sum(p.numel() for p in model.parameters())}")
state = torch.load(cli.ckpt, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()
print(f"loaded checkpoint: {cli.ckpt}")
print(f"alpha = {float(model.get_alpha()):.6f}  (summary: 0.109196)")

qpw, ewi, _ = GT.compute_and_reorder_quadrature_weights(
    x_train=data_x, delta=delta, dx=dx, alpha=args.alpha_0, p_order=5,
    S=S, device=device, validate_quad_match=True,
    interior_mask=interior_mask)
edge_index, _ = GT.generate_quadrature_ordered_connectivity(
    data_x, delta, dx, S, interior_mask=interior_mask)
datasets = GT.build_datasets(data_x, data_u, data_f, split, edge_index,
                             delta, dx, S)
myloss = GT.LpLoss(size_average=False)

expected = {"train": 0.110266, "valid": 0.080245, "test": 0.079429}
for name in ("train", "valid", "test"):
    loader = DataLoader(datasets[name], batch_size=1, shuffle=False)
    err = GT.evaluate_model(model, loader, qpw, ewi, cond_f, myloss,
                            1, S, device)
    print(f"{name:5s}: rel-L2 = {err:.6f}   (summary.json: "
          f"{expected[name]:.6f}, diff {err - expected[name]:+.2e})")
