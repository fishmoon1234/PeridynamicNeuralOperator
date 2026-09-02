#!/usr/bin/env python3
"""Train the MD-based nrm_lr3e4 (nrm_n256_lr3e4) model: unscaled data (s=1, sigma_f=1), 256-wide networks, main lr 3e-4."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "MD_code"
DATA_DIR = ROOT / "MD_data"
WAVE_CSV = DATA_DIR / "simple_wavenumber_per_sample.csv"

MODEL = {
    "k_layer": "256_4",
    "g_layer": "256_4",
    "activation": "ReLU",
    "alpha_raw_initial": 0.4,
    "integration": "riemann",
}

DATA_SPLIT = {
    "train": 200,
    "validation": 50,
    "test": 20,
    "seed": 100,
}

OPTIMIZATION = {
    "epochs": 500,
    "patience": 100,
    "main_learning_rate": 3.0e-4,
    "main_decay": 0.995,
    "alpha_learning_rate": 2.0e-3,
    "alpha_decay": 0.999,
    "weight_decay": 5.0e-4,
    "data_loss_weight": 100.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Run one epoch with a 2/1/1 split")
    return parser.parse_args()


def validate_data() -> None:
    if not WAVE_CSV.is_file():
        raise FileNotFoundError(WAVE_CSV)
    mat_files = sorted(DATA_DIR.glob("*.mat"))
    if len(mat_files) != 24:
        raise RuntimeError(f"Expected 24 MAT files in {DATA_DIR}, found {len(mat_files)}")


def build_command(args: argparse.Namespace) -> list[str]:
    ntrain = 2 if args.smoke_test else DATA_SPLIT["train"]
    nvalid = 1 if args.smoke_test else DATA_SPLIT["validation"]
    ntest = 1 if args.smoke_test else DATA_SPLIT["test"]
    epochs = 1 if args.smoke_test else OPTIMIZATION["epochs"]
    patience = 1 if args.smoke_test else OPTIMIZATION["patience"]
    run_tag = "n256_lr3e4_smoke" if args.smoke_test else "nrm_n256_lr3e4"
    output_dir = args.output_dir.resolve()

    return [
        sys.executable,
        "-u",
        str(CODE_DIR / "group_train.py"),
        "--run-tag", run_tag,
        "--group-name", "low",
        "--num-groups", "3",
        "--wave-csv", str(WAVE_CSV),
        "--data-dir", str(DATA_DIR),
        "--reports-dir", str(output_dir / "reports"),
        "--results-dir", str(output_dir),
        "--k_layer", MODEL["k_layer"],
        "--g_layer", MODEL["g_layer"],
        "--act", MODEL["activation"],
        "--lrs", str(OPTIMIZATION["main_learning_rate"]),
        "--lr", str(OPTIMIZATION["main_decay"]),
        "--lrs_alpha", str(OPTIMIZATION["alpha_learning_rate"]),
        "--lr_alpha", str(OPTIMIZATION["alpha_decay"]),
        "--alpha_0", str(MODEL["alpha_raw_initial"]),
        "--wds", str(OPTIMIZATION["weight_decay"]),
        "--ntrain", str(ntrain),
        "--nvalid", str(nvalid),
        "--ntest", str(ntest),
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--beta", str(OPTIMIZATION["data_loss_weight"]),
        "--seed", str(DATA_SPLIT["seed"]),
        "--integration", MODEL["integration"],
    ]


def main() -> None:
    args = parse_args()
    validate_data()
    command = build_command(args)
    environment = os.environ.copy()
    # nrm_lr3e4 trains on UNSCALED data: s = 1 (no x,u rescale) and
    # sigma_f = 1 (force keeps only the sign flip applied by the loader),
    # so dx = 5.0 and delta = 3.01*dx = 15.05 in native lu.
    environment["SC_XU"] = "1.0"
    environment["SC_F"] = "1.0"
    environment.setdefault("MPLBACKEND", "Agg")

    print(f"SC_XU=1.0 SC_F=1.0 {shlex.join(command)}", flush=True)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, env=environment, check=True)


if __name__ == "__main__":
    main()
