# `nrm_lr3e4` (`nrm_n256_lr3e4`) training specification

## Data

The dataset contains 24 coarse-grained MD files and 800 displacement-force snapshots. Each supervised sample consists of the fixed node coordinates, a displacement field, and its corresponding MD force field.

| Item | Value |
| --- | --- |
| Input | Node coordinates `coords` and displacement field `disps` |
| Target | MD force field `forces` |
| Grid | 21 by 21 nodes |
| Scaling | none: `x <- x`, `u <- u`, `f <- -f` (native MD units; only the sign flip) |
| Grid spacing | `dx = 5.0` |
| Horizon | `delta = 3.01 dx = 15.05` |
| Data group | Low wave number: 1.000 to 3.162 |
| Split | 200 train, 50 validation, 20 test; random seed 100 |
| Unused snapshots | 530 |
| Batch size | 1 |

The loss is evaluated on the interior 15 by 15 nodes. The split is performed at snapshot level rather than by source file.

## Model

For each bond, the operator uses

```text
xi = x_j - x_i
eta = u_j - u_i
lambda = 1 + (||xi + eta|| - ||xi||) / (||xi|| + 1e-9)
g(lambda) = sign(lambda - 1) * softplus(MLP_g(lambda - 1))
k(xi) = MLP_k(xi / delta)
phi(lambda, xi) = g(lambda) * k(xi) * (||xi|| / delta)^(-alpha)
```

The force is evaluated with the uniform Riemann weight `dx^2`.

| Component | Architecture | Parameters |
| --- | --- | ---: |
| `MLP_g` | 1 -> 256 -> 256 -> 256 -> 256 -> 1, ReLU hidden activations | 198,145 |
| `MLP_k` | 2 -> 256 -> 256 -> 256 -> 256 -> 1, ReLU hidden activations | 198,401 |
| `alpha_raw` | One learned scalar with `alpha = softplus(alpha_raw) + 1e-6` | 1 |
| Total |  | 396,547 |

The launch value `alpha_0 = 0.4` initializes `alpha_raw`, so the effective initial exponent is approximately `0.913016`.

## Training

| Parameter | Value |
| --- | ---: |
| Optimizer | AdamW |
| Main learning rate | `3e-4` |
| Alpha learning rate | `2e-3` |
| Weight decay | `5e-4` |
| Epochs | 500 |
| Early-stopping patience | 100 |
| Data-loss weight | 100 |
| Integration | Riemann |

The training objective is

```text
L = 100 * ||f_pred - f_true||_2^2 / (||f_true||_2 + 1e-9)^2
    + (log(RMS(g)) - log(RMS(k)))^2.
```

No smoothness penalty is applied to `g` or `k`. The checkpoint is selected by the lowest validation relative L2 error.

The manuscript checkpoint was selected at epoch 433 (`best_epoch` in the archived `summary.json`). Re-evaluating the frozen checkpoint through this pipeline (`MD_code/verify_pretrained.py`) gives train, validation, and test relative L2 errors of `0.091885`, `0.080247`, and `0.079429`; these post-hoc eval-mode values are the recorded errors (the `0.110266` train figure in `summary.json` was accumulated during the epoch while the weights were still being updated). Its learned exponent is `alpha = 0.109196`.
