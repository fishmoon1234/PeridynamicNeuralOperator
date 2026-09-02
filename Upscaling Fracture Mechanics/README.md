# Upscaling Fracture Mechanics

A mesoscale neural operator (MNO) trained on coarse-grained molecular
dynamics (MD), with its training package, total-force validation
against MD, and a downstream dynamic-fracture application.

```text
Upscaling Fracture Mechanics/
  training/                        training code and MD dataset
    MD_code/                       entry point, model, utilities
    MD_data/                       24 MD data files + wave-number CSV
    TRAINING_SPEC.md               detailed data/architecture/training spec
  models/nrm_n256_lr3e4/           the optimal checkpoint (model.ckpt, summary.json)
  validation/                      quasistatic pull tests; total force compared against MD
  downstream on crack branching/   dynamic crack propagation and branching in a pre-notched plate
```

The archived environment used Python 3.10 with PyTorch 2.5.1, NumPy
2.0.1, SciPy 1.15.2, and PyTorch Geometric 2.7.0 (plus Matplotlib and
scikit-learn).

## Training

The data are coarse-grained MD fields on a 21 by 21 grid (24 MAT
files, 800 displacement-force snapshots; low wave-number group, split
200/50/20 at snapshot level with seed 100).  Each sample maps node
coordinates and a displacement field to the MD force field; `g`, `k`,
and the exponent `alpha` have no direct labels and are learned through
the force-field loss, evaluated on the interior 15 by 15 nodes.
Training is in **native MD units** (no data rescaling; only the force
sign flip), so `dx = 5.0` lu and `delta = 3.01 dx = 15.05` lu.

Per bond, the operator is

```text
lambda = 1 + (||xi + eta|| - ||xi||) / (||xi|| + 1e-9)
g(lambda) = sign(lambda - 1) * softplus(MLP_g(lambda - 1))
phi(lambda, xi) = g(lambda) * k(xi/delta) * (||xi||/delta)^(-alpha)
```

with Riemann weights `dx^2`.  `MLP_g` (1->256^4->1) and `MLP_k`
(2->256^4->1), ReLU, 396,547 parameters total; AdamW with main
learning rate `3e-4`, alpha learning rate `2e-3`, weight decay `5e-4`,
500 epochs, early-stopping patience 100, data-loss weight 100.  The
checkpoint is selected by the lowest validation relative L2 error
(epoch 433).  Re-evaluating the frozen checkpoint through this
pipeline gives the recorded errors

```text
train / valid / test rel-L2 = 0.091885 / 0.080247 / 0.079429,   alpha = 0.109196
```

(see `training/TRAINING_SPEC.md` for the full specification).  From
the `training/` directory:

```bash
python3 MD_code/train.py --dry-run       # check command and bundled data
python3 MD_code/train.py --smoke-test    # one-epoch code-path test
python3 MD_code/train.py                 # full 500-epoch training
python3 MD_code/verify_pretrained.py     # reproduce the recorded errors
                                         # from ../models/nrm_n256_lr3e4
```

## Validation: total-force benchmarks

Quasistatic displacement-controlled pull tests on a 2L by 4L domain:
pristine (no pre-crack) in `nofracture_estimate_force.py`, and with a
horizontal pre-crack of length L at the mid-plane y = 0 in
`fracture_estimate_force.py`.  A displacement u = (0, +-v t) with pull
rate v is prescribed on the top and bottom of the domain.  We record
the total force transmitted across y = 0 and compare it against MD
simulations of the same setting.

Both fracture and non-fracture cases are based on the staggered
("middle") grid -- node rows at y = +-dx/2, ..., so the measurement
planes y = 0, +-L (and the pre-notch) lie BETWEEN grid layers.  Both
drivers record the transmitted force across y = 0 and y = +-L in
physical MD force units (`force.txt`), plus the continuum-elasticity
estimate E*eps_yy*(2LH).  Recommended commands (from `validation/`):

No fracture (pristine plate, learned g):

```bash
python nofracture_estimate_force.py \
  --model-dir ../models/nrm_n256_lr3e4 --unit-s 1.0 --precision float32 \
  --unit-sigma-f 1.0 --g-rule "learned" --L 50 --dx 5 --pull-rate 0.025 \
  --gamma 1 --dt 10. --n-steps 20 --newton-tol 1e-4 --verbose-newton \
  --snapshot-interval 1 --newton-max-iter 30 \
  --out-dir results_force_L50_nrm_n256_lr3e4_nofracture_middle
```

With fracture (pre-notched plate, learned g + fracture rule, 
100x200 domain, trained horizon delta=15.05, dx=5.0):

```bash
python fracture_estimate_force.py \
  --model-dir ../models/nrm_n256_lr3e4 --unit-s 1.0 --precision float32 \
  --unit-sigma-f 1.0 --L 50 --dx 5 --pull-rate 0.025 \
  --gamma 1 --dt 1. --n-steps 160 --newton-tol 1e-4 --verbose-newton \
  --snapshot-interval 1 --newton-max-iter 30 \
  --out-dir results_force_L50_nrm_n256_lr3e4_middle
```

With fracture (pre-notched plate, learned g + fracture rule, 
200x400 domain, trained horizon delta=15.05, dx=5.0):

```bash
python fracture_estimate_force.py \
  --model-dir ../models/nrm_n256_lr3e4 --unit-s 1.0 --precision float32 \
  --unit-sigma-f 1.0 --L 100 --dx 5 --pull-rate 0.05 \
  --gamma 1 --dt 1. --n-steps 160 --newton-tol 1e-4 --verbose-newton \
  --snapshot-interval 1 --newton-max-iter 30 \
  --out-dir results_force_L100_nrm_n256_lr3e4_middle
```

With fracture (pre-notched plate, learned g + fracture rule, 
200x400 domain, horizon scaled up to delta=30.1, dx=10.0):

```bash
python fracture_estimate_force.py \
  --model-dir ../models/nrm_n256_lr3e4 --unit-s 1.0 --precision float32 \
  --unit-sigma-f 1.0 --L 100 --dx 10 --pull-rate 0.05 \
  --gamma 2 --dt 1. --n-steps 160 --newton-tol 1e-4 --verbose-newton \
  --snapshot-interval 1 --newton-max-iter 30 \
  --out-dir results_force_L100_delta30_nrm_n256_lr3e4_middle
```

Reference results for this model (measured MD comparisons):

1. The no-fracture learned-g force curve matches the MD reference to 3.8%
relative L2 (the elasticity estimate to 1.8% with linearized g). MD result 
is recorded in MD_nofracture_totalforce.dat. The MNO result with the 
learned g is in MNO_force_nofracture_L50.txt; the linearized-g result 
is in MNO_force_nofracture_linearg_L50.txt.

2. The pre-notched test peaks at 583 fu at t = 92 vs. the measured MD
572 fu at t = 96, with matching brittle collapse. MD result is recorded 
in MD_L50_totalforce.dat. MNO result is in MNO_force_fracture_L50.txt.

3. The pre-notched test peaks at 923 fu at t = 69 vs. the measured MD
928 fu at t = 67, with matching brittle collapse. MD result is recorded 
in MD_L100_totalforce.dat. MNO result is in MNO_force_fracture_L100.txt. 
The MNO result with rescaled delta (from 15.05 to 30.10) is stored in
MNO_force_fracture_L100_delta30.txt, which peaks at 864 fu at t = 68.

A post-processing and comparison script (reproducing the two figures
in this folder from the bundled data files) is provided by running:

```bash
python plot_lr3e4_summary.py
```

## Downstream: crack branching

As a downstream application, we consider crack propagation and
branching in a prototypical brittle-fracture example: a pre-notched
thin rectangular plate subject to tensile loads on its top and bottom.
The plate measures 1 $`\mu m`$ by 0.4 $`\mu m`$ (= 10000 lu by ~4000 lu)
with an initial edge crack of length 0.5 $`\mu m`$.
A constant total load $`T`$ (`--traction`, in fu) is applied on the top
and bottom of the sample from time zero, equivalent to a tensile load
$`\sigma = T/(W H) \approx 1.07 \times 10^{-4}\, T\ \mathrm{fu/lu^2} \approx 1.07 \times 10^{3}\, T\ \mathrm{N/m^2}`$,
where $`W = 10000`$ lu is the plate width (the length of the top/bottom
boundary) and $`H = 0.935`$ lu is the slab thickness.
All other boundaries, including the new surfaces created by the running
crack, are treated as free surfaces.
The density $`\rho`$ (`--rho`) is given in MD mass units per
$`\mathrm{lu}^3`$; one such unit corresponds to
$`1000\ \mathrm{kg/m^3}`$, so $`\rho = 8.0`$ means
$`8000\ \mathrm{kg/m^3}`$.

`fracture_sim_implicit_bodyload.py` solves this dynamic fracture
problem with an implicit Newmark scheme, using Newton's method as the
nonlinear solver at each time step.
The solver is scaled up by a factor $`\gamma = 10`$, giving an
effective horizon of 150.5 lu on a grid of spacing 50 lu.
We have tested total loads from $`T = 1000`$ to $`20000`$ fu at
$`\rho = 8.0`$ (in the physical range of polymers); movies for the
intermediate case $`T = 10000`$ fu (traction10000_damage.mov,
traction10000_displacement.mov) are also included.

The command below reproduces the exemplar case $`T = 1000`$ fu
($`= 10^{-4}\ \mu N`$, $`\sigma \approx 1.07 \times 10^{6}\ \mathrm{N/m^2}`$),
with exemplar results provided in traction1000_damage.mov and
traction1000_displacement.mov.

```bash
python -u fracture_sim_implicit_bodyload.py \
  --model-dir ../models/nrm_n256_lr3e4 --unit-s 1.0 --precision float32 --unit-sigma-f 1.0 \
  --Nx 201 --Ny 81 --dx 50.0 --gamma 10 \
  --traction 1000 --rho 8.0 --rescale 1. --dt 0.1 --n-steps 50000 \
  --bondsoft-decay 10000 --protect-boundary \
  --newton-tol 1e-4 --newton-max-iter 10 --cg-max-iter 50 \
  --max-stab-passes 10 \
  --print-interval 10 --snapshot-interval 100 --verbose-newton \
  --out-dir results_bodyload_prod_lr3e4_t1000p0_rho8p0
```


The command below reproduces the exemplar case $`T = 20000`$ fu
($`= 2\times 10^{-3}\ \mu N`$, $`\sigma \approx 2.14 \times 10^{7}\ \mathrm{N/m^2}`$),
with exemplar results provided in traction20000_damage.mov and
traction20000_displacement.mov.

```bash
python -u fracture_sim_implicit_bodyload.py \
  --model-dir ../models/nrm_n256_lr3e4 --unit-s 1.0 --precision float32 --unit-sigma-f 1.0 \
  --Nx 201 --Ny 81 --dx 50.0 --gamma 10 \
  --traction 20000 --rho 8.0 --rescale 1. --dt 0.1 --n-steps 50000 \
  --bondsoft-decay 10000 --protect-boundary \
  --newton-tol 1e-4 --newton-max-iter 10 --cg-max-iter 50 \
  --max-stab-passes 10 \
  --print-interval 10 --snapshot-interval 100 --verbose-newton \
  --out-dir results_bodyload_prod_lr3e4_t20000p0_rho8p0
```
