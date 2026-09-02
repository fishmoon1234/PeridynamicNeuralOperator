# MPNO — Monotone Peridynamic Neural Operator

Code and data accompanying the paper *"Monotone Peridynamic Neural Operator"*.

The framework learns nonlocal constitutive models for hyperelastic materials by parameterising two functions of the deformation invariant and bond vector:

- `g(lambda)` — a monotone scalar energy-density function approximated by a Monotone Gradient Network (MGN);
- `k(xi)` — a peridynamic influence kernel approximated by an MLP.

It is validated on analytical 1D / 2D Blatz-Ko datasets and on coarse-grained molecular-dynamics (MD) data for a graphene-like sheet.

---

## Repository layout

```
MPNO_paper/
|-- BlatzKo_1d/                            1D Blatz-Ko experiments
|   |-- 1d_nonlocal_BlatzKo_analytical_data/   data generation + .mat datasets
|   |-- Cond/                                  condition-number study
|   |-- PNO_MGN_ex22_k1k2/                     one-/two-phase solver comparison
|   |-- PNO_MGN_ex32_{k1, k2, k1k2}/           Ex-II convergence (learn g / k / both)
|   |-- PNO_MGN_ex37_{k1, k2, k1k2}/           Ex-I  convergence (learn g / k / both)
|   |-- PNO_MGN_ex38_{k1, k2, k1k2}/           1D discontinuous dataset
|   |-- ex35/                                  MGN vs MLP vs ICNN: training scripts
|   |-- PNO_{ICNN, MGN, NN}_ex35_k1/           MGN vs MLP vs ICNN: multi-seed sweeps
|   |-- PNO_MGN_ex35_k1k2_cMGN/                ex35 joint g+k learning
|   |-- PNO_SB_ex35_k1k2/                      State-based PNO baseline on ex35
|
|-- BlatzKo_2d/                            2D Blatz-Ko experiments
|   |-- generate_data/                         data generation + .mat dataset
|   |-- PNO_MGN_ex5_g/                         learn g (kernel fixed)
|   |-- PNO_MGN_ex5_k/                         learn k (energy fixed)
|
|-- ex35_plot_u_comparsion/                Plotting utilities for the ex35 figure
|
|-- MD/large_strain_241215/                Molecular-dynamics experiments
|   |-- ProcessData/                           raw MD -> training .mat pipeline
|   |-- PNO_MGN_DBC_xi_cMGNz_twophase_newnormalization3_tuneparas/  MPNO
|   |-- PNO_NN_DBC_xi_cMGNz_twophase_normalization/                  MLP-PNO baseline
|   |-- PNO_SB_normalization/                                        State-based PNO baseline
|
|-- requirements.txt
|-- .gitignore
`-- README.md  (this file)
```

Each leaf experiment directory follows the same pattern:

- `PNO_BB.py`               — main training entry
- `PNO_BB_test_*.py`        — evaluation scripts (model error, b error, u solver, etc.)
- `CG_solver.py`            — linear pre-solver used by the two-phase solver
- `egnn_gcl.py`             — peridynamic graph network layers
- `utilities_INO_PD.py`     — shared utilities (data loader, neighbour sets, normalisation)
- `train.sh`, `test.sh`     — shell wrappers
- `Results/<config>/`       — checkpoints + per-sample error records
- `logs/`                   — stdout/stderr from training and testing

For experiments comparing g architectures (ex35 family), additional files appear:
- `convex_init.py`, `convex_modules.py` — ICNN building blocks

---

## Environment

Tested with Python 3.10 on CUDA 11.x. Install with:

```bash
pip install -r requirements.txt
```

`torch_geometric` may require building extensions (`pyg-lib`, `torch-scatter`, `torch-sparse`) that match your PyTorch + CUDA version; see <https://pytorch-geometric.readthedocs.io>.

---

## Datasets

### 1D Blatz-Ko (analytical)

Pre-generated `.mat` files live under
`BlatzKo_1d/1d_nonlocal_BlatzKo_analytical_data/BlatzKo_data_1d/`
and a duplicate copy of the ex35 data under
`ex35_plot_u_comparsion/DATA/`.

Re-generation:
```bash
cd BlatzKo_1d/1d_nonlocal_BlatzKo_analytical_data
python3 BlatzKo_1d_analytical_u_ex22.py
python3 BlatzKo_1d_analytical_u_ex32.py
python3 BlatzKo_1d_analytical_u_ex35.py
python3 BlatzKo_1d_analytical_u_ex35_ood.py
python3 BlatzKo_1d_analytical_u_ex37.py
python3 BlatzKo_1d_analytical_u_ex38.py
# OOD sample sets for ex32 / ex37
python3 BlatzKo_1d_analytical_u_ex32_sample.py
python3 BlatzKo_1d_analytical_u_ex37_sample.py
python3 BlatzKo_1d_analytical_u_ex37_sample_10.py
python3 BlatzKo_1d_analytical_u_ex37_sample_largestrain.py
```

### 2D Blatz-Ko (analytical)

`BlatzKo_2d/generate_data/analytical_u/DATA/BK_2d_ex5_ndata_100_Nx_33.mat`
is the merged dataset used by ex5. To regenerate (parallel jobs, ten samples per job):
```bash
cd BlatzKo_2d/generate_data/analytical_u
for i in $(seq 0 9); do
    python3 ex5_batch.py --ndata 10 --seed $((1000 + i)) --job_id $i
done
```

### MD (molecular dynamics)

The pre-processed `.mat` files used directly by the training scripts are
included:

- `MD/large_strain_241215/ProcessData/cg-md-250212.mat` — aggregated training set
- `MD/large_strain_241215/ProcessData/test-250328.mat` — aggregated test set
- `MD/large_strain_241215/ProcessData/MD_index_274.mat` — fixed train/valid indices
- `MD/large_strain_241215/ProcessData/cg-md-250212_index_b1_0.02.mat` and
  related `index_*.mat` — sample-selection indices used by the preprocessing pipeline

The **raw MD trajectories** (~2.4 GB of per-step `.dat` files) are not bundled
with the repository because of their size. They are only needed if you want
to re-run `ProcessData/process_data.py` end-to-end. Contact the authors for
the raw archive, or use the external storage link provided alongside the
published paper.

---

## Reproducing a single experiment

Each experiment directory has its own `train.sh` and `test.sh`. From inside
the directory, e.g.:

```bash
cd BlatzKo_1d/PNO_MGN_ex38_k1k2_cMGN
bash train.sh   # writes checkpoints into Results/<config>/, logs into logs/
bash test.sh    # evaluates the trained model and writes *.txt records
```

For the MD experiments:
```bash
cd MD/large_strain_241215/PNO_MGN_DBC_xi_cMGNz_twophase_newnormalization3_tuneparas
bash train.sh
bash test.sh
bash test_disk.sh
```

---

## Algorithm

The two-phase solver used to evaluate learned models (Algorithm 1 of the
paper) is implemented as

1. **Linear pre-solver** — `CG_solver.py`, applied with the linearised
   PD operator on the learned model.
2. **Nonlinear refinement** — `scipy.optimize.fsolve` with the
   Levenberg-Marquardt method, called from each `PNO_BB_test_u.py`.

These two files are present in every experiment directory.

---

## Citation

Citation entry will be added once the paper is published.

## License

To be added.
