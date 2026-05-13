1D Blatz-Ko analytical data generation
======================================

Examples used in the paper:

ex22  - g(x) = x * exp(-50 * x**2), mixed-frequency u (J=40)
        Smooth training dataset (500 samples) for the one-/two-phase
        solver comparison.

ex32  - g(x) = x * exp(-50 * x**2) * (delta - |x|), mixed-frequency u (J=40)
        Smooth dataset (500 samples) for the Ex-II convergence study.

ex32_sample  - out-of-distribution sample variant of ex32
        Generated from the same model with shifted parameter
        ranges; used for OOD evaluation.

ex35  - g(x) = pi * (x - x**(-3)) + sin(pi * x),
        k(x) = exp(-50 * x**2) * (delta - |x|), mixed-frequency u (J=40)
        Manufactured dataset (400 samples) for the MGN / MLP / ICNN
        robustness comparison.

ex35_ood  - out-of-distribution test set for ex35 (100 samples)
        Lambda range extended beyond the training coverage.

ex37  - g(x) = x - x**(-3), k(x) = 2c * cos(pi * x), mixed-frequency u (J=40)
        Smooth dataset (400 samples) for the Ex-I convergence study.

ex37_sample / ex37_sample_10 / ex37_sample_largestrain
        Out-of-distribution sample variants of ex37 (100 samples each)
        used for OOD evaluation under various stretch ranges.

ex38  - same g, k as ex37; the displacement is augmented with a
        Heaviside step (C = 0.01 at x_0 = 0.49).
        Discontinuous dataset (400 samples) for the 1D
        discontinuous-data experiment.

filter_data_ex35_ood.py  - post-processing filter applied to the
        ex35_ood dataset.

plot_data.py / plot_data_ex38.py  - visualisation utilities.

utilities_INO_PD.py  - shared data-loading utilities.

run.sh  - example launcher for a generation run.
