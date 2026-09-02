MD experiments — molecular-dynamics-derived training data
=========================================================

ProcessData/  - preprocessing pipeline (raw .dat -> aggregated .mat)
    Step 1  process_data.py            extract coords, disps, forces, mass
    Step 2  plot_data.py               sanity-check norms of u and b
    Step 3a choose_data_b.py           select samples by body-force magnitude
    Step 3b choose_large_strain.py     restrict to large-strain regime
    Step 4  filter_data.py             optional small-strain filter
    Step 5  process_test_data.py       prepare test set (different domain)
    Indices for the fixed 274-sample training set are stored in
    MD_index_274.mat (MATLAB-generated).

ProcessData/*.mat
    cg-md-250212.mat                   aggregated training trajectories
    test-250328.mat                    aggregated test trajectories
    cg-md-250212_index_b1_0.02.mat     selected 274-sample subset
    MD_index_274.mat                   fixed train/valid index split
    index_cg-md-250212_*.mat           per-step selection indices

Training directories
    PNO_MGN_DBC_xi_cMGNz_twophase_newnormalization3_tuneparas/
                                       proposed MPNO (g = MGN, k = MLP)
    PNO_NN_DBC_xi_cMGNz_twophase_normalization/
                                       MLP-PNO baseline (g, k both MLPs)
    PNO_SB_normalization/              State-based PNO baseline

Each training directory contains its own train.sh / test.sh /
test_disk.sh and writes to Results/<config>/ and logs/.

The raw MD .dat archive (~2.4 GB) is not bundled here; contact the
authors or use the external storage link from the paper if you need
to re-run process_data.py from scratch.
