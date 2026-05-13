ex32：Ex-II, second-order.
ex37：Ex-I, firsr-order.
ex35：A special example, compare MGN, MLP and ICNN.
    square loss: learnt g is more smooth, close to the monotonicity; L1 loss: it's easy to learn a non-monotone g. So MLP is using the L1 loss.
    PNO_NN_ex35_k1: 
        plot_NN_MGN_ICNN_L1loss.py: plot comparison figure and compute the model error (L2(rho) error) using the selected model.
        plot_u_NN_MGN_ICNN_L1loss.py: compute the error u using the selected model.  (Note: the wrong solution index is different for different machines.)
ex38：Discontinuous data.

Cond: Compute the condition number.