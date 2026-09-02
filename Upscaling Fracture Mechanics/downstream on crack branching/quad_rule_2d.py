"""
2D quadrature rule for integrals with singular weight r^{-alpha} on a circular region.
Target: I = int_{B_delta} f(x,y) * r^{-alpha} dx dy. Use quadweights_2d to get points/weights,
quad_rule_2d(integrand, points_weights) with integrand = f*r^{-alpha} to approximate I.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import scipy.integrate as integrate

# ========== 2D Quadrature Rule Functions ==========

def Pbasis_2d(x, y, order):
    """Generate 2D polynomial basis functions up to given order"""
    basis = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            basis.append((x**i) * (y**j))
    return np.array(basis)


def tau_value_2d(delta, alpha, p_order, tol=1e-14):
    """
    Compute tau values for 2D quadrature weights over circular region with radius delta.
    Basis functions are normalized by delta: basis_i((x-center)/delta, (y-center)/delta).
    """
    nbasis = (p_order + 1) * (p_order + 2) // 2
    tau = np.zeros(nbasis)

    basis_idx = 0
    for i in range(p_order + 1):
        for j in range(p_order + 1 - i):
            def integrand_polar(r, theta):
                if r < tol:
                    return 0.0
                x_norm = r * np.cos(theta) / delta
                y_norm = r * np.sin(theta) / delta
                return (x_norm**i) * (y_norm**j) * np.power(r, -alpha) * r

            result = integrate.dblquad(
                integrand_polar,
                0, 2*np.pi,
                lambda theta: 0,
                lambda theta: delta,
                epsabs=1e-14,
                epsrel=1e-14
            )
            tau[basis_idx] = result[0]
            basis_idx += 1

    return tau


def quadweights_2d(a, b, delta, h, alpha, p_order, tol=1e-14, reg_method='kernel', lambda_reg=1e-6):
    """
    Generate 2D quadrature weights for integration over circular region.

    Solves the kernel-regularized moment-matching problem (same as 1D reg_method='kernel'):
        min_{w_j} (1/2) * sum_j w_j^2 * r_j^{-alpha}   (kernel in norm not squared)
        s.t.  int_{B_delta} q(x,y) * r^{-alpha} dx dy = sum_j q(x_j,y_j) * r_j^{-alpha} * w_j
              for all q in P_n (2D polynomials of order p_order).

    Parameters:
    -----------
    a, b : float
        Domain boundaries [a, b] x [a, b]
    delta : float
        Radius of circular integration region
    h : float
        Grid spacing
    alpha : float
        Singularity parameter (r^(-alpha))
    p_order : int
        Polynomial order
    tol : float, optional
        Tolerance for numerical integration in tau_value_2d (default 1e-14)
    reg_method : str
        'pinv': minimize L2 norm of w; 'kernel': minimize (1/2)*sum_j w_j^2*r_j^{-alpha}
    lambda_reg : float
        Regularization for kernel method (default 1e-6). M_reg = M + lambda_reg*I.

    Returns:
    --------
    points_weights : array
        Shape (n_points, 3) with columns [x, y, weight]. Weight is raw w_j.
        For int f*r^{-alpha}, use quad_rule_2d(integrand, points_weights) with integrand = f*r^{-alpha}.
    cond_num : float
        Condition number of the system matrix (M or M_reg)
    """
    center = (a + b) / 2.0

    # Symmetric grid anchored on multiples of h around 0 (not on a=-delta).
    # Using np.arange(a, b+h/2, h) anchors at -delta, which when delta is not
    # an integer multiple of h (e.g. delta=3.01*h) produces a grid shifted by
    # delta mod h from 0; the r<=delta filter then drops mirror pairs and the
    # stencil loses reflective symmetry.  We instead build a symmetric grid
    # centered on 0 with spacing h that just covers [-delta, +delta].
    m = int(round(delta / h))
    x_grid = np.arange(-m, m + 1) * h
    y_grid = np.arange(-m, m + 1) * h

    points = []
    for x in x_grid:
        for y in y_grid:
            dx = x - center
            dy = y - center
            r = np.sqrt(dx**2 + dy**2)
            if r <= delta and r > 1e-10:
                points.append([x, y, r])

    if len(points) == 0:
        raise ValueError("No points found in circular region. Try increasing delta or decreasing h.")

    points = np.array(points)
    n_points = points.shape[0]
    nbasis = (p_order + 1) * (p_order + 2) // 2

    B = np.zeros((nbasis, n_points))
    for j in range(n_points):
        x = points[j, 0]
        y = points[j, 1]
        r = points[j, 2]
        x_norm = (x - center) / delta
        y_norm = (y - center) / delta
        basis = Pbasis_2d(x_norm, y_norm, p_order)
        B[:, j] = basis * np.power(r, -alpha)

    rhs = tau_value_2d(delta, alpha, p_order, tol)

    if reg_method == 'pinv':
        M = np.matmul(B, B.transpose())
        cond_num = np.linalg.cond(M)
        if cond_num > 1e10:
            reg = 1e-10 * np.trace(M) / M.shape[0] * np.eye(M.shape[0])
            M = M + reg
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)
        weights = np.matmul(B.transpose(), np.matmul(M_inv, rhs))

    elif reg_method == 'kernel':
        r_vec = points[:, 2]
        W_vec = np.power(r_vec, alpha)
        B_D = B * W_vec
        M = np.matmul(B_D, B.transpose())
        M_reg = M + lambda_reg * np.eye(nbasis)
        cond_num = np.linalg.cond(M_reg)
        try:
            lambdas = np.linalg.solve(M_reg, rhs)
        except np.linalg.LinAlgError:
            lambdas = np.matmul(np.linalg.pinv(M_reg), rhs)
        weights = np.matmul(B_D.transpose(), lambdas)

    else:
        raise ValueError("reg_method must be 'pinv' or 'kernel', got %s" % reg_method)

    points_weights = np.column_stack([points[:, 0], points[:, 1], weights])
    return points_weights, cond_num


def quad_rule_2d(integrand, points_weights):
    """
    Apply 2D quadrature rule: sum_j integrand(x_j, y_j) * w_j.

    Stored weight is raw w_j. To approximate int f(x,y)*r^{-alpha} dx dy,
    pass integrand that returns f(x,y)*r^{-alpha} at (x,y).

    Parameters:
    -----------
    integrand : callable
        (x, y) -> value. For int f*r^{-alpha}, pass f(x,y)*r^{-alpha}.
    points_weights : array
        Shape (n_points, 3) with columns [x, y, weight] from quadweights_2d.

    Returns:
    --------
    integral : float
        Approximate integral (sum of integrand * weight over points).
    """
    x_vals = points_weights[:, 0]
    y_vals = points_weights[:, 1]
    weights = points_weights[:, 2]

    if np.isscalar(x_vals[0]):
        f_vals = np.array([integrand(x_vals[i], y_vals[i]) for i in range(len(x_vals))])
    else:
        f_vals = integrand(x_vals, y_vals)

    return np.sum(f_vals * weights)


def test_2d_quadrature_accuracy(tol=1e-14, alpha=0.5, p_orders=(2, 3, 4, 5, 6)):
    """
    Test 2D quadrature rule convergence (scipy dblquad as reference).
    Plots error vs h and condition number vs h (same style as test_quadrature_rule/quad_rule_2d).
    """
    print("=" * 60)
    print("Testing 2D Quadrature Rule Accuracy")
    print("=" * 60)

    delta = 0.25
    a, b = -delta, delta

    def test_func(x, y):
        if isinstance(x, np.ndarray):
            return np.exp(-(x**2 + y**2))
        return np.exp(-(x**2 + y**2))

    center = (a + b) / 2.0

    def integrand_ref(x, y):
        r = np.sqrt((x - center)**2 + (y - center)**2)
        if r < 1e-10:
            return 0.0
        return test_func(x, y) * np.power(r, -alpha)

    def circle_bound(x):
        y_min = center - np.sqrt(delta**2 - (x - center)**2)
        y_max = center + np.sqrt(delta**2 - (x - center)**2)
        return y_min, y_max

    x_min = center - delta
    x_max = center + delta

    try:
        result = integrate.dblquad(
            integrand_ref,
            x_min, x_max,
            lambda x: circle_bound(x)[0],
            lambda x: circle_bound(x)[1],
            epsabs=1e-14,
            epsrel=1e-14
        )
        ref_integral = result[0]
        print(f"\nReference integral (scipy): {ref_integral:.10e}\n")
    except Exception as e:
        print(f"Warning: Could not compute reference integral: {e}")
        def integrand_polar(r, theta):
            x = center + r * np.cos(theta)
            y = center + r * np.sin(theta)
            if r < 1e-10:
                return 0.0
            return test_func(x, y) * np.power(r, -alpha) * r
        result = integrate.dblquad(
            integrand_polar,
            0, 2*np.pi,
            lambda theta: 0,
            lambda theta: delta,
            epsabs=1e-14,
            epsrel=1e-14
        )
        ref_integral = result[0]
        print(f"Reference integral (polar): {ref_integral:.10e}\n")

    h_values = [delta / i for i in [4, 6, 8, 10, 12, 14, 16]]
    all_errors = {}
    all_h_values = {}
    all_cond_nums = {}

    for p_order in p_orders:
        print(f"\n{'='*60}\nTesting p_order = {p_order}\n{'='*60}")
        errors = []
        cond_nums = []
        valid_h_values = []
        nbasis = (p_order + 1) * (p_order + 2) // 2
        print(f"{'h':<10} {'N_points':<12} {'Integral':<20} {'Error':<20} {'Relative Error':<20}")
        print("-" * 85)

        for h in h_values:
            try:
                points_weights, cond_num = quadweights_2d(a, b, delta, h, alpha, p_order, tol)
                n_points = points_weights.shape[0]
                if n_points < nbasis:
                    print(f"{h:<10.5f} Warning: Only {n_points} points, need at least {nbasis} basis functions")
                    continue
                integral = quad_rule_2d(integrand_ref, points_weights)
                error = abs(integral - ref_integral)
                rel_error = error / abs(ref_integral) if abs(ref_integral) > 1e-14 else error
                errors.append(error)
                cond_nums.append(cond_num)
                valid_h_values.append(h)
                print(f"{h:<10.5f} {n_points:<12} {integral:<20.10e} {error:<20.10e} {rel_error:<20.10e}")
            except Exception as e:
                print(f"{h:<10.5f} Error: {e}")
                continue

        all_errors[p_order] = errors
        all_h_values[p_order] = valid_h_values
        all_cond_nums[p_order] = cond_nums

    # Plot: error vs h (same style as test_quadrature_rule/quad_rule_2d)
    print("\n" + "=" * 60)
    print("Plotting convergence results...")
    print("=" * 60)

    import matplotlib
    from matplotlib.ticker import LogLocator

    fontsize = 20
    plt.rcParams.update({'font.size': fontsize})

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = matplotlib.colormaps.get_cmap('Set2')
    colors = [cmap(i) for i in range(len(p_orders))]
    markers = ['s', 'v', 'o', 'd', 'p', 'h', 'x', '*', '+', '^']

    for idx, p_order in enumerate(p_orders):
        if p_order in all_errors and len(all_errors[p_order]) > 0:
            h_vals = all_h_values[p_order]
            err_vals = all_errors[p_order]
            marker = markers[idx % len(markers)]
            ax.plot(h_vals, err_vals, marker + '-',
                    color=colors[idx], label=f'p={p_order}', linewidth=2, markersize=7)

    ref_error = None
    ref_h = None
    for p_order in p_orders:
        if p_order in all_errors and len(all_errors[p_order]) > 0:
            h_vals = all_h_values[p_order]
            err_vals = all_errors[p_order]
            if len(err_vals) > 0 and err_vals[0] > 0:
                ref_error = err_vals[0]
                ref_h = h_vals[0]
                break
    if ref_error is not None and ref_h is not None:
        slope = min(1.0, 2.0 - alpha)
        all_h_for_ref = []
        for p_order in p_orders:
            if p_order in all_h_values:
                all_h_for_ref.extend(all_h_values[p_order])
        if all_h_for_ref:
            h_min = min(all_h_for_ref)
            h_max = max(all_h_for_ref)
            h_ref_line = np.logspace(np.log10(h_min), np.log10(h_max), 100)
            ref_line_errors = ref_error * np.power(h_ref_line / ref_h, slope)
            ax.plot(h_ref_line, ref_line_errors, 'k--', linewidth=2, alpha=0.7,
                    label=f'slope = {slope:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=list(range(2, 10)), numticks=100))
    ax.set_xlabel('Mesh size h', fontsize=fontsize)
    ax.set_ylabel('Error', fontsize=fontsize)
    ax.set_title(f'2D Quadrature Rule Convergence ($\\alpha={alpha}$)', fontsize=fontsize + 2)
    ax.grid(True, which='both', linestyle='--', color='gray', alpha=0.6)
    ax.legend(fontsize=fontsize - 2)
    plt.tight_layout()

    plot_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'quad_plots')
    os.makedirs(plot_dir, exist_ok=True)
    name = f'quadrature_convergence_2d_alpha{alpha}_small_h.png'
    plt.savefig(os.path.join(plot_dir, name), dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {os.path.join(plot_dir, name)}")
    plt.close()

    # Plot: condition number vs h
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for idx, p_order in enumerate(p_orders):
        if p_order in all_cond_nums and len(all_cond_nums[p_order]) > 0:
            h_vals = all_h_values[p_order]
            cond_vals = all_cond_nums[p_order]
            marker = markers[idx % len(markers)]
            ax2.plot(h_vals, cond_vals, marker + '-',
                     color=colors[idx], label=f'p={p_order}', linewidth=2, markersize=7)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax2.xaxis.set_minor_locator(LogLocator(base=10, subs=list(range(2, 10)), numticks=100))
    ax2.set_xlabel('Mesh size h', fontsize=fontsize)
    ax2.set_ylabel('Condition number', fontsize=fontsize)
    ax2.set_title(f'Condition Number vs Mesh Size ($\\alpha={alpha}$)', fontsize=fontsize + 2)
    ax2.grid(True, which='both', linestyle='--', color='gray', alpha=0.6)
    ax2.legend(fontsize=fontsize - 2)
    plt.tight_layout()
    name2 = f'condition_number_2d_alpha{alpha}_small_h.png'
    plt.savefig(os.path.join(plot_dir, name2), dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {os.path.join(plot_dir, name2)}")
    plt.close()

    # Convergence analysis
    print("\n" + "=" * 60)
    print("Convergence Analysis:")
    print("=" * 60)
    for p_order in p_orders:
        if p_order not in all_errors or len(all_errors[p_order]) < 2:
            continue
        errors = all_errors[p_order]
        h_vals = all_h_values[p_order]
        valid_errors = [e for e in errors if not np.isnan(e) and e > 0]
        if len(valid_errors) >= 2:
            convergence_rates = []
            for i in range(1, len(valid_errors)):
                if valid_errors[i - 1] > 0:
                    rate = np.log(valid_errors[i] / valid_errors[i - 1]) / np.log(h_vals[i] / h_vals[i - 1])
                    convergence_rates.append(rate)
            if convergence_rates:
                avg_rate = np.mean(convergence_rates)
                print(f"p_order = {p_order}: average convergence rate = {avg_rate:.4f}")
    print("=" * 60)

    return all_h_values, all_errors, ref_integral


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 2D quadrature rule convergence')
    parser.add_argument('--alpha', type=float, default=0.5, help='Singularity parameter')
    parser.add_argument('--p_orders', type=int, nargs='+', default=[2, 3, 4, 5, 6], help='Polynomial orders')
    args = parser.parse_args()

    print("=== 2D Quadrature Rule Implementation ===")
    print()
    test_2d_quadrature_accuracy(tol=1e-14, alpha=args.alpha, p_orders=args.p_orders)


if __name__ == "__main__":
    main()
