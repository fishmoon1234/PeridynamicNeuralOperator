#!/usr/bin/env python3
"""Quasistatic loading of a PRISTINE (un-notched) plate with a
peridynamic neural operator (PNO) as the constitutive model: the
fracture-free reference for fracture_estimate_force.py.

Constitutive model (per bond):
    lambda = 1 + (|xi + eta| - |xi|)/|xi|
    f_ij = g(lambda) k_sym(xi/delta) |xi/delta|^(-alpha)
           (xi+eta)/|xi+eta| dx^2
with the symmetrized kernel k_sym(z) = (k(z)+k(-z))/2.  There is no
pre-notch and bond breaking is disabled: every step is a (nonlinear)
elastic equilibrium.  The softening function is selected by --g-rule:
'linear' replaces the learned g by its tangent at lambda = 1 (slope
fitted over 0 <= lambda-1 <= 0.01); 'learned' uses the raw network g.
The operator amplitude carries the unit conversion s^2/sigma_f; the
horizon is a model property, delta = gamma * delta_MD, independent of
the grid.

Geometry: the plate [-L, L] x [-2L - dx/2, 2L + dx/2] on a staggered
grid with node rows at y = +-dx/2, +-3dx/2, ...; the measurement
planes y = 0, +-L fall between rows, so the cross-section force sum
has no on-row special cases.

Loading and solve: the top/bottom grip rows (|y| >= 2L - strip_width)
prescribe u_y = +-pull_rate * t (x free); the center line x = 0 pins
u_x = 0.  Each step solves the static equilibrium f_int(u) = 0 on
free dofs by a globalized (line search + Levenberg-Marquardt)
matrix-free Newton-Krylov method.  Free surfaces: --boundary nothing
(untreated) or --boundary traction (mirror fictitious nodes with the
Lu-Li zero-traction condition on the side edges).

Outputs per step (force.txt, physical MD force units): grip reactions,
the bond force transmitted across y = 0 and y = +-L, and the
continuum-elasticity estimate E * eps_yy * (2L*H) with
E = 382 fu/lu^2 and H = 0.935 lu.

Units: MD units (lu = 1e-10 m); (s, sigma_f) of the checkpoint's
training scaling are auto-inferred from summary.json or given by
--unit-s/--unit-sigma-f.

Usage:
    python nofracture_estimate_force.py [options]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from egnn_gcl import E_GCL_GKN, unsorted_segment_sum
from utilities_INO_PD import parse_layer_info, compute_quadrature_weights_simple

# Fixed internal settings (were CLI options in the
# full driver):
BONDSOFT_DECAY = 10000.0  # bondsoft sharpness (near-hard bond break)


def find_learned_g_peak(model, lam_min=1.001, lam_max=1.5, n=4001,
                        device="cpu"):
    """Locate the tension-side peak of the learned g(lambda)."""
    lam = torch.linspace(lam_min, lam_max, n, device=device)
    with torch.no_grad():
        g = model.signed_g_difference(lam).reshape(-1)
    i = int(torch.argmax(g).item())
    return float(lam[i].item()), float(g[i].item())


def load_summary(p):
    return json.load(open(Path(p) / "summary.json"))


def get_act(name):
    return {"ReLU": torch.nn.ReLU, "GELU": torch.nn.GELU,
            "Tanh": torch.nn.Tanh, "Softplus": torch.nn.Softplus}[name]


def build_model(summary, model_dir, device):
    k_layer = parse_layer_info(summary["k_layer"], input_dim=2)
    g_layer = parse_layer_info(summary["g_layer"], input_dim=1)
    m = E_GCL_GKN(k_layer, g_layer, get_act(summary["act"]),
                   init_alpha=summary["alpha_0"], use_singular=True).to(device)
    m.load_state_dict(torch.load(model_dir / "model.ckpt", map_location=device))
    m.eval()
    return m


class SymmetrizedK(torch.nn.Module):
    """Even part of the learned kernel network:

        k_sym(zeta) = ( k(zeta) + k(-zeta) ) / 2.

    A bond-based force sum only ever samples the symmetric combination of
    a bond and its reverse, so k_sym is the physically identifiable part
    of the kernel (same construction as compute_NW_manuscript_symmetry.py).
    Installed over ``model.k`` after checkpoint loading, so every operator
    evaluation (f_int, diagnostics) uses it transparently.
    """

    def __init__(self, k_module):
        super().__init__()
        self.k_raw = k_module

    def forward(self, zeta):
        return 0.5 * (self.k_raw(zeta) + self.k_raw(-zeta))


# ═══════════════════════════════════════════════════════════════════════
#  Training-unit conversion (fracture_energy_derivation eq. 14; identical
#  to fracture_sim_implicit_norm.py / compute_NW_manuscript.discover_cases)
# ═══════════════════════════════════════════════════════════════════════

# Runs whose training scaling does not follow the delta-based convention.
UNIT_OVERRIDES = {
    "nrm_xu01f001": (0.1, 0.01),   # x,u scaled by 0.1, f by an extra 0.01
}


def resolve_training_units(summary, unit_s=None, unit_sigma_f=None):
    """Determine the (s, sigma_f) scalings applied to (x,u) and f when the
    checkpoint was trained, and the resulting MD-unit conversion:

        unit_fac = s^2 / sigma_f    (multiplies the operator output when
                                     the domain is expressed in MD units)
        delta_md = delta_train / s  (the model's native horizon in lu;
                                     15.05 for every run in this project)

    Inference rule (same as compute_NW_manuscript.discover_cases):
    explicit arguments win; otherwise the run_tag is looked up in
    UNIT_OVERRIDES; otherwise summary delta > 5 means the run was trained
    directly in MD units ((s, sigma_f) = (1, 1)) and delta <= 5 means the
    legacy x0.1 scaling ((0.1, 0.1)).

    Returns (s, sigma_f, unit_fac, delta_train, delta_md, source_str).
    """
    delta_train = float(summary["delta"])
    run_tag = summary.get("run_tag", "")
    if unit_s is not None and unit_sigma_f is not None:
        s, sf = float(unit_s), float(unit_sigma_f)
        source = "explicit --unit-s/--unit-sigma-f"
    else:
        if run_tag in UNIT_OVERRIDES:
            s, sf = UNIT_OVERRIDES[run_tag]
            source = f"UNIT_OVERRIDES[{run_tag!r}]"
        else:
            s, sf = (1.0, 1.0) if delta_train > 5.0 else (0.1, 0.1)
            source = f"inferred from summary delta={delta_train:g}"
        # allow overriding just one of the two
        if unit_s is not None:
            s = float(unit_s)
            source += f" + explicit --unit-s={s:g}"
        if unit_sigma_f is not None:
            sf = float(unit_sigma_f)
            source += f" + explicit --unit-sigma-f={sf:g}"
    unit_fac = s * s / sf
    delta_md = delta_train / s
    return s, sf, unit_fac, delta_train, delta_md, source


# ═══════════════════════════════════════════════════════════════════════
#  Domain setup
# ═══════════════════════════════════════════════════════════════════════

def create_rectangular_grid(Nx, Ny, dx):
    """Create uniform rectangular grid [0, (Nx-1)*dx] x [0, (Ny-1)*dx].

    Node ordering: index = ix * Ny + iy  (x-major).
    Returns coords [N, 2].
    """
    ix = torch.arange(Nx, dtype=torch.get_default_dtype())
    iy = torch.arange(Ny, dtype=torch.get_default_dtype())
    grid_x, grid_y = torch.meshgrid(ix, iy, indexing='ij')
    coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1) * dx
    return coords


# ═══════════════════════════════════════════════════════════════════════
#  Fast graph construction for rectangular grids
# ═══════════════════════════════════════════════════════════════════════

def build_rectangular_graph(Nx, Ny, dx, delta):
    """Build edge_index and edge-to-quadrature-weight mapping.

    Uses the same enumeration order as quadweights_2d (x outer, y inner)
    so that edge_weight_indices[e] gives the correct quadrature weight index.

    For boundary nodes, neighbors outside the domain are skipped,
    producing incomplete neighborhoods (free-surface-like behaviour).

    Returns:
        edge_index:         [2, n_edges] long tensor
        edge_weight_indices: [n_edges] long tensor  (maps edge -> quad weight)
        n_quad:             int  (number of quadrature points per full stencil)
    """
    # Mirror the enumeration order of quadweights_2d.  Build a symmetric
    # grid anchored on multiples of dx around 0 (NOT np.arange(-delta,
    # delta+dx/2, dx), which when delta != m*dx produces a half-step-shifted
    # grid and an asymmetric stencil).  q_idx tracks the index inside the
    # qpw array; we skip creating edges for the (di=0, dj=0) self-loop but
    # still increment q_idx so subsequent indices stay aligned.
    a, b = -delta, delta
    center = (a + b) / 2.0
    m = int(round(delta / dx))
    x_grid = np.arange(-m, m + 1) * dx
    y_grid = np.arange(-m, m + 1) * dx

    quad_offsets = []
    q_idx = 0
    for xv in x_grid:
        for yv in y_grid:
            rx = xv - center
            ry = yv - center
            r = np.sqrt(rx ** 2 + ry ** 2)
            if r <= delta and r > 1e-10:
                di = round(rx / dx)
                dj = round(ry / dx)
                if di != 0 or dj != 0:          # skip self-loop
                    quad_offsets.append((di, dj, q_idx))
                q_idx += 1                       # always advance
    n_quad = q_idx

    # Build edges — O(N * n_quad)
    rows, cols, qidxs = [], [], []
    for ix in range(Nx):
        for iy in range(Ny):
            node = ix * Ny + iy
            for di, dj, qi in quad_offsets:
                ni, nj = ix + di, iy + dj
                if 0 <= ni < Nx and 0 <= nj < Ny:
                    rows.append(node)
                    cols.append(ni * Ny + nj)
                    qidxs.append(qi)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight_indices = torch.tensor(qidxs, dtype=torch.long)
    return edge_index, edge_weight_indices, n_quad


# ═══════════════════════════════════════════════════════════════════════
#  PNO forward with damage mask
# ═══════════════════════════════════════════════════════════════════════

def apply_operator_with_damage(model, u, x, edge_index, delta, dx,
                               bond_strength, device, g_fn=None,
                               gamma_scale=1.0, rescale_factor=1.0):
    """Replicate E_GCL_GKN.forward() (norm variant) with a per-bond multiplier.

    Uses the normalised singular factor  (|xi|/delta)^{-alpha}  and
    Riemann (dx^2) integration, matching the norm_r08 training setup.

    If ``g_fn`` is provided, it overrides ``model.signed_g_difference``
    for the softening evaluation (used to apply the lam-cut wrapper).

    Returns:
        force        [N, 2]  force at each node
        bond_stretch [n_edges]  stretch  s_ij = lambda_ij - 1
    """
    x_d = x.to(device)
    u_d = u.to(device)
    ei = edge_index.to(device)
    bs = bond_strength.to(device)

    row, col = ei
    ksi = x_d[col] - x_d[row]                          # [E, 2]
    eta = u_d[col] - u_d[row]                           # [E, 2]
    ksi_plus_eta = ksi + eta
    ksi_norm = torch.norm(ksi, dim=1, keepdim=True)     # [E, 1]
    kpe_norm = torch.norm(ksi_plus_eta, dim=1, keepdim=True)
    lambdaa = 1.0 + (kpe_norm - ksi_norm) / (ksi_norm + 1e-9)
    bond_dir = ksi_plus_eta / (kpe_norm + 1e-9)

    with torch.no_grad():
        if g_fn is None:
            g_NN = model.signed_g_difference(lambdaa).reshape(-1, 1)
        else:
            g_NN = g_fn(lambdaa).reshape(-1, 1)
        ksi_2d = ksi / (delta + 1e-12)
        k_NN = model.k(ksi_2d)
        alpha_eff = model.get_alpha()
        # Normalised singular factor: (|xi|/delta)^{-alpha}
        phi_NN = g_NN * k_NN * (ksi_norm / (delta + 1e-12)) ** (-alpha_eff)

    weighted = phi_NN * bond_dir * (dx ** 2) * bs.unsqueeze(1)
    force = unsorted_segment_sum(weighted, row, num_segments=x_d.size(0))
    force = force / (gamma_scale ** 3) * rescale_factor
    stretch = (lambdaa.squeeze(1) - 1.0).detach()
    return force, stretch


def internal_force_grad(model, u, x, edge_index, delta, dx,
                        bond_strength, device, g_fn=None,
                        gamma_scale=1.0, rescale_factor=1.0):
    """Differentiable internal force f_int(u; bond_strength) — norm variant.

    Same physics as ``apply_operator_with_damage`` but does NOT call
    ``torch.no_grad()``, so autograd can build a graph through u.  Used
    by Newton's matrix-free Jacobian-vector product.  Uses the normalised
    singular factor and Riemann (dx^2) weights.
    """
    x_d  = x.to(device)
    ei   = edge_index.to(device)
    bs   = bond_strength.to(device).detach()

    row, col = ei
    ksi = x_d[col] - x_d[row]
    eta = u[col] - u[row]
    ksi_plus_eta = ksi + eta
    ksi_norm = torch.norm(ksi, dim=1, keepdim=True)
    kpe_norm = torch.norm(ksi_plus_eta, dim=1, keepdim=True)
    lambdaa  = 1.0 + (kpe_norm - ksi_norm) / (ksi_norm + 1e-9)
    bond_dir = ksi_plus_eta / (kpe_norm + 1e-9)

    if g_fn is None:
        g_NN = model.signed_g_difference(lambdaa).reshape(-1, 1)
    else:
        g_NN = g_fn(lambdaa).reshape(-1, 1)
    ksi_2d   = ksi / (delta + 1e-12)
    k_NN     = model.k(ksi_2d)
    alpha_eff = model.get_alpha()
    # Normalised singular factor: (|xi|/delta)^{-alpha}
    phi_NN   = g_NN * k_NN * (ksi_norm / (delta + 1e-12)) ** (-alpha_eff)

    weighted = phi_NN * bond_dir * (dx ** 2) * bs.unsqueeze(1)
    force = unsorted_segment_sum(weighted, row, num_segments=x_d.size(0))
    force = force / (gamma_scale ** 3) * rescale_factor
    return force


def compute_force_across_y0(model, u, x, edge_index, delta, dx,
                            bond_strength, device, g_fn=None,
                            gamma_scale=1.0, rescale_factor=1.0, y0=0.0):
    """Total peridynamic force transmitted across the horizontal line y = y0
    (the notch plane).

    Sums, over every directed bond (row, col) whose reference bond vector
    xi_ij = x[col] - x[row] crosses the line y = y0 (row strictly below,
    x[row,1] < y0 <= x[col,1]; the mirror-direction copy of the same
    geometric bond, with row and col swapped, is skipped so each bond is
    counted exactly once), the quantity

        gamma^{-(d+1)} * b_ij * phi(xi_ij, lambda_ij)
            * (xi_ij + eta_ij) / |xi_ij + eta_ij| * dx^2,

    where phi = g(lambda) * k(xi/delta) * (|xi|/delta)^{-alpha} is the same
    per-bond kernel used by ``apply_operator_with_damage`` /
    ``internal_force_grad``, b_ij = bond_strength, eta_ij = u[col]-u[row],
    and d = 2 is the spatial dimension so gamma^{-(d+1)} = gamma_scale**-3
    (matching the gamma_scale**3 normalisation used elsewhere). This is the
    standard bond-based-peridynamics "force across a surface" (Silling &
    Zimmermann, 2000): the net force the upper half-space (y > y0) exerts
    on the lower half-space (y < y0) through the bonds spanning the cut --
    a cross-check independent of the nodal-residual boundary reaction
    force.

    Returns:
        F  torch.Tensor [2]  total PHYSICAL force vector across the cut,
           in MD force units integrated over the slab thickness
           (scaling dx^4 * H_SLAB, H_SLAB = 0.935 lu; see the comment at
           the scaling step below).
    """
    x_d = x.to(device)
    u_d = u.to(device)
    ei = edge_index.to(device)
    bs = bond_strength.to(device)
    epsilon = 1e-8

    row, col = ei
    with torch.no_grad():
        ksi = x_d[col] - x_d[row]
        eta = u_d[col] - u_d[row]
        ksi_plus_eta = ksi + eta
        ksi_norm = torch.norm(ksi, dim=1, keepdim=True)
        kpe_norm = torch.norm(ksi_plus_eta, dim=1, keepdim=True)
        lambdaa = 1.0 + (kpe_norm - ksi_norm) / (ksi_norm + 1e-9)
        bond_dir = ksi_plus_eta / (kpe_norm + 1e-9)

        if g_fn is None:
            g_NN = model.signed_g_difference(lambdaa).reshape(-1, 1)
        else:
            g_NN = g_fn(lambdaa).reshape(-1, 1)
        ksi_2d = ksi / (delta + 1e-12)
        k_NN = model.k(ksi_2d)
        alpha_eff = model.get_alpha()
        phi_NN = g_NN * k_NN * (ksi_norm / (delta + 1e-12)) ** (-alpha_eff)

        weighted = phi_NN * bond_dir * bs.unsqueeze(1)
        # Physical force scaling: each directed bond contributes a force
        # DENSITY pair phi*dx^2; converting the sum over crossing bonds
        # to a physical MD force multiplies by the source-node volume
        # dx^2 * H_SLAB (slab thickness H_SLAB = 0.935 lu), i.e. a
        # total factor dx^4 * H_SLAB.
        weighted = (weighted / (gamma_scale ** 3) * rescale_factor
                    * (dx ** 4) * 0.935)

        # Both sides of the split use one single threshold, a
        # quarter-cell below the plane: a node exactly on the plane row
        # belongs to the upper side, every crossing bond is counted
        # exactly once, and the dx/4 margin is robust in float32.
        thr = y0 - 0.25 * dx
        crosses = (x_d[row, 1] < thr) & (x_d[col, 1] >= thr)
        return weighted[crosses].sum(dim=0)


# ═══════════════════════════════════════════════════════════════════════
#  Inner solve helpers (g-continuation primer + LM Newton-Krylov)
# ═══════════════════════════════════════════════════════════════════════

def ghost_force_across_y0(x, ghosts, g_rows, g_idx, wg, dx, y0=0.0,
                          epsilon=1e-8):
    """Correction to the cross-section force for --boundary traction:
    the force transmitted through GHOST bonds whose segment crosses the
    plane y = y0 (side-edge ghosts within one horizon of x = +-L have
    bonds to interior nodes on the other side of the plane; these are
    invisible to compute_force_across_y0, which sums interior bonds
    only).

    ``wg`` are the per-ghost-bond weighted forces returned by
    internal_force_mirror(return_ghost_edge_forces=True): the force ON
    the interior receiver i FROM its ghost bond, already carrying the
    phi*dx^2 quadrature and the gamma/rescale factors.  Convention
    identical to compute_force_across_y0 (force the upper side exerts
    on the lower side): +wg when the interior node is below the plane
    and the ghost above, -wg for the reverse; the remaining physical
    scaling dx^2 * H_SLAB completes the dx^4*H convention.
    """
    y_i = x[g_rows, 1].to(wg.device)
    y_g = ghosts["pos"][g_idx, 1].to(wg.device)
    thr = y0 - 0.25 * dx        # same single-threshold convention as
    up = (y_i < thr) & (y_g >= thr)   # compute_force_across_y0
    dn = (y_g < thr) & (y_i >= thr)
    sgn = up.to(wg.dtype) - dn.to(wg.dtype)
    return (wg * sgn.unsqueeze(1)).sum(dim=0) * (dx ** 2) * 0.935


E_YOUNG = 382.0     # Young's modulus of the MD mesh: 3.82 GPa = 382 fu/lu^2
H_SLAB = 0.935      # MD slab thickness [lu]


def compute_force_elastic(u, Nx, Ny, dx, Lx, device):
    """Continuum-elasticity estimate of the force transmitted across the
    mid-plane y = 0: evaluate the Green-Lagrange strain

        eps = 1/2 (grad u + grad u^T + grad u^T grad u)

    by central finite differences on the grid, take its yy-component on
    the node row(s) nearest y = 0 (the two rows at y = -+dx/2 on the
    staggered grid), average across the width, and convert with the
    MD Young's modulus (MD_results.pdf: E = 3.82 GPa = 382 fu/lu^2):

        F_elastic = E * mean(eps_yy) * (2L * H),   H = 0.935 lu,

    i.e. stress times the cross-sectional area.  Physical MD force
    units (fu), directly comparable to the recorded force_y0_y.
    """
    U = u.detach().cpu().numpy().reshape(Nx, Ny, 2)   # axis0 = x, axis1 = y
    dU_dx = np.gradient(U, dx, axis=0)                # [Nx, Ny, 2]
    dU_dy = np.gradient(U, dx, axis=1)
    if Ny % 2 == 1:
        rows = [(Ny - 1) // 2]                # a node row at y = 0
    else:
        rows = [Ny // 2 - 1, Ny // 2]         # staggered: y = -+dx/2
    eps_list = []
    for j0 in rows:
        duy_dy = dU_dy[:, j0, 1]
        dux_dy = dU_dy[:, j0, 0]
        eps_list.append(duy_dy + 0.5 * (dux_dy ** 2 + duy_dy ** 2))
    eps_yy = np.mean(eps_list, axis=0)
    return float(E_YOUNG * eps_yy.mean() * (Lx * H_SLAB))


def _cg_spd(matvec, b, tol=1e-6, max_iter=200):
    """Standard (unpreconditioned) CG for an SPD matrix-free operator."""
    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r.reshape(-1), r.reshape(-1))
    bnorm = b.norm().clamp_min(1e-30)
    if r.norm() / bnorm < tol:
        return x
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = torch.dot(p.reshape(-1), Ap.reshape(-1)).clamp_min(1e-30)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        if r.norm() / bnorm < tol:
            break
        rs_new = torch.dot(r.reshape(-1), r.reshape(-1))
        beta = rs_new / rs_old.clamp_min(1e-30)
        p = r + beta * p
        rs_old = rs_new
    return x


def _robust_newton_solve_residual(u_init, residual_fn,
                                  tol=1e-4, max_iter=50,
                                  cg_tol=1e-2, cg_max_iter=60,
                                  verbose=False, tag="", free_mask=None):
    """Globalized Levenberg-Marquardt (Gauss-Newton normal equations).

    ``_newton_solve_residual`` solves the plain Newton system J du = -R
    and only accepts a step if it happens to reduce |R|; this is *not*
    guaranteed for any lam-damped variant of it either
    ((J + lam I) du = -R) unless J is symmetric positive (semi-)definite,
    which is not guaranteed here (f_int's Jacobian need not be symmetric
    -- e.g. it is only exactly symmetric if the learned per-bond kernel
    exactly obeys Newton's third law, which it does not in general).

    This solves the Levenberg-Marquardt normal equations instead:

        (J^T J + lam I) du = -J^T R

    via CG (the system is SPD by construction, for any lam >= 0,
    regardless of whether J itself is symmetric), computed matrix-free
    with a forward-mode JVP (J p) composed with a reverse-mode VJP
    (J^T (.)). ``du`` is then *guaranteed* to be a descent direction of
    the merit function f(u) = 0.5*|R(u)|^2 (up to CG truncation error),
    so a standard Armijo backtracking line search on f(u) is guaranteed
    to find an accepted step for small enough step size -- except
    exactly at a stationary point of f (|J^T R| = 0), which for a
    genuinely non-conservative/non-monotone operator can be a local
    minimum of f with R != 0 (a true limitation of gradient-based
    methods, not a bug). ``lam`` grows 10x on a failed step and shrinks
    by half on a successful one (standard LM trust-region-like control).

    Relative convergence (``rel = |R_k| / |R_0|``) is self-normalized by
    THIS call's own initial residual (it=0), not an externally-supplied
    scale -- important because callers reuse this solver across a
    sequence of g-continuation stages whose residual magnitude changes
    by orders of magnitude between stages; a single fixed external
    reference computed once (e.g. from a different, unblended g) would
    make ``rel`` meaningless for the other stages.

    ``free_mask`` (optional, same shape as ``u_init``, True = solve for
    this dof): when given, every unknown (``du``) is projected onto this
    subspace at *every* CG iteration (both the search direction fed into
    the matrix-free ``J``/``J^T`` and the matvec's output are masked to
    zero outside it, and the right-hand side ``b = -J^T R`` is masked
    too), so ``du`` is exactly, bit-for-bit zero outside ``free_mask`` in
    every CG iterate -- not just at convergence. This is the standard
    ``P^T A P`` reduced-system trick for eliminating fixed dofs from an
    iterative solve: masked dofs are never part of the linear algebra at
    all (as opposed to merely adding a penalty/soft-pin residual term for
    them, which only pins them up to the solver's convergence tolerance).
    """
    u = u_init.detach().clone()
    history = []
    ref = None
    lam = 1e-6
    free_mask_f = None if free_mask is None else free_mask.to(u.dtype)
    for it in range(max_iter):
        u_var = u.detach().requires_grad_(True)
        R = residual_fn(u_var)
        rnorm = R.detach().norm().item()
        if ref is None:
            ref = max(rnorm, 1e-12)
        rel = rnorm / ref
        history.append(rel)
        if verbose:
            print(f"    [qnewton{tag}] it={it:3d}  |R|={rnorm:.3e}  "
                  f"rel={rel:.3e}  lam={lam:.1e}")
        if rel < tol:
            return u.detach(), history, True

        u_d = u.detach()
        R_d = R.detach()
        f0 = 0.5 * rnorm * rnorm

        def jvp_(p):
            _, Jp = torch.autograd.functional.jvp(
                residual_fn, (u_d,), (p,), create_graph=False)
            return Jp.detach()

        def vjp_(v):
            _, JTv = torch.autograd.functional.vjp(residual_fn, u_d, v)
            return JTv.detach()

        g = vjp_(R_d)  # gradient of the merit function, J^T R
        if free_mask_f is not None:
            g = g * free_mask_f
        b = -g

        accepted = False
        lam_try = lam
        for _lm_attempt in range(10):
            def matvec(p, _lam=lam_try):
                p_in = p if free_mask_f is None else p * free_mask_f
                out = vjp_(jvp_(p_in)) + _lam * p_in
                return out if free_mask_f is None else out * free_mask_f
            du = _cg_spd(matvec, b, tol=cg_tol, max_iter=cg_max_iter)
            if free_mask_f is not None:
                du = du * free_mask_f
            step = 1.0
            for _ls in range(30):
                tu = u + step * du
                with torch.no_grad():
                    R_try = residual_fn(tu)
                f_try = 0.5 * R_try.norm().item() ** 2
                if f_try <= (1.0 - 1e-4 * step) * f0:
                    u = tu.detach()
                    accepted = True
                    lam = max(1e-8, lam_try * 0.3)
                    break
                step *= 0.5
            if accepted:
                break
            lam_try = lam_try * 10.0
        if not accepted:
            if verbose:
                print(f"    [qnewton{tag}] stalled at it={it}  rel={rel:.3e}  "
                      f"(LM damping up to lam={lam_try:.1e} did not help -- "
                      f"|grad f|={g.norm().item():.3e})")
            return u.detach(), history, rel < tol
    return u.detach(), history, (history[-1] < tol if history else False)


def compute_isolated_nodes(edge_index, bond_alive, n_nodes):
    """Boolean mask [N]: True for a node with NO intact incident bond
    (``bond_alive`` summed over both endpoints of every incident edge is
    0 -- e.g. a node fully inside the pre-notch, or one whose last
    remaining bonds broke during the simulation). Such a node's internal
    force f_int(u) is identically 0 for EVERY u (no bond term in the sum
    ever references it), so its free-dof residual row -f_int(u) is
    identically 0 regardless of its displacement: the row/column of the
    Newton Jacobian at that dof is exactly zero, giving the linear system
    an unresolved (rank-deficient) direction with no physical restoring
    force to pin it down.
    """
    dev = bond_alive.device
    row = edge_index[0].to(dev)
    col = edge_index[1].to(dev)
    alive = bond_alive.unsqueeze(1)
    deg_alive = (unsorted_segment_sum(alive, row, n_nodes)
                + unsorted_segment_sum(alive, col, n_nodes)).squeeze(1)
    return deg_alive < 0.5


# ═══════════════════════════════════════════════════════════════════════
#  Optional Lu-Li zero-traction free surfaces (--boundary traction):
#  mirror fictitious nodes + per-block shifts p_alpha, ported verbatim
#  (Lu-Li zero-traction surfaces)
# ═══════════════════════════════════════════════════════════════════════

BLOCKS = ("left", "right", "crack_up", "crack_dn")


def build_ghosts(x, dx, delta, Lx_half, notch_half, notch_y=0.0):
    """Mirror-type fictitious nodes for the free surfaces.

    Returns a dict with, per ghost node:
        pos     [G, 2]  ghost coordinates
        b_idx   [G]     surface-node index (linear extension), or -1
        m_idx   [G]     mirror-node index inside the body
        c_b,c_m [G]     coefficients: u_g = c_b*u[b] + c_m*u[m] (+ p)
        block   [G]     block id into BLOCKS
    Edge ghosts (x = +-L) use the linear extension (c_b, c_m) = (2, -1);
    crack-face ghosts use the even reflection (0, 1).
    """
    m = int(round(delta / dx))
    N = x.shape[0]
    # node lookup by (rounded) coordinates
    key = {(round(float(px) / dx), round(float(py) / dx)): i
           for i, (px, py) in enumerate(x.tolist())}

    pos, b_idx, m_idx, c_b, c_m, block = [], [], [], [], [], []

    xs = x[:, 0]
    ys = x[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_vals = sorted({round(float(v) / dx) for v in ys.tolist()})

    def add(p, b, mi, cb, cm, blk):
        pos.append(p); b_idx.append(b); m_idx.append(mi)
        c_b.append(cb); c_m.append(cm); block.append(blk)

    has_notch = notch_half >= 0.0    # notch_half < 0 disables the crack
    jc = round(notch_y / dx)

    # -- left / right edges: ghosts at x = x_min - k dx and x_max + k dx
    for blk, x_edge, sgn in ((0, x_min, -1.0), (1, x_max, +1.0)):
        ie = round(x_edge / dx)
        for jy in y_vals:
            if (jy == jc and has_notch
                    and notch_half >= Lx_half - 1e-9):
                # the notch reaches this edge: the on-row edge node is
                # isolated (all bonds severed) and never updated, so no
                # ghost may be extended from it
                continue
            b = key[(ie, jy)]
            for k in range(1, m + 1):
                mi = key.get((ie - sgn * k, jy))
                if mi is None:
                    continue
                add((x_edge + sgn * k * dx, jy * dx), b, mi, 2.0, -1.0, blk)

    # -- crack faces: phantom nodes across y = notch_y, |x| <= notch_half.
    #    Upper face (material y > 0) gets ghosts at y = -k dx carrying the
    #    even reflection of the UPPER field; lower face symmetric.
    #    (skipped entirely when the notch is disabled, notch_half < 0 --
    #    otherwise np.arange would still emit ix = 0 and create spurious
    #    phantoms that break the linear patch test)
    crack_cols = ([round(v / dx) for v in
                   np.arange(-notch_half, notch_half + dx / 2, dx)]
                  if has_notch else [])
    for blk, sgn in ((2, +1.0), (3, -1.0)):        # +1: upper-face material
        for ix in crack_cols:
            for k in range(1, m + 1):
                mi = key.get((ix, jc + round(sgn) * k))   # mirror in material
                if mi is None:
                    continue
                add((ix * dx, (jc - round(sgn) * k) * dx), -1, mi, 0.0, 1.0, blk)

    dtype = torch.get_default_dtype()
    return dict(pos=torch.tensor(pos, dtype=dtype),
                b_idx=torch.tensor(b_idx, dtype=torch.long),
                m_idx=torch.tensor(m_idx, dtype=torch.long),
                c_b=torch.tensor(c_b, dtype=dtype).unsqueeze(1),
                c_m=torch.tensor(c_m, dtype=dtype).unsqueeze(1),
                block=torch.tensor(block, dtype=torch.long))


def build_ghost_edges(x, ghosts, delta, notch_half, notch_y=0.0,
                      exclude=None):
    """Directed bonds (interior <- ghost): edge_index_g[0] = interior
    receiver i, edge_index_g[1] = ghost index (into the ghost arrays).

    Edge-block ghosts bond to every interior node within the horizon (the
    real neighbour does not exist outside the domain).  Crack-face
    ghosts bond only to interior nodes on THEIR side of the crack, and
    only along bonds that the pre-notch actually severed (the segment
    crosses the notch span); elsewhere real bonds exist and are kept.

    ``exclude`` (bool [N], optional): receiver nodes that must get NO
    ghost bonds -- the nodes isolated by the pre-notch (all real bonds
    severed) carry no boundary condition at all and are never updated;
    attaching mirror-ghost bonds would drag these free-floating crack
    fragments with one-sided forces.  Only reachable when the notch end
    comes within one horizon of a free edge (--notch-half-length >
    L - delta), but excluded unconditionally for safety.
    """
    gp = ghosts["pos"]
    blk = ghosts["block"]
    N = x.shape[0]
    rows, gidx = [], []
    for g in range(gp.shape[0]):
        d = x - gp[g]
        r = torch.norm(d, dim=1)
        near = (r <= delta) & (r > 1e-10)
        cand = torch.nonzero(near).squeeze(1)
        b = int(blk[g])
        for i in cand.tolist():
            if exclude is not None and bool(exclude[i]):
                continue
            yi = float(x[i, 1]); yg = float(gp[g, 1])
            if b == 2 and yi <= notch_y:          # upper face serves y > 0
                continue
            if b == 3 and yi >= notch_y:          # lower face serves y < 0
                continue
            if b >= 2:
                # only replace bonds severed by the notch: the segment
                # i -> ghost must cross y = notch_y inside the notch span
                dy = yg - yi
                if abs(dy) < 1e-12:
                    continue
                t = (notch_y - yi) / dy
                x_int = float(x[i, 0]) + t * (float(gp[g, 0]) - float(x[i, 0]))
                if not (0.0 <= t <= 1.0 and abs(x_int) <= notch_half + 1e-9):
                    continue
            rows.append(i)
            gidx.append(g)
    return (torch.tensor(rows, dtype=torch.long),
            torch.tensor(gidx, dtype=torch.long))


# ═══════════════════════════════════════════════════════════════════════
#  Ghost-aware internal force
# ═══════════════════════════════════════════════════════════════════════

def ghost_displacement(u, ghosts, p_blocks):
    """u_g = c_b * u[b] + c_m * u[m] + p_block  (differentiable in u)."""
    b = ghosts["b_idx"].clamp(min=0)
    ug = ghosts["c_b"] * u[b] + ghosts["c_m"] * u[ghosts["m_idx"]]
    return ug + p_blocks[ghosts["block"]]


def internal_force_mirror(model, u, x, edge_index, bond_strength,
                          ghosts, g_rows, g_idx, p_blocks,
                          delta, dx, device, g_fn=None,
                          gamma_scale=1.0, rescale_factor=1.0,
                          return_ghost_edge_forces=False):
    """f_int including ghost bonds.  Differentiable in u (the ghost
    displacements are assembled from u); p_blocks is treated as data.

    Interior bonds use the operator formula above; ghost bonds append
    rows to the same segment sum with strength 1.
    """
    N = x.shape[0]
    x_all = torch.cat([x.to(device), ghosts["pos"].to(device)], dim=0)
    ug = ghost_displacement(u, ghosts, p_blocks)
    u_all = torch.cat([u, ug], dim=0)

    row_i = edge_index[0].to(device)
    col_i = edge_index[1].to(device)
    row_g = g_rows.to(device)
    col_g = (g_idx + N).to(device)
    row = torch.cat([row_i, row_g])
    col = torch.cat([col_i, col_g])
    bs = torch.cat([bond_strength.to(device),
                    torch.ones(len(g_rows), device=device,
                               dtype=bond_strength.dtype)])

    ksi = x_all[col] - x_all[row]
    eta = u_all[col] - u_all[row]
    ksi_plus_eta = ksi + eta
    ksi_norm = torch.norm(ksi, dim=1, keepdim=True)
    kpe_norm = torch.norm(ksi_plus_eta, dim=1, keepdim=True)
    lambdaa = 1.0 + (kpe_norm - ksi_norm) / (ksi_norm + 1e-9)
    bond_dir = ksi_plus_eta / (kpe_norm + 1e-9)

    if g_fn is None:
        g_NN = model.signed_g_difference(lambdaa).reshape(-1, 1)
    else:
        g_NN = g_fn(lambdaa).reshape(-1, 1)
    k_NN = model.k(ksi / (delta + 1e-12))
    alpha_eff = model.get_alpha()
    phi_NN = g_NN * k_NN * (ksi_norm / (delta + 1e-12)) ** (-alpha_eff)

    weighted = phi_NN * bond_dir * (dx ** 2) * bs.unsqueeze(1)
    weighted = weighted / (gamma_scale ** 3) * rescale_factor
    force = unsorted_segment_sum(weighted, row, num_segments=x_all.size(0))[:N]
    if return_ghost_edge_forces:
        E_int = len(row_i)
        stretch_int = (lambdaa[:E_int, 0] - 1.0).detach()
        return force, weighted[E_int:], stretch_int
    return force


def solve_traction_shifts(model, u, x, edge_index, bond_strength, ghosts,
                          g_rows, g_idx, p_blocks, delta, dx, device,
                          g_fn, gamma_scale, rescale_factor,
                          iters=4, verbose=False):
    """Lu-Li eq. (50)/(52) with t_alpha = 0: per block alpha, find the
    shift p_alpha such that the summed ghost->interior bond force
    vanishes.  Small per-block 2x2 Newton with autograd Jacobians."""
    blk_of_edge = ghosts["block"][g_idx].to(device)
    p = p_blocks.clone()
    for _ in range(iters):
        p_var = p.detach().requires_grad_(True)

        def tractions(pv):
            _, wg, _ = internal_force_mirror(
                model, u, x, edge_index, bond_strength, ghosts, g_rows,
                g_idx, pv, delta, dx, device, g_fn=g_fn,
                gamma_scale=gamma_scale, rescale_factor=rescale_factor,
                return_ghost_edge_forces=True)
            t = torch.zeros(len(BLOCKS), 2, device=device, dtype=u.dtype)
            t = t.index_add(0, blk_of_edge, wg)
            return t

        t = tractions(p_var)
        tn = t.detach().norm(dim=1)
        if verbose:
            print("      [traction] |t_alpha| =",
                  " ".join(f"{v:.3e}" for v in tn.tolist()))
        if float(tn.max()) < 1e-10:
            break
        # Newton direction per block (blocks are decoupled: t_alpha
        # depends only on p_alpha for frozen u).
        active = []
        dp_all = torch.zeros_like(p)
        for a in range(len(BLOCKS)):
            if not (blk_of_edge == a).any():
                continue
            J = torch.zeros(2, 2, device=device, dtype=u.dtype)
            for c in range(2):
                grad = torch.autograd.grad(t[a, c], p_var, retain_graph=True,
                                           create_graph=False)[0]
                J[c] = grad[a]
            try:
                dp = torch.linalg.solve(J, -t[a].detach())
            except RuntimeError:
                continue
            # Physical trust region: the learned g softens past its
            # peak, so t(p) is NON-monotone and t -> 0 again as
            # |p| -> inf (ghost bonds stretched past the peak carry
            # vanishing force).  An uncapped Newton step that jumps
            # over the force peak lands on that decaying branch and
            # diverges (p -> +-thousands).  The true shifts are tiny
            # (fractions of dx), so cap each update at dx/2.
            dpn = float(dp.norm())
            cap = 0.5 * dx
            if dpn > cap:
                dp = dp * (cap / dpn)
            dp_all[a] = dp
            active.append(a)
        # Per-block backtracking on |t_alpha|: accept a (possibly
        # halved) step only if it decreases the block traction; a
        # block that cannot decrease |t| keeps its current p this
        # iteration instead of being thrown onto the softening branch.
        scale = {a: 1.0 for a in active}
        pending = list(active)
        p_acc = p.clone()
        for _bt in range(6):
            if not pending:
                break
            p_try = p_acc.clone()
            for a in pending:
                p_try[a] = p[a] + scale[a] * dp_all[a]
            with torch.no_grad():
                t_try = tractions(p_try)
            tn_try = t_try.norm(dim=1)
            still = []
            for a in pending:
                if float(tn_try[a]) <= (1.0 - 1e-4) * float(tn[a]):
                    p_acc[a] = p_try[a]
                else:
                    scale[a] *= 0.5
                    still.append(a)
            pending = still
        p = p_acc.detach()
    return p


def solve_quasistatic_step(model, u_n, dirichlet_mask, u_dirichlet_target,
                           x, edge_index, delta, dx,
                           bond_alive, bond_strength, breakable, s0,
                           bondsoft_decay, g_fn, device,
                           newton_tol=1e-4, newton_max_iter=50,
                           cg_tol=1e-2, cg_max_iter=60,
                           verbose=False, gamma_scale=1.0,
                           rescale_factor=1.0):
    """Genuine quasistatic equilibrium: solve f_int(u) = 0 directly.

    Solves the free-dof static-equilibrium residual

        R(u)_{i,c} = -f_int(u)_{i,c}                    free
        R(u)_{i,c} = u_{i,c} - u_dirichlet_target_{i,c}  Dirichlet

    via matrix-free Newton-Krylov (``_robust_newton_solve_residual``),
    globalized with a backtracking line search + Levenberg-Marquardt
    damping; there is NO mass/inertia term at all.

    Nodes with no intact incident bond (see ``compute_isolated_nodes``) are
    additionally excluded from the free-dof solve entirely: their
    displacement is held fixed (frozen at its value when the isolation is
    detected -- i.e. u_n at the start of the step, or the latest u if a
    mask-stabilization pass isolates further nodes mid-step) rather than
    being left as an unconstrained Newton unknown with an identically-zero
    residual row/Jacobian. This is recomputed at the top of every Newton
    call in this function, since ``bond_alive`` only changes between
    stabilization passes (never mid-Newton-solve, where the damage mask is
    frozen).

    There is no velocity/acceleration state to carry between loading
    steps: the run is a pure sequence of static equilibria.

    Returns:
        u_np1, n_iter, converged, n_new_broken, f_int_np1
    """
    N = x.shape[0]

    def effective_mask_target(u_current):
        """Combine the boundary Dirichlet mask with isolated-node freezing.
        Dirichlet dofs keep their prescribed target; isolated (non-
        Dirichlet) dofs are pinned to ``u_current`` (excluded from the
        solve); all other dofs remain free."""
        isolated = compute_isolated_nodes(edge_index, bond_alive, N)
        isolated_2 = isolated.unsqueeze(1).expand(-1, 2)
        eff_mask = dirichlet_mask | isolated_2
        eff_target = torch.where(dirichlet_mask, u_dirichlet_target, u_current)
        return eff_mask, eff_target, isolated

    def make_residual(g_fn_use, eff_mask):
        """R(u)_i = -f_int(u)_i on free dofs, and *exactly* 0 (not a
        soft u - eff_target pin) on Dirichlet/isolated dofs -- those
        dofs are excluded from the solve entirely (see ``free_mask`` in
        ``_robust_newton_solve_residual``), so their displacement is
        never a function of how well the residual is minimized; it is
        held bit-for-bit at the value it was hard-set to before the
        solve started."""
        def R(u):
            f_int = internal_force_grad(model, u, x, edge_index, delta, dx,
                                        bond_strength, device,
                                        g_fn=g_fn_use,
                                        gamma_scale=gamma_scale,
                                        rescale_factor=rescale_factor)
            R_free = -f_int
            return torch.where(eff_mask, torch.zeros_like(u), R_free)
        return R

    u = u_n.detach().clone()
    u[dirichlet_mask] = u_dirichlet_target[dirichlet_mask]

    eff_mask, eff_target, isolated = effective_mask_target(u)
    n_isolated = int(isolated.sum().item())
    if verbose and n_isolated > 0:
        print(f"    [quasistatic] {n_isolated} isolated nodes (no intact "
              f"bonds) excluded from the solve; frozen at current u")
    u[eff_mask] = eff_target[eff_mask]

    u, hist, conv = _robust_newton_solve_residual(
        u, make_residual(g_fn, eff_mask),
        tol=newton_tol, max_iter=newton_max_iter,
        cg_tol=cg_tol, cg_max_iter=cg_max_iter,
        verbose=verbose, tag="", free_mask=~eff_mask)
    n_iter = len(hist)

    with torch.no_grad():
        f_int, stretch = apply_operator_with_damage(
            model, u, x, edge_index, delta, dx, bond_strength,
            device, g_fn=g_fn, gamma_scale=gamma_scale,
            rescale_factor=rescale_factor)
    n_new = update_bonds(stretch, bond_alive, s0, breakable,
                         bond_strength=bond_strength,
                         bondsoft_decay=bondsoft_decay,
                         edge_index=edge_index)
    n_new_total = n_new

    stab_pass = 0
    max_stab_passes = 20
    while n_new > 0 and stab_pass < max_stab_passes:
        stab_pass += 1
        if verbose:
            print(f"    [quasistatic] mask grew (+{n_new}); "
                  f"stabilization pass {stab_pass} with frozen new mask")
        eff_mask, eff_target, isolated = effective_mask_target(u)
        n_isolated = int(isolated.sum().item())
        if verbose and n_isolated > 0:
            print(f"    [quasistatic] {n_isolated} isolated nodes excluded "
                  f"from the solve (stabilization pass {stab_pass})")
        u[eff_mask] = eff_target[eff_mask]
        u, hist_s, conv_s = _robust_newton_solve_residual(
            u, make_residual(g_fn, eff_mask),
            tol=newton_tol, max_iter=newton_max_iter,
            cg_tol=cg_tol, cg_max_iter=cg_max_iter,
            verbose=verbose, tag=f"-stab{stab_pass}", free_mask=~eff_mask)
        n_iter += len(hist_s)
        conv = conv and conv_s
        with torch.no_grad():
            f_int, stretch = apply_operator_with_damage(
                model, u, x, edge_index, delta, dx, bond_strength,
                device, g_fn=g_fn, gamma_scale=gamma_scale,
                rescale_factor=rescale_factor)
        n_new = update_bonds(stretch, bond_alive, s0, breakable,
                             bond_strength=bond_strength,
                             bondsoft_decay=bondsoft_decay,
                             edge_index=edge_index)
        n_new_total += n_new

    if n_new > 0 and verbose:
        print(f"    [quasistatic] WARNING: stabilization hit max_stab_passes="
              f"{max_stab_passes} with {n_new} bonds still breaking; "
              f"mask not fully stabilized.")

    return u, n_iter, conv, n_new_total, f_int


# ═══════════════════════════════════════════════════════════════════════
#  Bond bookkeeping (fracture disabled in this driver)
# ═══════════════════════════════════════════════════════════════════════

def solve_quasistatic_step_traction(
        model, u_n, dirichlet_mask, u_dirichlet_target,
        x, edge_index, delta, dx,
        bond_alive, bond_strength, breakable, s0,
        bondsoft_decay, g_fn, device,
        ghosts, g_rows, g_idx, p_blocks,
        newton_tol=1e-4, newton_max_iter=50,
        cg_tol=1e-2, cg_max_iter=60,
        traction_iters=3, p_newton_iters=4, max_stab_passes=10,
        verbose=False, gamma_scale=1.0, rescale_factor=1.0):
    """Quasistatic step with mirror fictitious nodes and the Lu-Li
    zero-traction condition on the free surfaces (--boundary traction).

    Alternates (a) the LM Newton interior equilibrium with the ghost
    displacements slaved to u and the block shifts p frozen, and (b) the
    per-block 2x2 Newton for the shifts p_alpha driving the summed
    ghost->interior bond force (the discrete surface traction, Lu-Li
    eq. 50) to zero; then breaks over-stretched interior bonds and
    repeats with the frozen new mask (mask stabilization).

    Returns:
        u, n_iter, converged, n_new_broken, f_int, p_blocks, t_res
    (t_res [n_blocks, 2]: residual block tractions, for monitoring).
    """
    N = x.shape[0]
    u = u_n.detach().clone()
    n_new_total = 0
    n_iter = 0
    conv = True
    for stab in range(max_stab_passes):
        for outer in range(traction_iters):
            isolated = compute_isolated_nodes(edge_index, bond_alive, N)
            eff_mask = dirichlet_mask | isolated.unsqueeze(1).expand(-1, 2)

            def residual(uu):
                f = internal_force_mirror(
                    model, uu, x, edge_index, bond_strength, ghosts,
                    g_rows, g_idx, p_blocks, delta, dx, device,
                    g_fn=g_fn, gamma_scale=gamma_scale,
                    rescale_factor=rescale_factor)
                return torch.where(eff_mask, torch.zeros_like(uu), -f)

            u[eff_mask] = torch.where(dirichlet_mask, u_dirichlet_target,
                                      u)[eff_mask]
            u, hist, conv = _robust_newton_solve_residual(
                u, residual, tol=newton_tol,
                max_iter=newton_max_iter, cg_tol=cg_tol,
                cg_max_iter=cg_max_iter, verbose=verbose,
                tag=f"-b{stab}o{outer}", free_mask=~eff_mask)
            n_iter += len(hist)

            p_blocks = solve_traction_shifts(
                model, u.detach(), x, edge_index, bond_strength, ghosts,
                g_rows, g_idx, p_blocks, delta, dx, device, g_fn,
                gamma_scale, rescale_factor, iters=p_newton_iters,
                verbose=verbose)

        # bond breaking on INTERIOR bonds only; if the mask grew,
        # re-solve with the frozen new mask
        with torch.no_grad():
            f_int, w_ghost, stretch = internal_force_mirror(
                model, u, x, edge_index, bond_strength, ghosts, g_rows,
                g_idx, p_blocks, delta, dx, device, g_fn=g_fn,
                gamma_scale=gamma_scale, rescale_factor=rescale_factor,
                return_ghost_edge_forces=True)
        n_new = update_bonds(stretch, bond_alive, s0, breakable,
                             bond_strength=bond_strength,
                             bondsoft_decay=bondsoft_decay,
                             edge_index=edge_index)
        n_new_total += n_new
        if n_new == 0:
            break

    blk_of_edge = ghosts["block"][g_idx].to(device)
    t_res = torch.zeros(len(BLOCKS), 2, device=device)
    t_res = t_res.index_add(0, blk_of_edge, w_ghost)
    return u, n_iter, conv, n_new_total, f_int, p_blocks, t_res


def bondsoft(bond_stretch, s0, decay, eps=1e-3):
    """One-sided bond softening that preserves the g peak.

    Returns 1 for s <= s0 and exp(-decay * smooth_max(s - s0 - eps, 0))
    for s > s0, where smooth_max uses a pseudo-Huber regularisation with
    width sqrt(1e-6).  The small shift eps ensures bsoft(s0) = 1 exactly,
    so the peak of g_soft matches the peak of the original g.

        bondsoft(s) = 1                                   for s <= s0
        bondsoft(s) ~ exp(-decay * (s - s0))              for s >> s0,

    with a smooth crossover of width ~eps near s0.
    Values lie in (0, 1]; a larger ``decay`` makes the transition sharper.
    """
    above = (bond_stretch - s0) > 0
    t = bond_stretch - s0 - eps
    overshoot = above.float() * 0.5 * (t + torch.sqrt(t * t + 1e-6))
    return torch.exp(-decay * overshoot)


def build_G(g_fn, model, s0, bondsoft_decay, lam_min=0.5, lam_max=2.0, n=20001,
            device=None):
    """Build a numerical antiderivative G of the softened g on a fine grid.

    G(s) = integral_0^s  g_soft(t) dt,   G(0) = 0

    where g_soft(t) = bondsoft(t, s0, bondsoft_decay) * g_fn(1+t).

    Returns:
        s_grid  np.ndarray [n]  bond-stretch grid
        G_grid  np.ndarray [n]  G values (G(0) = 0 by construction)
    """
    import numpy as np
    from scipy.integrate import cumulative_trapezoid

    lam = torch.linspace(lam_min, lam_max, n, device=device)
    s   = lam - 1.0
    with torch.no_grad():
        g_cut  = g_fn(lam).reshape(-1)
        # bondsoft only on tension side (s > 0); compression side unchanged
        bsoft  = torch.where(s > 0,
                             bondsoft(s, s0=s0, decay=bondsoft_decay),
                             torch.ones_like(s))
        g_soft = (bsoft * g_cut).cpu().numpy()
    s_np = s.cpu().numpy()

    G_full = cumulative_trapezoid(g_soft, s_np, initial=0.0)
    i0     = int(np.argmin(np.abs(s_np)))   # index closest to s=0
    G_grid = G_full - G_full[i0]            # shift so G(0) = 0
    return s_np, G_grid


def invG(gamma, g_fn, model, s0, bondsoft_decay,
         lam_min=0.5, lam_max=2.0, n=20001, device=None):
    """Solve G(s_tilde) = G(s0) / gamma for s_tilde > 0.

    Uses the antiderivative G of the softened-g (G(0)=0) built on a fine
    grid and locates the root by linear interpolation.

    Parameters
    ----------
    gamma : float
        Scaling factor (> 0).  gamma > 1 means the target is smaller than
        G(s0), so s_tilde < s0;  gamma < 1 gives s_tilde > s0.
    g_fn : callable
        Softened/cutoff g function  g(lambda) -> tensor.
    model : E_GCL_GKN
        Loaded PNO model (used indirectly via g_fn; kept for API clarity).
    s0 : float
        Critical stretch used in bondsoft.
    bondsoft_decay : float
        Sharpness parameter of bondsoft.
    lam_min, lam_max : float
        Search range for lambda (s = lambda - 1).
    n : int
        Number of grid points for numerical integration.

    Returns
    -------
    s_tilde : float
        Solution to G(s_tilde) = G(s0) / gamma, or NaN if no solution found.
    """
    import numpy as np

    s_grid, G_grid = build_G(g_fn, model, s0, bondsoft_decay,
                             lam_min=lam_min, lam_max=lam_max, n=n,
                             device=device)

    # evaluate G at s0 by interpolation
    G_s0   = float(np.interp(s0, s_grid, G_grid))
    target = G_s0 / gamma

    # Search on the tension side (s > 0) for the root G(s) - target = 0.
    pos = s_grid > 0
    s_pos = s_grid[pos]
    G_pos = G_grid[pos]

    residual = G_pos - target
    # find sign change
    sign_changes = np.where(np.diff(np.sign(residual)))[0]
    if len(sign_changes) == 0:
        return float("nan")

    # linear interpolation in the first bracketing interval
    i = sign_changes[0]
    s_lo, s_hi = s_pos[i], s_pos[i + 1]
    r_lo, r_hi = residual[i], residual[i + 1]
    s_tilde = float(s_lo - r_lo * (s_hi - s_lo) / (r_hi - r_lo))
    return s_tilde


def update_bonds(bond_stretch, bond_alive, s0, breakable=None,
                 bond_strength=None, bondsoft_decay=None, edge_index=None):
    """Break alive bonds where stretch > s0 (lambda > 1 + s0).  Irreversible.

    Maintains two parallel per-bond quantities, updated in lockstep on
    exactly the same edge set ``cond`` so they never desynchronise:
      * ``bond_alive``    -- binary {0, 1}, monotone non-increasing.
                             Used purely so that the number of newly
                             broken bonds is a well-defined integer.
      * ``bond_strength`` -- hard-zeroed (set to 0.0) for every edge
                             in ``cond``.  This is the field that
                             actually enters the operator as a force
                             multiplier.

    When ``edge_index`` is provided, the reverse bond (j→i) is also
    explicitly zeroed whenever (i→j) breaks, guaranteeing symmetry:
    both bond_alive and bond_strength are set to 0 for the reverse edge
    regardless of its own stretch value.  Since bond_stretch is
    symmetric by construction (|ksi+eta| is the same for both
    directions), the reverse edge is normally already in ``cond``;
    this symmetrization is a cheap safety guarantee against any future
    asymmetry in accumulated damage state.

    Lock invariant.  Whatever logical mask ``bond_alive`` is locked
    against (e.g. via ``breakable``) is also locked for
    ``bond_strength``: the same ``cond`` is used to write into both,
    so a bond that is forbidden to ``break`` is equally forbidden to
    ``soften``.

    ``bond_strength`` is optional only for backward compatibility.
    ``bondsoft_decay`` is accepted but no longer used (kept for API
    compatibility; the break is now always hard: strength = 0).

    If ``breakable`` is provided (bool tensor [n_edges]), only those
    edges can break.  Used to protect boundary-layer bonds from
    artificial surface-effect fracture.
    """
    cond = (bond_stretch > s0) & (bond_alive > 0.5)
    if breakable is not None:
        cond = cond & breakable
    bond_alive[cond] = 0.0
    if bond_strength is not None and cond.any():
        # Hard break: set strength to 0 immediately for all bonds past
        # the g-peak (lambda > 1 + s0).  No soft-decay tail.
        bond_strength[cond] = 0.0
        # Symmetrize: explicitly kill the reverse (j→i) for every broken
        # (i→j) to guarantee bond_strength stays symmetric even if prior
        # damage accumulated asymmetrically.
        if edge_index is not None:
            row_arr = edge_index[0]
            col_arr = edge_index[1]
            n_edges = int(bond_alive.shape[0])
            # Build (row, col) -> edge_idx map once per call (O(n_edges))
            edge_to_idx = {(int(row_arr[e]), int(col_arr[e])): e
                           for e in range(n_edges)}
            for e in cond.nonzero(as_tuple=False).squeeze(1).tolist():
                rev = edge_to_idx.get((int(col_arr[e]), int(row_arr[e])), -1)
                if rev >= 0:
                    bond_alive[rev] = 0.0
                    bond_strength[rev] = 0.0
    return cond.sum().item()


def compute_damage(edge_index, bond_alive, n_nodes):
    """Damage phi_i = (broken neighbours) / (total neighbours).

    Sums over BOTH endpoints (row and col) of each directed edge so that
    an undirected bond contributes to both of its incident nodes even when
    the underlying graph stencil is directionally asymmetric (which it is
    here: np.arange in build_rectangular_graph yields a half-FP-shifted
    quadrature grid that omits the (0,-3) and (-3,0) offsets while keeping
    their mirrors).  This makes damage a function of the bond set alone
    and removes the spurious y-asymmetry of the directed-row-only count.
    """
    dev = bond_alive.device
    row = edge_index[0].to(dev)
    col = edge_index[1].to(dev)
    broken = (1.0 - bond_alive).unsqueeze(1)
    ones = torch.ones_like(broken)
    bc = (unsorted_segment_sum(broken, row, n_nodes)
          + unsorted_segment_sum(broken, col, n_nodes)).squeeze()
    tc = (unsorted_segment_sum(ones, row, n_nodes)
          + unsorted_segment_sum(ones, col, n_nodes)).squeeze()
    return bc / (tc + 1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════

def _apply_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 20, 'axes.labelsize': 24, 'axes.titlesize': 26,
        'xtick.labelsize': 20, 'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'axes.edgecolor': '#333333', 'axes.labelcolor': '#222222',
        'text.color': '#222222', 'axes.linewidth': 1.1,
        'axes.grid': False,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2.2, 'lines.markersize': 7,
        'legend.frameon': False,
    })


COLORS = ['#6C9BC2', '#E07B54', '#8FBE7A', '#C490D1', '#E6C960', '#7CD1C8']


def plot_damage(x, damage, Nx, Ny, step, t, out_dir):
    _apply_style()
    import matplotlib.pyplot as plt
    d2d = damage.cpu().numpy().reshape(Nx, Ny).T
    xc = x[:, 0].cpu().numpy()
    yc = x[:, 1].cpu().numpy()

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(d2d, origin='lower', cmap='hot_r', vmin=0, vmax=0.6,
                   extent=[xc.min(), xc.max(), yc.min(), yc.max()],
                   aspect='equal')
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Damage $\\phi$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Step {step},  $T$ = {t:.5f}')
    plt.savefig(Path(out_dir) / f'damage_{step:06d}.png', dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_displacement(x, u, Nx, Ny, step, t, out_dir):
    """Plot the x- and y-components of the displacement field side by side."""
    _apply_style()
    import matplotlib.pyplot as plt
    ux = u[:, 0].cpu().numpy().reshape(Nx, Ny).T
    uy = u[:, 1].cpu().numpy().reshape(Nx, Ny).T
    xc = x[:, 0].cpu().numpy()
    yc = x[:, 1].cpu().numpy()
    extent = [xc.min(), xc.max(), yc.min(), yc.max()]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, field, label in ((axes[0], ux, '$u_x$'), (axes[1], uy, '$u_y$')):
        vmax = np.abs(field).max()
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(field, origin='lower', cmap='coolwarm',
                       vmin=-vmax, vmax=vmax,
                       extent=extent, aspect='equal')
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(label)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(f'{label},  step {step},  $T$ = {t:.5f}')

    plt.tight_layout()
    plt.savefig(Path(out_dir) / f'disp_{step:06d}.png', dpi=300,
                bbox_inches='tight')
    plt.close()


def plot_bonds_across_y0(x, edge_index, bond_alive, out_dir, y0=0.0, epsilon=1e-8):
    """Visualize the directed bonds counted by ``compute_force_across_y0``:
    every bond (row, col) whose *reference* endpoints straddle the cut line
    y = y0 (row strictly below the line, col at/above it -- the same
    ``crosses`` condition used inside ``compute_force_across_y0``, so each
    geometric bond is drawn once, not twice). Bonds are colored by whether
    they are still intact (``bond_alive``) or already broken. The crossing
    set only depends on the fixed reference geometry, so this is a single
    static diagnostic (not a per-step snapshot).
    """
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    xc = x.cpu().numpy()
    row, col = edge_index
    row_n, col_n = row.cpu().numpy(), col.cpu().numpy()
    # Same single-threshold convention as compute_force_across_y0 (see the
    # note there): using row<y0-eps / col>=y0+eps independently creates a
    # dead zone that drops every bond touching y0 exactly.
    crosses = (xc[row_n, 1] <= (y0 + epsilon)) & (xc[col_n, 1] >= (y0 - epsilon))
    idx = np.nonzero(crosses)[0]

    alive_np = bond_alive.cpu().numpy()
    segs = [[xc[row_n[e]], xc[col_n[e]]] for e in idx]
    seg_colors = [COLORS[2] if alive_np[e] > 0.5 else '#BBBBBB' for e in idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(xc[:, 0], xc[:, 1], s=3, color='#DDDDDD', zorder=1)
    if segs:
        lc = LineCollection(segs, colors=seg_colors, linewidths=1.2, zorder=2)
        ax.add_collection(lc)
    ax.axhline(y0, color='black', linestyle='--', linewidth=1.0, zorder=0)

    n_alive = int((alive_np[idx] > 0.5).sum()) if len(idx) else 0
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    ax.set_title(f'Bonds crossing $y={y0:g}$: {len(idx)} total, {n_alive} intact')
    plt.savefig(Path(out_dir) / 'bonds_y0.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_history(hist, out_dir):
    _apply_style()
    import matplotlib.pyplot as plt
    t = hist['time']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].plot(t, hist['disp_applied'], color=COLORS[0])
    axes[0, 0].set_ylabel('Applied displacement')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_title('Applied pull displacement')

    axes[0, 1].plot(t, hist['n_broken'], color=COLORS[1])
    axes[0, 1].set_ylabel('Total broken bonds')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_title('Broken bonds')

    axes[1, 0].plot(t, hist['max_damage'], color=COLORS[2])
    axes[1, 0].set_ylabel('Max damage')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_title('Maximum damage')

    axes[1, 1].plot(t, hist['max_disp_y'], color=COLORS[3])
    axes[1, 1].set_ylabel('Max $|u_y|$')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_title('Maximum y-displacement')

    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'history.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_force_curve(hist, out_dir, L=None, pull_rate=None):
    """Reaction (pulling) force at the top/bottom Dirichlet strips, and the
    cross-check bond force transmitted across the notch plane y=0, y=+L,
    and y=-L: both vs. time and vs. the prescribed pulling displacement."""
    _apply_style()
    import matplotlib.pyplot as plt
    t = hist['time']
    disp = hist['disp_applied']

    yL_lbl  = f'$y=L$'  if L is None else f'$y={L:.4g}$'
    ynL_lbl = f'$y=-L$' if L is None else f'$y=-{L:.4g}$'
    rate_lbl = 't' if pull_rate is None else f'{pull_rate:.4g}t'

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    for ax, xs in ((axes[0], t), (axes[1], disp)):
        ax.plot(xs, hist['reaction_top_y'], color=COLORS[0], label='top strip')
        ax.plot(xs, hist['reaction_bot_y'], color=COLORS[1], label='bottom strip')
        ax.plot(xs, hist['force_y0_y'],  color=COLORS[2], linestyle='--', label='across $y=0$')
        ax.plot(xs, hist['force_yL_y'],  color=COLORS[3], linestyle='--', label=f'across {yL_lbl}')
        ax.plot(xs, hist['force_ynL_y'], color=COLORS[4], linestyle='--', label=f'across {ynL_lbl}')
        ax.plot(xs, hist['force_y0_elastic'], color='k', linestyle=':',
                label=r'elastic $E\,\epsilon_{yy}\,(2L\,H)$')
        ax.set_ylabel('Force $F_y$')
        ax.legend()
    axes[0].set_xlabel('Time')
    axes[0].set_title('Reaction force vs. time')
    axes[1].set_xlabel(f'Prescribed pull displacement ${rate_lbl}$')
    axes[1].set_title('Force-displacement curve')

    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'force.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_force_history(hist, out_dir):
    """Write the per-step force history (reaction forces at the top/bottom
    Dirichlet strips and bond forces across y=0, y=+L, y=-L) to force.txt."""
    cols = ['time', 'disp_applied', 'reaction_top_y', 'reaction_bot_y',
            'force_y0_x', 'force_y0_y',
            'force_yL_x', 'force_yL_y',
            'force_ynL_x', 'force_ynL_y',
            'force_y0_elastic']
    header = '  '.join(f'{c:>16s}' for c in cols)
    n_rows = len(hist['time'])
    with open(Path(out_dir) / 'force.txt', 'w') as f:
        f.write('# ' + header + '\n')
        for i in range(n_rows):
            row = '  '.join(f'{hist[c][i]:16.8e}' for c in cols)
            f.write('  ' + row + '\n')


# ═══════════════════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════════════════

def run_simulation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Precision selection.  The PNO checkpoint stores float32 parameters;
    # we cast the model to the chosen dtype after loading so every forward
    # pass and every state tensor (u, v, a, ...) live in that dtype.
    if args.precision == "float64":
        dtype = torch.float64
    elif args.precision == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"--precision must be float32 or float64, got {args.precision!r}")
    torch.set_default_dtype(dtype)
    print(f"Precision: {torch.get_default_dtype()}")

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ───────────────────────────────────
    summary = load_summary(model_dir)
    model = build_model(summary, model_dir, device).to(dtype)
    alpha = model.get_alpha().item()
    # Kernel symmetrization is always on.
    model.k = SymmetrizedK(model.k)
    kernel_tag = "k_sym(zeta) = (k(zeta) + k(-zeta))/2  [symmetrized, always on]"
    dx = args.dx
    print(f"Model: {summary['run_tag']}  alpha={alpha:.4f}")
    print(f"Operator: norm variant  phi = g * k(xi/delta) * (|xi|/delta)^(-alpha)")
    print(f"Kernel: {kernel_tag}")
    print(f"Integration: Riemann  dx^2 = {dx**2:.4f}")

    # ── 1a. Training-unit -> MD-unit conversion (derivation eq. 14) ─────
    (unit_s, unit_sf, unit_fac, delta_train, delta_md,
     unit_source) = resolve_training_units(summary, args.unit_s,
                                           args.unit_sigma_f)
    print(f"Training units: (s, sigma_f)=({unit_s:g}, {unit_sf:g})  "
          f"[{unit_source}]")
    print(f"  (g*k)_MD = (g*k)_learned * s^2/sigma_f = "
          f"(g*k)_learned * {unit_fac:g}")
    print(f"  delta_train={delta_train:g}  ->  delta_MD = "
          f"delta_train/s = {delta_md:g} lu")
    # The horizon is a property of the LOADED MODEL, not of the grid:
    # delta = gamma * delta_MD, INDEPENDENT of dx.  Refining dx only
    # refines the quadrature of the same physical operator (more bonds
    # inside the fixed horizon); it no longer shrinks the horizon.
    # (--gamma still performs a deliberate horizon rescale, with the
    # 1/gamma^3 amplitude factor and the invG critical-stretch remap.)
    delta = args.gamma * delta_md
    print(f"Horizon: delta = gamma * delta_MD = {args.gamma:g} * "
          f"{delta_md:g} = {delta:g} lu  (m-ratio delta/dx = "
          f"{delta/dx:.3f}; training used 3.01)")
    print(f"  MD->SI: dx = {dx:g} lu = {dx*1e-10:.3e} m;  "
          f"delta = {delta*1e-10:.3e} m;  dt = {args.dt:g} tu = "
          f"{args.dt*1e-12:.3e} s")
    # ── 1b. Locate the learned-g peak; resolve lam_cut and s0 ───────────
    lam_peak, g_peak = find_learned_g_peak(model, device=device)
    print(f"Learned g peak: lam_peak={lam_peak:.4f}  g_peak={g_peak:.4e}"
          f"  -> peak stretch s_peak={lam_peak - 1.0:.4f}")

    if args.g_rule == "linear":
        # LINEARIZED g: replace the learned g(lambda) by its tangent at
        # lambda = 1, with the slope identified from the SMALL-
        # DEFORMATION tension range 0 <= lambda - 1 <= 0.01 (least-
        # squares linear fit of the learned g over that window):
        #     g_lin(lambda) = g(1) + g'(1) * (lambda - 1).
        # The material is then exactly linear at all stretches -- no
        # softening branch, forces scale linearly with the applied
        # displacement (up to geometric nonlinearity of the bond
        # stretch), which makes the comparison with the continuum
        # elasticity estimate E*eps_yy*(2L*H) clean.
        with torch.no_grad():
            lam_fit = torch.linspace(1.0, 1.01, 201, device=device)
            g_fit = model.signed_g_difference(
                lam_fit.reshape(-1, 1)).reshape(-1)
            s_fit = lam_fit - 1.0
            sbar = s_fit.mean()
            gbar = g_fit.mean()
            g_slope = float(((s_fit - sbar) * (g_fit - gbar)).sum()
                            / ((s_fit - sbar) ** 2).sum())
            g_at_1 = float(gbar - g_slope * sbar)

        def g_fn(lambdaa):
            return g_at_1 + g_slope * (lambdaa - 1.0)
        print(f"  g: LINEARIZED  g_lin(lam) = g(1) + g'(1)*(lam-1), "
              f"g'(1) = {g_slope:.6g}, g(1) = {g_at_1:.3g} "
              f"(least-squares fit over 0 <= lam-1 <= 0.01)")
    else:
        # RAW learned g: no cutoff, no softening of any kind -- every
        # bond evaluates the network's g(lambda) as trained.
        def g_fn(lambdaa):
            return model.signed_g_difference(lambdaa)
        print("  g: RAW learned g (no cutoff tail, no bondsoft -- "
              "fracture disabled)")

    s0 = lam_peak - 1.0           # critical stretch = peak stretch (fixed)
    print(f"  s0 = lam_peak - 1 = {s0:.4f}")

    # ── 1c. Compute effective softening point s_tilde_0 via invG ─────────
    gamma_scale = args.gamma
    # Fold the training-unit amplitude conversion (s^2/sigma_f) into the
    # operator's scalar multiplier.  unit_fac only rescales g*k uniformly,
    # so lambda, the g peak, s0, and the invG remap are all unaffected.
    rescale_factor = unit_fac
    if unit_fac != 1.0:
        print(f"  operator amplitude: rescale_factor = unit_fac = {unit_fac:g}")
    s_tilde_0 = invG(gamma_scale, g_fn, model, s0, BONDSOFT_DECAY,
                     device=device)
    if np.isnan(s_tilde_0):
        print(f"  WARNING: invG returned nan for gamma={gamma_scale:.4f}; "
              f"falling back to s0={s0:.4f}")
        s_tilde_0 = s0
    else:
        print(f"  gamma={gamma_scale:.4f}  s_tilde_0={s_tilde_0:.6f} "
              f"(= invG({gamma_scale:.4f}, s0={s0:.4f}))")

    # ── 2. Domain: [-L, L] x [-2L, 2L], centered at origin ────────────
    L = args.L
    Lx_target = 2.0 * L
    # Staggered geometry: total height 4L + dx gives an EVEN
    # number of rows placed at y = +-dx/2, +-3dx/2, ..., +-(2L + dx/2):
    # the crack/measurement plane y = 0 lies BETWEEN the two middle
    # layers, and y = +-L also falls between rows.
    Ly_target = 4.0 * L + dx
    Nx = int(round(Lx_target / dx)) + 1
    Ny = int(round(Ly_target / dx)) + 1
    N = Nx * Ny
    x = create_rectangular_grid(Nx, Ny, dx)
    Lx = (Nx - 1) * dx
    Ly = (Ny - 1) * dx
    x[:, 0] -= Lx / 2.0
    x[:, 1] -= Ly / 2.0
    y_top, y_bot = Ly / 2.0, -Ly / 2.0
    print(f"Grid : Nx={Nx}  Ny={Ny}  dx={dx:.3f}  delta={delta:.4f}")
    print(f"Domain: [{-Lx/2.0:.1f}, {Lx/2.0:.1f}] x [{y_bot:.1f}, {y_top:.1f}]"
          f"  (L={L:.1f})  N={N}")

    # ── 3. Graph ─────────────────────────────────────────────────────
    print("Building graph ...")
    t0 = time.time()
    edge_index, _, _ = build_rectangular_graph(Nx, Ny, dx, delta)
    n_edges = edge_index.shape[1]
    print(f"  edges={n_edges}  ({time.time()-t0:.1f}s)")

    # ── 4. (norm variant) Riemann integration: no quadrature weights needed ──
    print(f"Integration: Riemann  w_j = dx^2 = {dx**2:.4f} per bond")

    # ── 5. Pristine plate: no pre-notch, fracture disabled ────────────
    notch_y = 0.0
    bond_alive = torch.ones(edge_index.shape[1], device=device)
    bond_strength = bond_alive.clone()
    isolated0 = compute_isolated_nodes(edge_index, bond_alive, N)  # none
    print("  Pristine plate: no pre-notch; bond breaking DISABLED "
          "(pure elastic reference)")

    # Diagnostic: which bonds compute_force_across_y0 sums over (static,
    # depends only on the reference geometry, not on time).
    plot_bonds_across_y0(x, edge_index, bond_alive, out_dir, y0=notch_y)

    # ── 6. Boundary conditions ────────────────────────────────────────
    # Top/bottom strips of thickness --strip-width: y-velocity pinned to
    # a constant pull rate (+0.025 top, -0.025 bottom), i.e. y-displacement
    # = +-0.025 t; x is left completely free in these strips. Center
    # vertical line x=0: x-displacement pinned to 0 for all time (symmetry
    # condition preventing horizontal rigid-body drift); y is free there.
    strip_width = args.strip_width
    # grips: |y| >= 2L - strip_width (NOT y_top - strip_width: y_top is
    # 2L + dx/2 here) -- for the defaults, the rows beyond |y| = 90.
    top_strip = x[:, 1] >= (2.0 * L - strip_width)
    bot_strip = x[:, 1] <= -(2.0 * L - strip_width)
    PULL_RATE = args.pull_rate
    eps_x0 = 1e-3 * dx
    x0_mask = x[:, 0].abs() < eps_x0

    dirichlet_mask = torch.zeros(N, 2, dtype=torch.bool, device=device)
    dirichlet_mask[:, 1] = (top_strip | bot_strip).to(device)
    dirichlet_mask[:, 0] = x0_mask.to(device)

    def dirichlet_target(t):
        u_d = torch.zeros(N, 2, device=device)
        u_d[top_strip, 1] = PULL_RATE * t
        u_d[bot_strip, 1] = -PULL_RATE * t
        # u_d[x0_mask, 0] stays 0.
        return u_d

    print(f"  Boundary strips: width={strip_width:.3f}  "
          f"top_nodes={top_strip.sum().item()}  bot_nodes={bot_strip.sum().item()}  "
          f"pull_rate=+-{PULL_RATE:.4g} (y only, x free)")
    print(f"  Center line x=0: {x0_mask.sum().item()} nodes pinned "
          f"(x-displacement=0 for all time, y free)")

    # ── No fracture: bond breaking disabled for EVERY bond ────────────
    # (update_bonds only acts on edges flagged breakable; an all-False
    # mask makes it a guaranteed no-op, so the damage state never
    # changes and every step is a purely elastic equilibrium.)
    breakable = torch.zeros(edge_index.shape[1], dtype=torch.bool,
                            device=device)

    # ── 6b. Free-surface treatment ────────────────────────────────────
    ghosts = g_rows = g_idx = p_blocks = None
    if args.boundary == "traction":
        ghosts = build_ghosts(x, dx, delta, Lx / 2.0, -1.0,
                              notch_y=notch_y)      # notch disabled
        g_rows, g_idx = build_ghost_edges(x, ghosts, delta, -1.0,
                                          notch_y=notch_y,
                                          exclude=isolated0.cpu())
        for key in ("pos", "b_idx", "m_idx", "c_b", "c_m", "block"):
            ghosts[key] = ghosts[key].to(device)
        if args.ghost_order == 0:
            # Zeroth-order ghost extension u_fict = u(xbar) + p_alpha
            # instead of the linear mirror 2u(xbar) - u(2xbar-x) + p.
            # The mirror's -u(2xbar-x) term acts as an ANTI-spring
            # between each interior receiver and the mirror source
            # node; with the softening learned g this makes the
            # edge-detachment mode a negative-curvature direction of
            # the equilibrium system (measured Rayleigh quotient
            # -0.25 at 1% nominal strain vs +8.7 for generic modes),
            # so long Newton solves drift into a spurious detached-
            # surface state.  Constant extension removes the
            # anti-spring (+0.21 along the same mode) at the cost of
            # exactness for linear displacement fields.
            ghosts["c_b"] = torch.ones_like(ghosts["c_b"])
            ghosts["c_m"] = torch.zeros_like(ghosts["c_m"])
            print("  ghost extension: ZEROTH order (u_fict = u(xbar) + p)")
        p_blocks = torch.zeros(len(BLOCKS), 2, device=device)
        n_per = {b: int((ghosts["block"] == a).sum())
                 for a, b in enumerate(BLOCKS)}
        print(f"  Free surfaces: mirror fictitious nodes + Lu-Li zero "
              f"traction ({ghosts['pos'].shape[0]} ghosts {n_per}, "
              f"{len(g_rows)} ghost bonds)")
    else:
        print("  Free surfaces: do-nothing (truncated horizon)")

    # ── 7. Initialise state (quasistatic: displacement only) ──────────
    u = torch.zeros(N, 2, device=device)
    dt = args.dt   # loading increment: applied displacement = pull_rate*step*dt

    # ── 8. Time stepping ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Simulation (quasistatic):  dt={dt}  n_steps={args.n_steps}  "
          f"s0={s0:.6f}  s_tilde_0={s_tilde_0:.6f}  "
          f"gamma={gamma_scale:.4f}  rescale={rescale_factor:.6g}")
    print(f"{'='*70}")

    hist = {'time': [], 'n_broken': [], 'max_damage': [], 'max_disp_y': [],
            'disp_applied': [], 'reaction_top_y': [], 'reaction_bot_y': [],
            'force_y0_x': [], 'force_y0_y': [],
            'force_yL_x': [], 'force_yL_y': [],
            'force_ynL_x': [], 'force_ynL_y': [],
            'force_y0_elastic': []}
    total_broken = (bond_alive < 0.5).sum().item()

    # Initial snapshot
    damage = compute_damage(edge_index, bond_alive, N)
    plot_damage(x, damage, Nx, Ny, 0, 0.0, out_dir)
    plot_displacement(x, u, Nx, Ny, 0, 0.0, out_dir)

    sim_start = time.time()

    for step in range(1, args.n_steps + 1):
        t_sim = step * dt

        u_prev = u
        u_d_t = dirichlet_target(t_sim)

        if args.boundary == "traction":
            (u, n_iter, converged, n_new, f_int, p_blocks,
             t_res) = solve_quasistatic_step_traction(
                model, u_prev, dirichlet_mask, u_d_t,
                x, edge_index, delta, dx,
                bond_alive, bond_strength, breakable, s_tilde_0,
                BONDSOFT_DECAY, g_fn, device,
                ghosts, g_rows, g_idx, p_blocks,
                newton_tol=args.newton_tol,
                newton_max_iter=args.newton_max_iter,
                cg_tol=args.cg_tol, cg_max_iter=args.cg_max_iter,
                traction_iters=args.traction_iters,
                p_newton_iters=args.p_newton_iters,
                verbose=(args.verbose_newton or step == 1),
                gamma_scale=gamma_scale,
                rescale_factor=rescale_factor)
        else:
            t_res = None
            u, n_iter, converged, n_new, f_int = solve_quasistatic_step(
                model, u_prev, dirichlet_mask, u_d_t,
                x, edge_index, delta, dx,
                bond_alive, bond_strength, breakable, s_tilde_0,
                BONDSOFT_DECAY, g_fn, device,
                newton_tol=args.newton_tol,
                newton_max_iter=args.newton_max_iter,
                cg_tol=args.cg_tol, cg_max_iter=args.cg_max_iter,
                verbose=(args.verbose_newton or step == 1),
                gamma_scale=gamma_scale,
                rescale_factor=rescale_factor)
        # Quasistatic force balance: the reaction ("would-be" external
        # force) at every dof is simply -f_int.
        reaction_full = -f_int

        total_broken += n_new

        reaction_top_y = reaction_full[top_strip, 1].sum().item()
        reaction_bot_y = reaction_full[bot_strip, 1].sum().item()

        # Cross-check: bond force across y=0, y=+L, y=-L (independent of
        # the nodal-residual boundary reaction force above).
        with torch.no_grad():
            force_y0 = compute_force_across_y0(
                model, u, x, edge_index, delta, dx, bond_strength, device,
                g_fn=g_fn, gamma_scale=gamma_scale,
                rescale_factor=rescale_factor, y0=0.0)
            force_yL = compute_force_across_y0(
                model, u, x, edge_index, delta, dx, bond_strength, device,
                g_fn=g_fn, gamma_scale=gamma_scale,
                rescale_factor=rescale_factor, y0=L)
            force_elastic = compute_force_elastic(u, Nx, Ny, dx, Lx,
                                                  device)
            if args.boundary == "traction":
                _, wg_now, _ = internal_force_mirror(
                    model, u, x, edge_index, bond_strength, ghosts,
                    g_rows, g_idx, p_blocks, delta, dx, device,
                    g_fn=g_fn, gamma_scale=gamma_scale,
                    rescale_factor=rescale_factor,
                    return_ghost_edge_forces=True)
                force_y0 = force_y0 + ghost_force_across_y0(
                    x, ghosts, g_rows, g_idx, wg_now, dx, y0=0.0)
                force_yL = force_yL + ghost_force_across_y0(
                    x, ghosts, g_rows, g_idx, wg_now, dx, y0=L)
            force_ynL = compute_force_across_y0(
                model, u, x, edge_index, delta, dx, bond_strength, device,
                g_fn=g_fn, gamma_scale=gamma_scale,
                rescale_factor=rescale_factor, y0=-L)
            if args.boundary == "traction":
                force_ynL = force_ynL + ghost_force_across_y0(
                    x, ghosts, g_rows, g_idx, wg_now, dx, y0=-L)

        # Monitoring
        max_disp_y = u[:, 1].abs().max().item()
        damage = compute_damage(edge_index, bond_alive, N)
        max_dam = damage.max().item()

        hist['time'].append(t_sim)
        hist['n_broken'].append(total_broken)
        hist['max_damage'].append(max_dam)
        hist['max_disp_y'].append(max_disp_y)
        hist['disp_applied'].append(PULL_RATE * t_sim)
        hist['reaction_top_y'].append(reaction_top_y)
        hist['reaction_bot_y'].append(reaction_bot_y)
        hist['force_y0_x'].append(force_y0[0].item())
        hist['force_y0_y'].append(force_y0[1].item())
        hist['force_yL_x'].append(force_yL[0].item())
        hist['force_yL_y'].append(force_yL[1].item())
        hist['force_ynL_x'].append(force_ynL[0].item())
        hist['force_ynL_y'].append(force_ynL[1].item())
        hist['force_y0_elastic'].append(force_elastic)

        if step % args.print_interval == 0 or n_new > 0 or not converged:
            tag = "" if converged else "  [Quasistatic Newton NOT converged]"
            print(f"  step {step:6d}  t={t_sim:.5f}  "
                  f"|u_y|max={max_disp_y:.4e}  dam={max_dam:.3f}  "
                  f"F_top={reaction_top_y:.4e}  F_bot={reaction_bot_y:.4e}  "
                  f"F_y[0]={force_y0[1].item():.4e}  "
                  f"F_y[+L]={force_yL[1].item():.4e}  "
                  f"F_y[-L]={force_ynL[1].item():.4e}  "
                  f"F_E={force_elastic:.4e}  "
                  f"broken={total_broken} (+{n_new})  newton_it={n_iter}"
                  + (f"  |t_res|max={float(t_res.norm(dim=1).max()):.2e}"
                     if t_res is not None else "") + tag)

        # Snapshot
        if step % args.snapshot_interval == 0:
            plot_damage(x, damage, Nx, Ny, step, t_sim, out_dir)
            plot_displacement(x, u, Nx, Ny, step, t_sim, out_dir)
            extra = ({'p_blocks': p_blocks.cpu().numpy()}
                     if p_blocks is not None else {})
            np.savez(out_dir / f'state_{step:06d}.npz',
                     x=x.cpu().numpy(), u=u.cpu().detach().numpy(),
                     damage=damage.cpu().numpy(),
                     bond_alive=bond_alive.cpu().numpy(),
                     bond_strength=bond_strength.cpu().numpy(),
                     t=t_sim, step=step, **extra)

        # Sanity guard
        if torch.isnan(u).any():
            print(f"  *** NaN displacement at step {step}; aborting.")
            break

    elapsed = time.time() - sim_start
    print(f"\nSimulation finished in {elapsed:.1f}s  ({elapsed/max(step,1):.3f}s/step)")

    # ── 9. Save outputs ──────────────────────────────────────────────
    np.savez(out_dir / 'history.npz',
             **{k: np.array(v) for k, v in hist.items()})

    config = vars(args).copy()
    config.update(delta=delta, alpha=alpha, N=N, n_edges=n_edges,
                  Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
                  total_broken=total_broken, elapsed_s=elapsed,
                  unit_s=unit_s, unit_sigma_f=unit_sf, unit_fac=unit_fac,
                  delta_train=delta_train, delta_md=delta_md,
                  rescale_factor_effective=rescale_factor)
    json.dump(config, open(out_dir / 'config.json', 'w'), indent=2,
              default=str)

    plot_history(hist, out_dir)
    plot_force_curve(hist, out_dir, L=L, pull_rate=PULL_RATE)
    save_force_history(hist, out_dir)

    # Final damage
    damage = compute_damage(edge_index, bond_alive, N)
    plot_damage(x, damage, Nx, Ny, step, step * dt, out_dir)

    print(f"Results saved to {out_dir}/")
    return hist


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=str,
                   default="../models/nrm_n256_lr3e4",
                   help="Directory of the trained PNO checkpoint "
                        "(model.ckpt + summary.json).")
    p.add_argument("--unit-s", type=float, default=0.1,
                   help="Training scaling s applied to (x, u) for this "
                        "checkpoint (fracture_energy_derivation eq. 14). "
                        "Default: auto-infer from summary.json delta "
                        "(delta > 5 -> 1.0, else 0.1) with per-run "
                        "overrides (see UNIT_OVERRIDES).")
    p.add_argument("--unit-sigma-f", type=float, default=0.01,
                   help="Training scaling sigma_f applied to f for this "
                        "checkpoint.  Default: auto-infer alongside "
                        "--unit-s.  The operator output is multiplied by "
                        "s^2/sigma_f to convert to MD units.")
    p.add_argument("--L", type=float, default=50.0,
                   help="Domain half-width; domain is [-L, L] x [-2L, 2L] "
                        "(default 50.0).")
    p.add_argument("--dx", type=float, default=5.0,
                   help="Grid spacing (default 5.0). Nx, Ny are derived "
                        "from L and dx. The horizon is NOT tied to dx: "
                        "delta = gamma * delta_MD from the loaded model; "
                        "refining dx only refines the quadrature.")
    p.add_argument("--g-rule", choices=("linear", "learned"),
                   default="linear",
                   help="Constitutive g: 'linear' (default) replaces the "
                        "learned g by its tangent at lambda=1, with the "
                        "slope least-squares fitted over the small-"
                        "deformation window 0 <= lambda-1 <= 0.01; "
                        "'learned' uses the raw network g.")
    p.add_argument("--ghost-order", type=int, default=1, choices=(0, 1),
                   help="Traction-mode ghost extension order: 1 = linear "
                        "mirror (affine-exact but the softening learned g "
                        "makes the edge-detachment mode unstable beyond "
                        "small loads), 0 = constant extension (stable; "
                        "first-order surface error). Only used with "
                        "--boundary traction.")
    p.add_argument("--strip-width", type=float, default=10.0,
                   help="Thickness of the top/bottom boundary strips within "
                        "which the y-displacement is prescribed "
                        "(x is always free). Default 10.0.")
    p.add_argument("--pull-rate", type=float, default=0.025,
                   help="Constant y-velocity prescribed on the top strip "
                        "(+pull_rate) and bottom strip (-pull_rate), i.e. "
                        "y-displacement = +-pull_rate * t. Default 0.025.")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Horizon rescale gamma (> 0): the simulation "
                        "horizon is delta = gamma * delta_MD (the model"
                        "'s native horizon), L[u] is divided by gamma^3, "
                        "and the softening onset s0 is replaced by "
                        "s_tilde_0 = invG(gamma, s0). Default 1.0.")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Loading increment: the applied grip displacement "
                        "grows by pull_rate*dt per step (quasistatic; no "
                        "physical time).")
    p.add_argument("--n-steps", type=int, default=200,
                   help="Number of loading steps (default 200)")
    p.add_argument("--snapshot-interval", type=int, default=1,
                   help="Steps between snapshot outputs")
    p.add_argument("--print-interval", type=int, default=50,
                   help="Steps between console prints")
    p.add_argument("--boundary", choices=["nothing", "traction"],
                   default="nothing",
                   help="Free-surface treatment of the left/right edges and "
                        "the crack faces. 'nothing' (default): truncated "
                        "horizon, no correction. 'traction': mirror-type "
                        "fictitious nodes with the Lu-Li zero-traction "
                        "condition -- per free surface a rigid ghost shift "
                        "p_alpha is solved so the summed ghost->interior "
                        "bond force vanishes (traction-free surfaces).")
    p.add_argument("--traction-iters", type=int, default=3,
                   help="[--boundary traction] alternations between the "
                        "interior Newton solve and the p_alpha shift solve "
                        "per loading step (default 3).")
    p.add_argument("--p-newton-iters", type=int, default=4,
                   help="[--boundary traction] per-block 2x2 Newton "
                        "iterations of the zero-traction shift solve "
                        "(default 4).")
    # ── Newton-Krylov knobs ────────────────────────────────────────────
    p.add_argument("--newton-tol", type=float, default=1e-4,
                   help="Newton relative residual tolerance per implicit step. "
                        "Default 1e-4 reflects the ReLU-MLP noise floor of the "
                        "learned g/k networks; tighten only if the MLP is C^1.")
    p.add_argument("--newton-max-iter", type=int, default=30,
                   help="Max Newton iterations per implicit step.")
    p.add_argument("--cg-tol", type=float, default=1e-3,
                   help="CG inner tolerance for the LM normal equations.")
    p.add_argument("--cg-max-iter", type=int, default=100,
                   help="CG max inner iterations.")
    p.add_argument("--verbose-newton", action="store_true", default=True,
                   help="Print per-iteration Newton diagnostics "
                        "(on by default).")
    p.add_argument("--precision", type=str, default="float64",
                   choices=["float32", "float64"],
                   help="Floating-point precision for the entire driver "
                        "(model, state, quadrature). Default: float64.")
    p.add_argument("--out-dir", type=str,
                   default="results_force_nofracture",
                   help="Output directory")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(args)
