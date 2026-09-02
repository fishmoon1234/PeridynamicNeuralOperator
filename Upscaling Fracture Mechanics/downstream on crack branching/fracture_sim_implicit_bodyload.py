#!/usr/bin/env python3
"""Implicit-Newmark dynamic fracture with BODY-FORCE loading on
delta-thick skins (successor of fracture_sim_implicit_traction.py; that
script is left untouched).

Differences from fracture_sim_implicit_traction.py
--------------------------------------------------
* STAGGERED GRID (as in nofracture_estimate_force_silling_Dirichlet.py /
  the *_middle drivers): Ny is forced EVEN, so with the crack plane at
  notch_y = Ly/2 = (Ny-1)/2 * dx (a half-integer multiple of dx) the
  pre-notch runs BETWEEN two layers of grid points -- no node lies on
  the crack plane.  The domain is [0, Lx] x [0, Ly] with
  Ly = (Ny-1)*dx (one extra dx of height compared to the odd-Ny parent).
* The mirror-ghost / Lu-Li traction machinery is REMOVED entirely.
  Loading is a uniform body-force density on the delta-thick top and
  bottom skins,

      b_top    = (0, +traction / ((Nx-1) dx * delta * H_code)),
      b_bottom = (0, -traction / ((Nx-1) dx * delta * H_code)),

  i.e. the prescribed total force +-traction spread over the skin
  volume (Nx-1)dx * delta * H_code.  The slab thickness is a PHYSICAL
  constant, 0.935 lu for every gamma; in the gamma-scaled length units
  of this driver its code value is H_code = 0.935/gamma.
* ALL boundaries (left/right edges, loaded skins, crack faces) are
  do-nothing (truncated horizon): no ghosts, no Dirichlet constraint.
  The +-b loading is antisymmetric so the net force is zero; rigid
  modes are controlled by inertia (pure Neumann dynamics, as in the
  body-force parent fracture_sim_implicit_norm.py).
* Units and horizon: lengths are in GAMMA-SCALED units (1 code unit =
  gamma lu; dx = 5 means gamma*5 lu); gamma enters the operator through
  the 1/gamma^3 amplitude and the invG rescaling law of g: the critical
  stretch is remapped to s_tilde_0 = invG(gamma, s0) and the SOFTENING
  CUT of g is relocated to the same remapped threshold (learned g up to
  1 + s_tilde_0, C1 exponential tail beyond), so softening and breaking
  happen together at the rescaled stretch.  delta is never multiplied
  by gamma.  The training stencil is fixed at
  m = 3.01 with the horizon following the grid: delta = 3.01*dx
  (default dx = 5 -> delta = 15.05).
* Symmetrized kernel k_sym(zeta) = (k(zeta)+k(-zeta))/2 by default
  (--raw-kernel restores the raw network).
* Time integration: implicit Newmark-beta (average acceleration),
      R(u) = rho/(dt^2 beta) (u - u_tilde) - f_int(u) - b,
  solved by the globalized LM Newton with mask stabilization.

Usage:
    python fracture_sim_implicit_bodyload.py \
        --model-dir ../models/nrm_xu01f001 --traction 0.15 [options]
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
from utilities_INO_PD import parse_layer_info


# ═══════════════════════════════════════════════════════════════════════
#  Inlined shared machinery (verbatim copies from the mirror driver, so
#  this script is fully self-contained; only the library modules
#  egnn_gcl / utilities_INO_PD are imported)
# ═══════════════════════════════════════════════════════════════════════

# Runs whose training scaling does not follow the delta-based convention.
UNIT_OVERRIDES = {
    "nrm_xu01f001": (0.1, 0.01),   # x,u scaled by 0.1, f by an extra 0.01
}

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


def find_learned_g_peak(model, lam_min=1.001, lam_max=1.5, n=4001,
                        device="cpu"):
    """Locate the tension-side peak of the learned g(lambda)."""
    lam = torch.linspace(lam_min, lam_max, n, device=device)
    with torch.no_grad():
        g = model.signed_g_difference(lam).reshape(-1)
    i = int(torch.argmax(g).item())
    return float(lam[i].item()), float(g[i].item())


def build_learned_cutoff_g(model, lam_cut, decay, device="cpu"):
    """Return a g_fn that uses the learned g for lam <= lam_cut and a
    C^1-matched exponential decay tail past lam_cut.

        g(lam) = g_learned(lam),                            lam <= lam_cut
        g(lam) = exp(-k s) * (g_cut + a s),                 lam >  lam_cut

    where s = lam - lam_cut, g_cut = g_learned(lam_cut),
          a = s_cut + k * g_cut, s_cut = g_learned'(lam_cut).
    """
    h = 1e-3
    dtype = torch.get_default_dtype()
    probe = torch.tensor([lam_cut - h, lam_cut, lam_cut + h],
                          dtype=dtype, device=device)
    with torch.no_grad():
        g_probe = model.signed_g_difference(probe).reshape(-1)
    g_cut = float(g_probe[1].item())
    s_cut = float(((g_probe[2] - g_probe[0]) / (2.0 * h)).item())
    k = float(decay)
    a = s_cut + k * g_cut

    lam_cut_t = torch.tensor(float(lam_cut), dtype=dtype, device=device)
    g_cut_t   = torch.tensor(g_cut,         dtype=dtype, device=device)
    s_cut_t   = torch.tensor(s_cut,         dtype=dtype, device=device)
    a_t       = torch.tensor(a,             dtype=dtype, device=device)
    k_t       = torch.tensor(k,             dtype=dtype, device=device)

    def g_fn(lambdaa):
        lam = lambdaa.reshape(-1, 1)
        g_main = model.signed_g_difference(lam)            # learned, (E,1)
        s = lam - lam_cut_t
        g_tail = torch.exp(-k_t * s) * (g_cut_t + a_t * s)
        return torch.where(lam > lam_cut_t, g_tail, g_main)

    info = {
        "form":         "learned on [.,lam_cut]; exp(-k s)(g_cut + a s) on tail",
        "lam_cut":      float(lam_cut),
        "decay":        k,
        "g_cut":        g_cut,
        "slope_at_cut": s_cut,
        "a":            a,
    }
    return g_fn, info


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


def initialize_prenotch(x, edge_index, notch_x_start, notch_x_end, notch_y, dx):
    """Break bonds crossing horizontal pre-notch [notch_x_start, notch_x_end] at y = notch_y."""

    # NOTE: epsilon must be large enough to survive float32 rounding near
    # the notch-row coordinate (e.g. for notch_y=10 in float32 the local
    # spacing is ~1e-6, so eps=1e-8 collapses to 0 and on-line nodes get
    # classified as neither above nor below).  A small fraction of dx is
    # safe and physically meaningless.
    epsilon = 1e-4 * float(dx)
    row, col = edge_index
    yi = x[row, 1]
    yj = x[col, 1]
    xi = x[row, 0]
    xj = x[col, 0]

    above_i = yi > (notch_y - epsilon)
    below_j = yj < (notch_y + epsilon)
    crosses1 = (above_i & below_j)

    below_i = yi < (notch_y + epsilon)
    above_j = yj > (notch_y - epsilon)
    crosses2 = (below_i & above_j)

    dy = yj - yi
    safe_dy = dy.clone()
    safe_dy[dy.abs() < 1e-30] = 1e-30
    t = (notch_y - yi) / safe_dy
    x_int = x[row, 0] + t * (x[col, 0] - x[row, 0])

    in_range = (x_int >= notch_x_start - epsilon) & (x_int <= notch_x_end + epsilon)

    # Bonds lying entirely along y = notch_y (dy ≈ 0): the t=0 convention
    # above sets x_int = xi (source-node x), which misses bonds where xi is
    # *outside* the notch but xj is *inside* -- e.g. a horizontal bond from
    # (x=30,y=0) to (x=25,y=0) gets x_int=30, wrongly surviving.  For these
    # bonds, use the full segment-overlap test: cut the bond if its x-interval
    # [min(xi,xj), max(xi,xj)] overlaps [notch_x_start, notch_x_end].
    on_notch_line = dy.abs() < epsilon
    x_lo = torch.minimum(xi, xj)
    x_hi = torch.maximum(xi, xj)
    in_range_overlap = (x_lo <= notch_x_end + epsilon) & (x_hi >= notch_x_start - epsilon)

    in_range_final = torch.where(on_notch_line, in_range_overlap, in_range)
    broken = (crosses1 | crosses2) & in_range_final

    # Symmetrize: for slanted bonds, x_int is a different floating-point
    # expression for (i->j) than for (j->i) (t vs 1-t from the other end),
    # so the two directions of one geometric bond could in principle land
    # on opposite sides of the in_range threshold.  Kill an edge whenever
    # EITHER direction tests as crossing, so bond_alive is exactly
    # symmetric by construction (matching the update_bonds guarantee).
    n_nodes = x.shape[0]
    key = row * n_nodes + col
    rev_key = col * n_nodes + row
    order = torch.argsort(key)
    pos = torch.searchsorted(key[order], rev_key).clamp(max=key.shape[0] - 1)
    rev_idx = order[pos]
    has_rev = key[rev_idx] == rev_key
    broken = broken | (has_rev & broken[rev_idx])

    bond_alive = torch.ones(edge_index.shape[1], dtype=torch.get_default_dtype())
    bond_alive[broken] = 0.0
    print(f"  Pre-notch: {broken.sum().item()} bonds broken "
          f"(x=[{notch_x_start:.1f}, {notch_x_end:.1f}], y={notch_y:.1f})")
    return bond_alive


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


def internal_force(model, u, x, edge_index, bond_strength,
                   delta, dx, device, g_fn=None,
                   gamma_scale=1.0, rescale_factor=1.0,
                   return_edge_stretch=False):
    """f_int over the interior bonds only (do-nothing free surfaces:
    truncated horizons, no ghosts).  Differentiable in u."""
    N = x.shape[0]
    row = edge_index[0].to(device)
    col = edge_index[1].to(device)
    bs = bond_strength.to(device)

    ksi = x[col] - x[row]
    eta = u[col] - u[row]
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
    force = unsorted_segment_sum(weighted, row, num_segments=N)
    if return_edge_stretch:
        return force, (lambdaa[:, 0] - 1.0).detach()
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
        F  torch.Tensor [2]  total force vector across the cut.
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
        # physical MD force: phi*dx^2 per bond times source volume
        # dx^2*H (share-driver convention).  H is the PHYSICAL slab
        # thickness 0.935 lu = 0.935/gamma in this driver's
        # gamma-scaled code length units.
        weighted = (weighted / (gamma_scale ** 3) * rescale_factor
                    * (dx ** 4) * (0.935 / gamma_scale))

        # NOTE: both sides of the split must share the SAME threshold
        # (y0-epsilon), not row<y0-eps / col>=y0+eps: with two separate
        # +-eps offsets, any node lying exactly ON y=y0 (extremely common
        # here -- y0=0 is an exact grid line) fails BOTH tests and its
        # bond is silently dropped from neither direction, undercounting
        # the force. Using one threshold makes a node exactly at y0
        # consistently belong to the "col/at-or-above" side only, so
        # every bond is still counted exactly once (no double-count) but
        # none are lost.
        # single threshold a quarter-cell below the plane: counts
        # every crossing bond exactly once, robust in float32
        thr = y0 - 0.25 * dx
        crosses = (x_d[row, 1] < thr) & (x_d[col, 1] >= thr)
        return weighted[crosses].sum(dim=0)


def save_snapshots(x, u, damage, Nx, Ny, step, t_sim, out_dir):
    """Per-snapshot plots: displacement components u_x, u_y and the
    damage field, each as a structured-grid image (x-major node order)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xc = x[:, 0].detach().cpu().numpy()
    yc = x[:, 1].detach().cpu().numpy()
    extent = [xc.min(), xc.max(), yc.min(), yc.max()]
    fields = [
        (u[:, 0].detach().cpu().numpy(), "coolwarm", "$u_x$", "disp_x"),
        (u[:, 1].detach().cpu().numpy(), "coolwarm", "$u_y$", "disp_y"),
        (damage.detach().cpu().numpy(), "hot_r", "damage $\\phi$", "damage"),
    ]
    for vals, cmap, label, tag in fields:
        img = vals.reshape(Nx, Ny).T
        if tag == "damage":
            vmin, vmax = 0.0, 0.6
        else:
            m = max(abs(float(vals.min())), abs(float(vals.max())), 1e-12)
            vmin, vmax = -m, m
        fig, ax = plt.subplots(figsize=(6, 9))
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect="equal")
        cb = fig.colorbar(im, ax=ax, shrink=0.75)
        cb.set_label(label)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(f"{label},  step {step},  $t={t_sim:g}$")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"{tag}_{step:06d}.png", dpi=200)
        plt.close(fig)



def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    print(f"Device: {device}  Precision: {dtype}")

    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(model_dir)
    model = build_model(summary, model_dir, device).to(dtype)
    if args.symmetrize_kernel:
        model.k = SymmetrizedK(model.k)
    print(f"Model: {summary['run_tag']}  alpha={model.get_alpha().item():.4f}  "
          f"kernel={'symmetrized' if args.symmetrize_kernel else 'raw'}")

    (unit_s, unit_sf, unit_fac, delta_train, delta_md,
     unit_source) = resolve_training_units(summary, args.unit_s,
                                           args.unit_sigma_f)
    # UNITS: this driver (like its parents) works in GAMMA-SCALED
    # length units -- 1 code length unit = gamma lu (dx = 5 means
    # gamma*5 lu).  The 1/gamma^3 operator amplitude and the invG
    # critical-stretch remap make the learned operator consistent in
    # these rescaled units, so delta is NOT multiplied by gamma here.
    # The quadrature stencil is fixed at the training ratio
    # m = delta/dx = 3.01: the horizon FOLLOWS the grid spacing,
    # delta = 3.01 * dx  (default dx = 5 -> delta = delta_MD = 15.05).
    dx = args.dx if args.dx is not None else delta_md / 3.01
    delta = 3.01 * dx
    rescale_factor = args.rescale * unit_fac
    print(f"Units: (s,sigma_f)=({unit_s:g},{unit_sf:g}) [{unit_source}]  "
          f"unit_fac={unit_fac:g}  delta={delta:g}")

    lam_peak, _ = find_learned_g_peak(model, device=device)
    lam_cut = lam_peak if args.lam_cut is None else float(args.lam_cut)
    g_fn, _ = build_learned_cutoff_g(model, lam_cut=lam_cut,
                                     decay=args.decay, device=device)
    s0 = (lam_peak - 1.0) if args.s0 is None else float(args.s0)
    # gamma-rescaling law of g: the critical stretch is remapped by
    # invG, s_tilde_0 = invG(gamma, s0) (< s0 for gamma > 1), and the
    # SOFTENING CUT of g is relocated to the SAME remapped threshold:
    # g follows the learned curve up to lam = 1 + s_tilde_0 and decays
    # along the C1-matched exponential tail beyond it, so bonds soften
    # and break at the same rescaled stretch (at gamma = 1 this reduces
    # to the usual cut at the learned peak).
    s_tilde_0 = invG(args.gamma, g_fn, model, s0, args.bondsoft_decay,
                     device=device)
    if np.isnan(s_tilde_0):
        s_tilde_0 = s0
    if abs(s_tilde_0 - s0) > 1e-12:
        g_fn, _ = build_learned_cutoff_g(model,
                                         lam_cut=1.0 + s_tilde_0,
                                         decay=args.decay, device=device)
    print(f"g peak {lam_peak:.4f}  s0={s0:.4f}  "
          f"s_tilde_0=invG(gamma={args.gamma:g})={s_tilde_0:.6f}  "
          f"(g softening cut AND break threshold at s_tilde_0)")

    # ── domain: [0, Lx] x [0, Ly], edge crack at y = Ly/2 ───────────────
    # STAGGERED: Ny must be EVEN so that the crack plane
    # notch_y = Ly/2 = (Ny-1)/2 * dx is a half-integer multiple of dx
    # and runs BETWEEN two layers of grid points (no node on the plane).
    Nx, Ny = args.Nx, args.Ny
    if Ny % 2 == 1:
        Ny += 1
        print(f"Ny bumped {args.Ny} -> {Ny} (even) so the crack plane "
              f"y = Ly/2 lies between two grid layers")
    N = Nx * Ny
    x = create_rectangular_grid(Nx, Ny, dx)
    Lx, Ly = (Nx - 1) * dx, (Ny - 1) * dx
    edge_index, _, _ = build_rectangular_graph(Nx, Ny, dx, delta)
    notch_y = Ly / 2.0
    notch_x_end = args.notch_frac * Lx
    bond_alive = initialize_prenotch(x, edge_index, -dx / 4.0, notch_x_end,
                                     notch_y, dx).to(device)
    bond_strength = bond_alive.clone()
    print(f"Domain [0,{Lx:g}]x[0,{Ly:g}]  N={N}  edges={edge_index.shape[1]}")

    # ── boundary protection (opt-in), unchanged from the parent ──────
    breakable = None
    if args.protect_boundary:
        pw = args.protect_width if args.protect_width is not None else delta
        bnd_node = ((x[:, 0] <= pw)              # left
                    | (x[:, 1] <= pw)            # bottom
                    | (x[:, 1] >= Ly - pw))      # top  (right: unprotected)
        touches = bnd_node[edge_index[0]] | bnd_node[edge_index[1]]
        breakable = (~touches).to(device)
        print(f"Boundary protection: ON  width={pw:g} "
              f"(left/top/bottom; right unprotected)  "
              f"{int(touches.sum())} edges locked, "
              f"{int(breakable.sum())} breakable")

    # ── body-force loading on the delta-thick top/bottom skins ───────
    # b = (0, +-traction / ((Nx-1)dx * delta * H)): the prescribed
    # total force +-traction spread uniformly over the skin volume
    # width * delta * H.  All surfaces are do-nothing (no ghosts, no
    # Dirichlet); the +-b pair carries zero net force and rigid modes
    # are controlled by inertia.
    # The slab thickness is a PHYSICAL constant, 0.935 lu.  The length
    # unit of a run is set by dx: the code horizon delta = 3.01*dx
    # represents the physical horizon gamma*delta_MD, so
    #     1 code length unit = gamma*delta_MD/(3.01*dx)  [lu]
    # (dx=5, gamma=10 -> unit = 10 lu, H_code = 0.0935;
    #  dx=50, gamma=10 -> unit = 1 lu,  H_code = 0.935).
    H_SLAB_LU = 0.935                    # physical thickness [lu]
    unit_lu = args.gamma * delta_md / (3.01 * dx)
    H_code = H_SLAB_LU / unit_lu         # thickness in code length units
    top_skin = x[:, 1] >= (Ly - delta)
    bot_skin = x[:, 1] <= delta
    b_mag = args.traction / ((Nx - 1) * dx * delta * H_code)
    b_ext = torch.zeros(N, 2)
    b_ext[top_skin, 1] = +b_mag
    b_ext[bot_skin, 1] = -b_mag
    b_ext = b_ext.to(device)
    print(f"Body load: b_y = +-{args.traction:g}/((Nx-1)dx*delta*H_code) "
          f"= +-{b_mag:.6g} on {int(top_skin.sum())}/"
          f"{int(bot_skin.sum())} top/bottom skin nodes "
          f"(delta-thick, do-nothing surfaces everywhere); "
          f"H = {H_SLAB_LU:g} lu physical = {H_code:g} code units "
          f"(length unit = {unit_lu:g} lu at gamma={args.gamma:g}, "
          f"dx={dx:g})")

    x = x.to(device)
    edge_index = edge_index.to(device)
    u = torch.zeros(N, 2, device=device)
    v = torch.zeros(N, 2, device=device)

    beta_nm, gamma_nm, rho, dt = (args.newmark_beta, args.newmark_gamma,
                                  args.rho, args.dt)
    mass_coef = rho / (dt * dt * beta_nm)

    # initial acceleration at t = 0 (u = 0 -> f_int = 0, a = b/rho)
    with torch.no_grad():
        f0 = internal_force(model, u, x, edge_index, bond_strength,
                            delta, dx, device, g_fn=g_fn,
                            gamma_scale=args.gamma,
                            rescale_factor=rescale_factor)
    a = (f0 + b_ext) / rho

    hist = {"time": [], "ke": [], "n_broken": [], "max_damage": [],
            "max_disp": []}
    total_broken = int((bond_alive < 0.5).sum().item())

    for step in range(1, args.n_steps + 1):
        t_sim = step * dt
        u_tilde = u + dt * v + dt * dt * (0.5 - beta_nm) * a
        v_tilde = v + dt * (1.0 - gamma_nm) * a

        n_new_step = 0
        for stab in range(args.max_stab_passes):
            def residual(uu):
                f = internal_force(
                    model, uu, x, edge_index, bond_strength,
                    delta, dx, device, g_fn=g_fn, gamma_scale=args.gamma,
                    rescale_factor=rescale_factor)
                return mass_coef * (uu - u_tilde) - f - b_ext

            u, hist_n, conv = _robust_newton_solve_residual(
                u.detach(), residual, tol=args.newton_tol,
                max_iter=args.newton_max_iter, cg_tol=args.cg_tol,
                cg_max_iter=args.cg_max_iter,
                verbose=args.verbose_newton,
                tag=f"-s{step}b{stab}", free_mask=None)

            with torch.no_grad():
                f_int, stretch = internal_force(
                    model, u, x, edge_index, bond_strength,
                    delta, dx, device, g_fn=g_fn, gamma_scale=args.gamma,
                    rescale_factor=rescale_factor,
                    return_edge_stretch=True)
            n_new = update_bonds(stretch, bond_alive, s_tilde_0, breakable,
                                 bond_strength=bond_strength,
                                 bondsoft_decay=args.bondsoft_decay,
                                 edge_index=edge_index)
            n_new_step += n_new
            if n_new == 0:
                break
        total_broken += n_new_step

        # Newmark correctors
        a = (f_int + b_ext) / rho
        v = v_tilde + dt * gamma_nm * a

        ke = 0.5 * rho * float((v ** 2).sum())
        damage = compute_damage(edge_index, bond_alive, N)
        hist["time"].append(t_sim)
        hist["ke"].append(ke)
        hist["n_broken"].append(total_broken)
        hist["max_damage"].append(float(damage.max()))
        hist["max_disp"].append(float(torch.norm(u, dim=1).max()))
        if step % args.print_interval == 0 or n_new_step > 0:
            print(f"step {step:5d}  t={t_sim:.4f}  KE={ke:.3e}  "
                  f"|u|max={hist['max_disp'][-1]:.3e}  "
                  f"broken={total_broken} (+{n_new_step})  conv={conv}")

        if step % args.snapshot_interval == 0 or step == args.n_steps:
            np.savez(out_dir / f"state_{step:06d}.npz",
                     x=x.cpu().numpy(), u=u.detach().cpu().numpy(),
                     v=v.detach().cpu().numpy(),
                     damage=damage.cpu().numpy(),
                     bond_alive=bond_alive.cpu().numpy(),
                     t=t_sim, step=step)
            save_snapshots(x, u, damage, Nx, Ny, step, t_sim, out_dir)

        if ke > 1e10 or np.isnan(ke):
            print(f"*** UNSTABLE at step {step}, KE={ke:.3e}")
            break

    np.savez(out_dir / "history.npz",
             **{k: np.array(v) for k, v in hist.items()})
    config = vars(args).copy()
    config.update(delta=delta, N=N, Ny_effective=Ny, Lx=Lx, Ly=Ly,
                  b_mag=b_mag, total_broken=total_broken,
                  unit_fac=unit_fac, s_tilde_0=s_tilde_0)
    json.dump(config, open(out_dir / "config.json", "w"), indent=2,
              default=str)
    print(f"saved outputs to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=str, default="../models/nrm_n256_lr3e4")
    p.add_argument("--Nx", type=int, default=101)
    p.add_argument("--Ny", type=int, default=41)
    p.add_argument("--dx", type=float, default=None,
                   help="grid spacing (gamma-scaled units; default "
                        "delta_MD/3.01 = 5).  The horizon follows the "
                        "training stencil: delta = 3.01*dx.")
    p.add_argument("--protect-boundary", action="store_true",
                   help="lock bonds touching nodes within --protect-width "
                        "of the left/top/bottom boundaries: no NEW damage "
                        "(breaking or softening) can occur there.  The "
                        "pre-notch is unaffected, and the right boundary "
                        "stays unprotected so the crack can run out "
                        "through it.")
    p.add_argument("--protect-width", type=float, default=None,
                   help="thickness of the protected boundary strips "
                        "(default: one horizon delta).")
    p.add_argument("--traction", type=float, default=0.15,
                   help="prescribed total load: a uniform body-force "
                        "density b = (0, +-traction/((Nx-1)dx*delta*H)) "
                        "[fu/lu^3, H=0.935] is applied on the delta-thick "
                        "top (+) and bottom (-) skins.")
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.001)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--notch-frac", type=float, default=0.5)
    p.add_argument("--s0", type=float, default=None)
    p.add_argument("--lam-cut", type=float, default=None)
    p.add_argument("--decay", type=float, default=10.0)
    p.add_argument("--bondsoft-decay", type=float, default=200.0,
                   help="softening sharpness of broken bonds past s0 "
                        "(bondsoft ~ exp(-decay*(s-s0))); default 200 = "
                        "the gradual softening of "
                        "fracture_sim_implicit_norm.py (10000 = the "
                        "near-hard break of the traction driver).")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--rescale", type=float, default=1.0)
    p.add_argument("--unit-s", type=float, default=None)
    p.add_argument("--unit-sigma-f", type=float, default=None)
    p.add_argument("--symmetrize-kernel", dest="symmetrize_kernel",
                   action="store_true", default=True)
    p.add_argument("--raw-kernel", dest="symmetrize_kernel",
                   action="store_false")
    p.add_argument("--newmark-gamma", type=float, default=0.5)
    p.add_argument("--newmark-beta", type=float, default=0.25)
    p.add_argument("--max-stab-passes", type=int, default=10)
    p.add_argument("--newton-tol", type=float, default=1e-4)
    p.add_argument("--newton-max-iter", type=int, default=30)
    p.add_argument("--cg-tol", type=float, default=1e-2)
    p.add_argument("--cg-max-iter", type=int, default=60)
    p.add_argument("--verbose-newton", action="store_true")
    p.add_argument("--print-interval", type=int, default=10)
    p.add_argument("--snapshot-interval", type=int, default=100)
    p.add_argument("--precision", type=str, default="float32",
                   choices=["float32", "float64"])
    p.add_argument("--out-dir", type=str, default="fracture_results_bodyload")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
