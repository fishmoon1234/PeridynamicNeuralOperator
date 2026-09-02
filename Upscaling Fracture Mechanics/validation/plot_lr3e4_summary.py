#!/usr/bin/env python3
"""Post-processing and comparison of the validation results against MD.

Reads the bundled result files (all in this folder):
  MNO_force_nofracture_L50.txt         no-fracture pull test, learned g
  MNO_force_nofracture_linearg_L50.txt no-fracture pull test, linearized g
  MNO_force_fracture_L50.txt           pre-notched, 100x200, delta=15.05
  MNO_force_fracture_L100.txt          pre-notched, 200x400, delta=15.05
  MNO_force_fracture_L100_delta30.txt  pre-notched, 200x400, delta=30.1
  MD_nofracture_totalforce.dat         MD reference, no fracture
  MD_L50_totalforce.dat                MD reference, 100x200
  MD_L100_totalforce.dat               MD reference, 200x400
MNO files are driver force.txt outputs (col 0: t, col 5: force across
y=0, col 10 where present: elasticity estimate); MD files carry (t,
force) with the applied grip displacement = 0.025*t.

Outputs:
  lr3e4_nofracture_force.png
  lr3e4_fracture_L_comparison.png

Usage:  python plot_lr3e4_summary.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ld = lambda name: np.loadtxt(os.path.join(HERE, name))

# ── 1. no-fracture pull test ─────────────────────────────────────────
d = ld("MNO_force_nofracture_L50.txt")
dl = ld("MNO_force_nofracture_linearg_L50.txt")
md = ld("MD_nofracture_totalforce.dat")

fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
V50 = 0.025                       # pull rate of the L=50 runs
ax.plot(V50 * dl[:, 0], dl[:, 5], "-", color="C0", lw=2.0,
        label="MNO: linearized $g$")
ax.plot(V50 * dl[:, 0], dl[:, 10], ":", color="k", lw=2.0,
        label=r"elastic $E\,\epsilon_{yy}\,(2L\,H)$")
ax.plot(V50 * d[:, 0], d[:, 5], "-", color="C1", lw=2.0,
        label="MNO: learned $g$")
ax.plot(V50 * md[:, 0], md[:, 1], "--", color="0.35", lw=2.3,
        label="MD")
ax.set_title("nrm_lr3e4, no-fracture pull test ($100x200$)", fontsize=11)
ax.set_xlabel("Applied displacement", fontsize=20)
ax.set_ylabel(r"Total force across $y=0$", fontsize=20)
ax.grid(alpha=0.3)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=12, loc="upper left")
out1 = os.path.join(HERE, "lr3e4_nofracture_force.png")
fig.savefig(out1, dpi=200)
print("saved", out1)

# ── 2. fracture: domain size and horizon vs. MD ──────────────────────
MNO = [("MNO_force_fracture_L50.txt", 0.025,
        r"$MNO: 100x200,\ \delta=15.05$", "C0", "-",  (-8, 6), "right"),
       ("MNO_force_fracture_L100.txt", 0.05,
        r"$MNO: 200x400,\ \delta=15.05$", "C2", "-",  (2, 6), "left"),
       ("MNO_force_fracture_L100_delta30.txt", 0.05,
        r"$MNO:200x400,\ \delta=30.1$", "C1", "--", (-5, 4), "right")]
MD = [("MD_L50_totalforce.dat",  0.025, "--", 2.3,
       "MD $100x200$", (10, -16)),
      ("MD_L100_totalforce.dat", 0.05, ":", 2.6,
       "MD $200x400$", (10, -16))]

fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
for name, v, lbl, color, ls, off, ha in MNO:
    dd = ld(name)
    tt, FF = v * dd[:, 0], dd[:, 5]
    ax.plot(tt, FF, ls, color=color, lw=1.9, label=lbl)
    i = FF.argmax()
    ax.plot(tt[i], FF[i], "o", color=color, ms=5, zorder=5)
    ax.annotate(f"{FF[i]:.0f}", (tt[i], FF[i]), xytext=off,
                textcoords="offset points", ha=ha, color=color,
                fontsize=20, fontweight="bold")
for name, v, ls, lw, lbl, off in MD:
    mm = ld(name)
    t_m, F_m = v * mm[:, 0], mm[:, 1]
    ax.plot(t_m, F_m, ls, color="0.35", lw=lw, label=lbl)
    i = F_m.argmax()
    ax.plot(t_m[i], F_m[i], "s", color="0.35", ms=5, zorder=5)
    ax.annotate(f"{F_m[i]:.0f}", (t_m[i], F_m[i]), xytext=off,
                textcoords="offset points", color="0.35",
                fontsize=20, fontweight="bold")
ax.set_title("nrm_lr3e4, pre-notched fracture test: domain size and "
             "horizon", fontsize=11)
ax.set_xlabel("Applied displacement", fontsize=20)
ax.set_ylabel(r"Total force across $y=0$", fontsize=20)
ax.set_xlim(0, 9)
ax.set_ylim(-10, 1100)
ax.grid(alpha=0.3)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=12, loc="upper right")
out2 = os.path.join(HERE, "lr3e4_fracture_L_comparison.png")
fig.savefig(out2, dpi=200)
print("saved", out2)
