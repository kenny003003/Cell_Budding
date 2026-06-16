"""
Figure 4 (simplified open reproduction): stress-/stiffness-dependent spheroid
growth emerging from the single-cell mechano-osmotic sizing checkpoint.

Panels
  (a) tumour size N(t) for an unconfined spheroid and for three ECM stiffnesses
      (0.58 / 0.85 / 1.1 kPa) -- growth is increasingly suppressed by stiffer
      matrix (cf. paper Fig. 4 stress-dependent growth).
  (b) final spheroid fold-growth vs ECM stiffness -- the key monotonic trend.
  (c) confinement stress that the cells equilibrate at, vs ECM stiffness, with
      the single-cell arrest threshold marked (growth stalls once cells are held
      below the mitotic volume V_div).
  (d) mid-plane snapshot of the 1.1 kPa spheroid, cells coloured by compressive
      stress sigma_g -- core cells are more compressed than rim cells (the
      paper's spatially varying cell volume).

Run:  python figure4.py    ->  writes ../figures/figure4_spheroid*.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import single_cell_model as scm
from parameters import default_params
from spheroid_model import (SpheroidParams, run_conditions, ECM_STIFFNESSES_KPA)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
COLORS = {"unconfined": "#444444", "0.58 kPa": "#2c7fb8",
          "0.85 kPa": "#d95f0e", "1.1 kPa": "#a50f15"}


def arrest_threshold_stress(p):
    """sigma_g at which a fully-synthesised cell (n_n=n_sat) sits exactly at the
    mitotic volume V_div -- above this, growth cannot pass the checkpoint."""
    from scipy.optimize import brentq
    f = lambda sg: scm.solve_state(p.n_sat, sg, p)["V"] - p.V_div
    return brentq(f, 0.0, 1000.0)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    sp = SpheroidParams()
    res = run_conditions(sp, include_free=True)
    p = sp.cell
    sig_star = arrest_threshold_stress(p)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.6))
    fig.suptitle("Stress-dependent spheroid growth from a mechano-osmotic sizing "
                 "checkpoint\n(simplified open reproduction of Senthilkumar et al., "
                 "PNAS 2026, Fig. 4)", fontsize=12, fontweight="bold")

    # (a) growth curves -----------------------------------------------------
    a = ax[0, 0]
    for label, s in res.items():
        ser = s.series()
        a.plot(ser["t"] / 24.0, ser["N"], color=COLORS[label], lw=2, label=label)
    a.set_xlabel("time [days]")
    a.set_ylabel("number of cells $N$")
    a.set_title("(a) tumour growth vs ECM stiffness")
    a.legend(frameon=False, fontsize=9)
    a.spines[["top", "right"]].set_visible(False)

    # (b) final fold-growth vs stiffness -----------------------------------
    b = ax[0, 1]
    kpa = list(ECM_STIFFNESSES_KPA)
    fold = [res[f"{k:g} kPa"].series()["V_total"][-1] / res[f"{k:g} kPa"].V0_total
            for k in kpa]
    free_fold = res["unconfined"].series()["V_total"][-1] / res["unconfined"].V0_total
    b.axhline(free_fold, color=COLORS["unconfined"], ls="--", lw=1.5,
              label=f"unconfined (×{free_fold:.0f})")
    b.plot(kpa, fold, "o-", color="#a50f15", lw=2, ms=9)
    for k, f in zip(kpa, fold):
        b.annotate(f"×{f:.1f}", (k, f), textcoords="offset points",
                   xytext=(6, 8), fontsize=9)
    b.set_xlabel("ECM stiffness $E_{ECM}$ [kPa]")
    b.set_ylabel("final spheroid volume fold-growth")
    b.set_title("(b) stiffer matrix → less growth")
    b.legend(frameon=False, fontsize=9)
    b.spines[["top", "right"]].set_visible(False)

    # (c) radial profile of cell volume (spatial variation) ----------------
    c = ax[1, 0]
    c.axhline(1.0, color="grey", ls=":", lw=1.5, label="mitotic checkpoint $V_{div}$")
    for label in ("0.58 kPa", "1.1 kPa"):
        s = res[label]
        rho = np.linalg.norm(s.pos - s.pos.mean(axis=0), axis=1)
        rho_n = rho / (rho.max() + 1e-9)
        c.scatter(rho_n, s.V / p.V_div, s=18, alpha=0.6,
                  color=COLORS[label], label=label)
    c.set_xlabel("normalised radial position  $r/R$  (0 = core)")
    c.set_ylabel("cell volume  $V / V_{div}$")
    c.set_title("(c) confined core cells held below the checkpoint")
    c.legend(frameon=False, fontsize=9)
    c.spines[["top", "right"]].set_visible(False)

    # (d) snapshot of the stiffest spheroid, true-to-scale cells -----------
    from matplotlib.collections import EllipseCollection
    d = ax[1, 1]
    s = res["1.1 kPa"]
    pos, sg, r = s.pos, s.sigma_g, s.r
    order = np.argsort(pos[:, 2])                    # back-to-front
    ec = EllipseCollection(widths=2 * r[order], heights=2 * r[order], angles=0,
                           units="xy", offsets=pos[order, :2],
                           transOffset=d.transData, cmap="inferno",
                           edgecolors="k", linewidths=0.3)
    ec.set_array(sg[order])
    d.add_collection(ec)
    L = 1.05 * (np.linalg.norm(pos[:, :2], axis=1) + r).max()
    d.set_xlim(-L, L); d.set_ylim(-L, L)
    d.set_aspect("equal")
    d.set_xlabel("x [µm]"); d.set_ylabel("y [µm]")
    d.set_title("(d) 1.1 kPa spheroid: core more compressed")
    cb = fig.colorbar(ec, ax=d, shrink=0.85)
    cb.set_label(r"$\sigma_g$ [Pa]")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIGDIR, "figure4_spheroid.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # text summary
    print("\nFinal state:")
    for label, s in res.items():
        ser = s.series()
        print(f"  {label:>10}: N={ser['N'][-1]:4d}  fold={ser['V_total'][-1]/s.V0_total:5.1f}"
              f"  arrested={100*ser['arrested_frac'][-1]:5.0f}%  "
              f"<sigma_g>={float(s.sigma_g.mean()):5.1f} Pa")
    return res


if __name__ == "__main__":
    main()
