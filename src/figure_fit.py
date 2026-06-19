"""
Quantitative fit of the stress-dependent-growth result: final spheroid
fold-growth vs ECM stiffness (simplified open model, spheroid_model.py).

Sweeps ECM stiffness over a fine grid with several random seeds, records the
final spheroid volume fold-growth (mean +/- s.d.), and fits the mechanistic form
implied by the confinement law -- growth arrests when the ECM stress reaches the
checkpoint threshold, which gives a hyperbola with a *critical stiffness* E_c
below which growth is unbounded:

    fold(E) = 1 + C / (E - E_c)

Points that hit the cell-count cap (very soft matrix, near/below E_c) are
excluded from the fit and drawn as open markers. Reports C, E_c and R^2.

Run:  python figure_fit.py   ->  writes ../figures/figure_fit.png
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import dataclasses
from spheroid_model import SpheroidParams, Spheroid, VolumeTable, ECM_STIFFNESSES_KPA

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def model(E, C, Ec):
    return 1.0 + C / (E - Ec)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    base = SpheroidParams()
    table = VolumeTable(base.cell)

    E_kpa = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.25, 1.45])
    seeds = [0, 1, 2, 3]

    mean, sd, capped = [], [], []
    for E in E_kpa:
        folds, hit_cap = [], False
        for sd_i in seeds:
            sp = dataclasses.replace(base, seed=sd_i)
            s = Spheroid(E_ecm=E * 1000.0, sp=sp, table=table)
            s.run()
            ser = s.series()
            folds.append(ser["V_total"][-1] / s.V0_total)
            hit_cap |= ser["N"][-1] >= sp.n_max
        mean.append(np.mean(folds)); sd.append(np.std(folds)); capped.append(hit_cap)
    mean = np.array(mean); sd = np.array(sd); capped = np.array(capped)

    # fit only the points that plateaued below the cap
    fitmask = ~capped
    (C, Ec), cov = curve_fit(model, E_kpa[fitmask], mean[fitmask],
                             p0=(3.0, 0.35), sigma=sd[fitmask] + 1e-6,
                             absolute_sigma=False, maxfev=10000)
    resid = mean[fitmask] - model(E_kpa[fitmask], C, Ec)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((mean[fitmask] - mean[fitmask].mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot

    # figure
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    Ef = np.linspace(max(E_kpa.min(), Ec + 0.02), E_kpa.max(), 300)
    ax.plot(Ef, model(Ef, C, Ec), "-", color="#a50f15", lw=2,
            label=f"fit  $1+C/(E-E_c)$\n$C$={C:.2f},  $E_c$={Ec:.2f} kPa,  $R^2$={r2:.3f}")
    ax.axvline(Ec, color="#a50f15", ls="--", lw=1,
               label=f"critical stiffness $E_c$={Ec:.2f} kPa")
    ax.errorbar(E_kpa[fitmask], mean[fitmask], yerr=sd[fitmask], fmt="o",
                color="#2c7fb8", ms=7, capsize=3, label="simulation (plateaued)")
    if capped.any():
        ax.errorbar(E_kpa[capped], mean[capped], yerr=sd[capped], fmt="o",
                    mfc="white", mec="#2c7fb8", color="#2c7fb8", ms=7, capsize=3,
                    label="cell-count cap (excluded)")
    # the three ECM stiffnesses used by the paper's AI surrogate
    for k in ECM_STIFFNESSES_KPA:
        ax.axvline(k, color="grey", ls=":", lw=0.8)
    ax.text(0.98, 0.95, "dotted: 0.58 / 0.85 / 1.1 kPa\n(paper's ECM stiffnesses)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="grey")

    ax.set_xlabel("ECM stiffness  $E_{ECM}$  [kPa]")
    ax.set_ylabel("final spheroid volume fold-growth")
    ax.set_title("Stress-dependent spheroid growth: fit of the model result")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "figure_fit.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print("\nfold-growth vs ECM stiffness:")
    for E, m, s_, c in zip(E_kpa, mean, sd, capped):
        print(f"  E={E:.2f} kPa:  fold = {m:5.1f} +/- {s_:4.1f}  {'(capped)' if c else ''}")
    print(f"\nfit:  fold(E) = 1 + {C:.3f} / (E - {Ec:.3f})   R^2 = {r2:.4f}")
    print(f"critical stiffness E_c = {Ec:.3f} kPa (growth unbounded below this)")
    return C, Ec, r2


if __name__ == "__main__":
    main()
