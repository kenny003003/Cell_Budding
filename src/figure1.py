"""
Reproduce Figure 1 of Senthilkumar et al. (PNAS 2026):
"Influence of confinement on single-cell growth".

Panels:
  (b) cell-volume growth curve, free vs confined  -> Fig 1b
  (c) intracellular hydrostatic pressure           -> Fig 1c
  (d) osmotic-pressure contributions               -> Fig 1d
  (e) competition between hydrostatic & osmotic     -> Fig 1e
  (f) mitotic volume checkpoint                     -> Fig 1f

Outputs a combined figure and individual panels to ../figures/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parameters import default_params
import single_cell_model as scm

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

HOUR = 3600.0
FREE_C = "#1f77b4"     # blue  - freely growing cell
CONF_C = "#d62728"     # red   - confined cell


def run():
    p = default_params()
    free = scm.simulate(p, confined=False)
    conf = scm.simulate(p, confined=True)
    return p, free, conf


def _hours(d):
    return d["t"] / HOUR


# --------------------------------------------------------------------------
def panel_volume(ax, p, free, conf):
    ax.plot(_hours(free), free["V"], color=FREE_C, lw=2.5, label="free (unconfined)")
    ax.plot(_hours(conf), conf["V"], color=CONF_C, lw=2.5, label="confined")
    ax.axhline(p.V_div, ls="--", color="0.4", lw=1.2)
    ax.text(12, p.V_div, "mitotic volume checkpoint $V_{div}$",
            va="bottom", ha="center", color="0.3", fontsize=8)
    ax.set_xlabel("time (h)")
    ax.set_ylabel(r"cell volume $V$ ($\mu$m$^3$)")
    ax.set_title("(b) cell-cycle volume growth")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.set_xlim(0, 24)
    ann = (f"free: $\\times${free['V'][-1]/free['V'][0]:.2f}\n"
           f"confined: $\\times${conf['V'][-1]/conf['V'][0]:.2f}")
    ax.text(0.98, 0.05, ann, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))


def panel_hydrostatic(ax, p, free, conf):
    ax.plot(_hours(free), free["dP"], color=FREE_C, lw=2.5, label="free")
    ax.plot(_hours(conf), conf["dP"], color=CONF_C, lw=2.5, label="confined")
    ax.set_xlabel("time (h)")
    ax.set_ylabel(r"hydrostatic pressure $\Delta P$ (Pa)")
    ax.set_title("(c) intracellular hydrostatic pressure")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(0, 24)
    ax.annotate(f"{free['dP'][0]:.0f} Pa", (0, free["dP"][0]),
                textcoords="offset points", xytext=(6, -10), fontsize=8, color=FREE_C)
    ax.annotate(f"{free['dP'][-1]:.0f} Pa", (24, free["dP"][-1]),
                textcoords="offset points", xytext=(-38, 4), fontsize=8, color=FREE_C)


def panel_osmotic(ax, p, free, conf):
    ax.plot(_hours(free), free["Pi_n"], color=FREE_C, lw=2.5,
            label=r"free: $\Pi_n$ (biomolecules)")
    ax.plot(_hours(conf), conf["Pi_n"], color=CONF_C, lw=2.5,
            label=r"confined: $\Pi_n$")
    ax.plot(_hours(free), free["dPi_ip"], color=FREE_C, lw=1.5, ls=":",
            label=r"$\Delta\Pi_{i,p}$ (perm. ions)")
    ax.set_xlabel("time (h)")
    ax.set_ylabel(r"osmotic pressure (Pa)")
    ax.set_title("(d) osmotic-pressure contributions")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlim(0, 24)
    ax.text(0.98, 0.05,
            "biomolecule synthesis ($\\Pi_n$) dominates;\n"
            "confinement -> water efflux -> higher $\\Pi_n$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))


def panel_competition(ax, p, free, conf):
    # competition between hydrostatic pressure (load) and osmotic pressure
    # (synthesis) that governs volume growth.
    ax.plot(_hours(conf), conf["sigma_g"], color="0.5", lw=2.0,
            label=r"contact stress $\sigma_g$ (confined)")
    ax.plot(_hours(conf), conf["dP"], color=CONF_C, lw=2.5,
            label=r"hydrostatic $\Delta P$ (confined)")
    ax.plot(_hours(conf), conf["Pi_n"], color=CONF_C, lw=2.0, ls="--",
            label=r"osmotic $\Pi_n$ (confined)")
    ax.plot(_hours(free), free["dP"], color=FREE_C, lw=2.5,
            label=r"hydrostatic $\Delta P$ (free)")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("pressure / stress (Pa)")
    ax.set_title("(e) hydrostatic-osmotic competition")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_xlim(0, 24)


def panel_checkpoint(ax, p, free, conf):
    V = np.linspace(0.6 * free["V"][0], 1.25 * p.V_div, 400)
    ax.plot(V, scm.p_division(V, p), color="0.3", lw=2.0)
    ax.axvline(p.V_div, ls="--", color="0.4", lw=1.2)
    ax.text(p.V_div, 0.5, "  $V_{div}$", color="0.3", fontsize=9, va="center")
    # endpoints of free / confined cells
    for d, c, lab in ((free, FREE_C, "free"), (conf, CONF_C, "confined")):
        Vend = d["V"][-1]
        ax.scatter([Vend], [scm.p_division(Vend, p)], color=c, s=60, zorder=5,
                   label=f"{lab} (end of cycle)")
        ax.annotate(f"{Vend:.0f}", (Vend, scm.p_division(Vend, p)),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8, color=c)
    ax.set_xlabel(r"cell volume $V$ ($\mu$m$^3$)")
    ax.set_ylabel(r"division probability $P_{div}$")
    ax.set_title("(f) mitotic sizing checkpoint")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.set_ylim(-0.05, 1.05)


def make_combined(p, free, conf):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    panel_volume(axs[0, 0], p, free, conf)
    panel_hydrostatic(axs[0, 1], p, free, conf)
    panel_osmotic(axs[0, 2], p, free, conf)
    panel_competition(axs[1, 0], p, free, conf)
    panel_checkpoint(axs[1, 1], p, free, conf)
    # summary text panel
    ax = axs[1, 2]
    ax.axis("off")
    txt = (
        "Replication of Fig. 1\n"
        "Senthilkumar et al., PNAS 2026\n"
        "(single-cell mechano-osmotic model)\n\n"
        f"Free cell:\n"
        f"   V: {free['V'][0]:.0f} -> {free['V'][-1]:.0f} um^3 "
        f"(x{free['V'][-1]/free['V'][0]:.2f})\n"
        f"   dP: {free['dP'][0]:.0f} -> {free['dP'][-1]:.0f} Pa\n"
        f"   -> reaches checkpoint, divides\n\n"
        f"Confined cell (sigma_g -> {p.sigma_g_peak:.0f} Pa):\n"
        f"   V: {conf['V'][0]:.0f} -> {conf['V'][-1]:.0f} um^3 "
        f"(x{conf['V'][-1]/conf['V'][0]:.2f})\n"
        f"   dP: {conf['dP'][0]:.0f} -> {conf['dP'][-1]:.0f} Pa\n"
        f"   -> arrested below checkpoint\n\n"
        "Paper targets:\n"
        "   ~2-fold volume increase (free)\n"
        "   dP ~210 -> ~340 Pa (free)\n"
        "   confinement prevents division"
    )
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace")

    fig.suptitle("Stress-dependent single-cell growth from a mechano-osmotic "
                 "coupling and volume checkpoint", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIGDIR, "figure1_single_cell.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)

    # individual panels
    for name, fn in (("b_volume", panel_volume), ("c_hydrostatic", panel_hydrostatic),
                     ("d_osmotic", panel_osmotic), ("e_competition", panel_competition),
                     ("f_checkpoint", panel_checkpoint)):
        f, a = plt.subplots(figsize=(5, 4))
        fn(a, p, free, conf)
        f.tight_layout()
        op = os.path.join(FIGDIR, f"fig1{name}.png")
        f.savefig(op, dpi=150)
        plt.close(f)
    print("wrote individual panels to", FIGDIR)


if __name__ == "__main__":
    p, free, conf = run()
    make_combined(p, free, conf)
