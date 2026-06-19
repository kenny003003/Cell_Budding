"""
Animate the simplified open spheroid model: unconfined vs ECM-confined tumour
growth, cells coloured by their mechano-osmotic compressive stress sigma_g.

Cells are drawn as TRUE-TO-SCALE discs (radius = real cell radius in micrometres,
on a shared axis scale across panels and frames), so contact, packing and the
relative spheroid sizes are shown faithfully -- the unconfined spheroid is a
large, dense, touching cluster; stiffer matrices stay small.

Run:  python animate_spheroid.py   ->  writes ../figures/spheroid_growth.gif
(uses matplotlib's Pillow writer; no ffmpeg required)
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import EllipseCollection
from matplotlib import cm, colors

from spheroid_model import SpheroidParams, run_conditions

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
PANELS = ["unconfined", "0.58 kPa", "0.85 kPa", "1.1 kPa"]


def _draw_spheroid(ax, pos, r, sg, L, norm, cmap):
    """Draw one spheroid as true-to-scale discs (xy projection, back-to-front)."""
    ax.clear()
    order = np.argsort(pos[:, 2])                 # back (-z) first, front (+z) on top
    ec = EllipseCollection(widths=2 * r[order], heights=2 * r[order], angles=0,
                           units="xy", offsets=pos[order, :2],
                           transOffset=ax.transData, cmap=cmap, norm=norm,
                           edgecolors="k", linewidths=0.25)
    ec.set_array(sg[order])
    ax.add_collection(ec)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def main(n_frames: int = 40, fps: int = 8):
    os.makedirs(FIGDIR, exist_ok=True)
    sp = SpheroidParams()
    res = run_conditions(sp, include_free=True)

    n_rec = min(len(res[k].history) for k in PANELS)
    idx = np.linspace(0, n_rec - 1, min(n_frames, n_rec)).astype(int)

    norm = colors.Normalize(vmin=0.0, vmax=85.0)
    cmap = cm.inferno
    # SHARED scale across panels so relative spheroid sizes are honest
    L = 1.05 * max(
        (np.linalg.norm(res[k].history[idx[-1]]["pos"][:, :2], axis=1)
         + res[k].history[idx[-1]]["r"]).max() for k in PANELS)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.suptitle("Mechano-osmotic spheroid growth: unconfined vs ECM confinement "
                 "(true-to-scale cells, coloured by stress $\\sigma_g$)",
                 fontsize=12, fontweight="bold")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.01)
    cbar.set_label(r"$\sigma_g$ [Pa]")

    def draw(fi):
        fr = idx[fi]
        for ax, label in zip(axes, PANELS):
            snap = res[label].history[fr]
            _draw_spheroid(ax, snap["pos"], snap["r"], snap["sigma_g"], L, norm, cmap)
            ax.set_title(f"{label}\n$t$={snap['t']/24:.0f} d,  $N$={snap['N']}",
                         fontsize=10)
        return axes

    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False)
    out = os.path.join(FIGDIR, "spheroid_growth.gif")
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"wrote {out}  ({len(idx)} frames, shared scale L={L:.0f} um)")


if __name__ == "__main__":
    main()
