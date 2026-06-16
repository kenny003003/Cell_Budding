"""
Animate the simplified open spheroid model: unconfined vs ECM-confined tumour
growth, cells coloured by their mechano-osmotic compressive stress sigma_g.

The unconfined spheroid keeps proliferating; stiffer matrices build hydrostatic
stress (bright) that holds cells below the mitotic volume checkpoint, arresting
growth -- the paper's stress-dependent-growth result, rendered as a movie.

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
from matplotlib import cm, colors

from spheroid_model import SpheroidParams, run_conditions

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
PANELS = ["unconfined", "0.58 kPa", "0.85 kPa", "1.1 kPa"]


def main(n_frames: int = 40, fps: int = 8):
    os.makedirs(FIGDIR, exist_ok=True)
    sp = SpheroidParams()
    res = run_conditions(sp, include_free=True)

    # common time axis (all conditions share the same recorded frames)
    n_rec = min(len(res[k].history) for k in PANELS)
    idx = np.linspace(0, n_rec - 1, min(n_frames, n_rec)).astype(int)

    # shared stress colour scale and per-panel spatial extent
    norm = colors.Normalize(vmin=0.0, vmax=85.0)
    cmap = cm.inferno
    extent = {k: max(np.linalg.norm(res[k].history[idx[-1]]["pos"], axis=1).max(),
                     1.0) * 1.15 for k in PANELS}

    fig = plt.figure(figsize=(14, 4.2))
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    fig.suptitle("Mechano-osmotic spheroid growth: unconfined vs ECM confinement "
                 "(cells coloured by compressive stress $\\sigma_g$)",
                 fontsize=12, fontweight="bold")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.01)
    cbar.set_label(r"$\sigma_g$ [Pa]")

    def draw(frame_i):
        fr = idx[frame_i]
        for ax, label in zip(axes, PANELS):
            ax.clear()
            snap = res[label].history[fr]
            pos, sg, r = snap["pos"], snap["sigma_g"], snap["r"]
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                       s=24 * (r / r.mean()) ** 2, c=sg, cmap=cmap, norm=norm,
                       edgecolors="k", linewidths=0.2, depthshade=True)
            L = extent[label]
            ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f"{label}\n$t$={snap['t']/24:.0f} d,  $N$={snap['N']}",
                         fontsize=10)
        return axes

    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False)
    out = os.path.join(FIGDIR, "spheroid_growth.gif")
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"wrote {out}  ({len(idx)} frames)")


if __name__ == "__main__":
    main()
