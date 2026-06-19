"""
Local ECM defect -> budding -> surface-tension (capillary) neck failure.

Design split (as requested):
  * BUDDING is set up by construction: a confined mother spheroid sits in an
    elastic ECM cavity; a local defect (hole) lets cells extrude and form a bud,
    which is held as its own growing spherical cavity tangent to the mother so it
    rounds into a clean daughter spheroid. This part is "scripted" geometry.
  * The FAILURE (detachment) is NOT scripted -- it is driven by surface energy.
    The neck connecting the two spheroids is a constriction of radius a. Surface
    tension gives it a Laplace (curvature) pressure gamma/a; the bud has gamma/R.
    When the neck is narrower than the bud (a < R) its higher pressure expels
    material into the bud, so the neck thins -- and because thinning raises its
    curvature further, the process is autocatalytic (the Rayleigh-Plateau
    capillary instability). The neck collapses on its own and the daughter
    detaches; once it does, the still-pressurised mother ejects the freed
    daughter away (a force, ~ the mother's hydrostatic pressure).

So budding is geometric, but WHEN and WHETHER the bud pinches off is decided by
the surface-tension energetics of the neck (gamma, the hole size, and the bud
size), not by a hand-set size threshold.

This reuses the single-cell mechano-osmotic volume law (L1) for cell size/growth.

Run:  python spheroid_bleb.py   ->  writes ../figures/spheroid_bleb.gif
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from parameters import Params, default_params, scaled_params
import single_cell_model as scm
from spheroid_model import VolumeTable


@dataclass
class BlebParams:
    n_seed: int = 480           # initial cells -> a large mother spheroid
    cell_scale: float = 0.5     # shrink cells (scale-invariant; smoother sphere)
    # confinement (ECM cavity) ----------------------------------------------
    E_ecm: float = 1100.0       # matrix stiffness [Pa]
    c_wall: float = 1.0         # confinement stress = c_wall * E_ecm * bulk overpacking
    sigma_cap: float = 320.0    # cap on confinement stress [Pa]
    phi_max: float = 0.64       # packing fraction (cavity capacity)
    compress: float = 0.98      # cavity radius = compress * seeded packing radius
    # the local defect ------------------------------------------------------
    defect_dir: tuple = (1.0, 0.0, 0.0)   # direction of the weak patch
    defect_halfangle: float = 0.22         # cone half-angle [rad] -> thin neck (early pinch)
    defect_depth: float = 1.0
    defect_open_t: float = 16.0            # [h] open the defect after this time
    div_bias: float = 1.6                  # outward bias for confined cells (extrusion)
    # surface-tension neck failure (the physical detachment trigger) --------
    gamma: float = 1.0          # surface tension (relative); scales neck Laplace pressure
    neck_rate: float = 3.0      # capillary thinning rate [um^2/h]
    a_min: float = 0.6          # neck collapses (detaches) below this radius [um]
    detach_eject: float = 0.6   # post-detach ejection speed per unit pressure [um/h]
    # cell-cell mechanics ---------------------------------------------------
    relax_frac: float = 0.30
    k_coh: float = 0.15         # cohesion (also sets surface tension scale)
    coh_range: float = 2.5      # cohesive range beyond contact [um]
    sub_steps: int = 7
    # growth ----------------------------------------------------------------
    dt: float = 2.0
    t_max: float = 140.0
    tau_div: float = 12.0
    checkpoint_alpha: float = 30.0
    n_max: int = 580
    seed: int = 1
    cell: Params = field(default_factory=default_params)

    def __post_init__(self):
        if self.cell_scale != 1.0:
            self.cell = scaled_params(self.cell, self.cell_scale)
            self.coh_range *= self.cell_scale


class BlebSpheroid:
    def __init__(self, bp: BlebParams, table: VolumeTable):
        self.bp = bp
        self.tab = table
        self.p = bp.cell
        self.rng = np.random.default_rng(bp.seed)
        n0 = bp.n_seed

        self.n_n = self.rng.uniform(0.6 * self.p.n_n0, 0.9 * self.p.n_sat, size=n0)
        self.sigma_g = np.zeros(n0)
        self.V = self.tab.volume(self.n_n, self.sigma_g)
        self.r = scm.radius_from_volume(self.V)

        R = scm.radius_from_volume(self.V.sum() / 0.64)
        u = self.rng.normal(size=(n0, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        self.pos = u * (R * self.rng.random(n0)[:, None] ** (1 / 3))

        self.n_hat = np.array(bp.defect_dir, float); self.n_hat /= np.linalg.norm(self.n_hat)
        self.t = 0.0
        self.cav_center = self.pos.mean(0).copy()
        self.R_cav = 1e6                      # no confinement while seeding
        self.escaped = np.zeros(n0, bool)
        self.R_bud = 0.0
        self.detach_gap = 0.0
        self._relax(40)

        self.cav_center = self.pos.mean(0).copy()
        self.R_cav = bp.compress * scm.radius_from_volume(self.V.sum() / bp.phi_max)
        self.escaped = np.zeros(self.pos.shape[0], bool)
        self.a_neck = self.R_cav * np.sin(bp.defect_halfangle)   # neck radius = hole radius
        self.detached = False
        self.sigma_mean = 0.0
        self.history = []

    # geometry -------------------------------------------------------------
    def _confine_stiffness(self, u):
        bp = self.bp
        if self.t < bp.defect_open_t:
            return np.ones(u.shape[0])
        cosang = u @ self.n_hat
        c0 = np.cos(bp.defect_halfangle)
        edge = max(0.03, 0.4 * (1.0 - c0))
        t = np.clip((cosang - c0) / edge, 0.0, 1.0)
        return 1.0 - bp.defect_depth * t

    def _bulk_overpack(self):
        V_in = self.V[~self.escaped].sum()
        cap = self.bp.phi_max * 4.0 / 3.0 * np.pi * self.R_cav ** 3
        return max(0.0, V_in / cap - 1.0)

    # mechanics: mother confined in its cavity; the bud (escaped cells) fills its
    # own growing spherical cavity tangent to the mother -> a clean round daughter
    def _relax(self, n=None):
        bp = self.bp
        n = bp.sub_steps if n is None else n
        for _ in range(n):
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            d = np.linalg.norm(diff, axis=2); np.fill_diagonal(d, np.inf)
            rs = self.r[:, None] + self.r[None, :]
            hat = np.divide(diff, d[:, :, None], out=np.zeros_like(diff),
                            where=d[:, :, None] < np.inf)
            overlap = np.where(d < rs, rs - d, 0.0)
            gap = d - rs
            coh = np.where((gap > 0) & (gap < bp.coh_range), gap, 0.0)
            same = self.escaped[:, None] == self.escaped[None, :]   # cohesion within a tissue
            m = overlap - bp.k_coh * coh * same
            disp = bp.relax_frac * (m[:, :, None] * hat).sum(axis=1)

            rel = self.pos - self.cav_center
            rho = np.linalg.norm(rel, axis=1)
            u = np.divide(rel, rho[:, None], out=np.zeros_like(rel), where=rho[:, None] > 1e-9)
            sfac = self._confine_stiffness(u)
            pen_in = np.where((rho + self.r > self.R_cav) & (~self.escaped),
                              rho + self.r - self.R_cav, 0.0) * sfac
            disp -= bp.relax_frac * pen_in[:, None] * u
            if self.escaped.any():
                R_bud = (self.escaped.sum() / bp.phi_max) ** (1.0 / 3.0) \
                    * self.r[self.escaped].mean()
                self.R_bud = R_bud
                bud_center = self.cav_center + (self.R_cav + R_bud + self.detach_gap) * self.n_hat
                relb = self.pos - bud_center
                rhob = np.linalg.norm(relb, axis=1)
                ub = np.divide(relb, rhob[:, None], out=np.zeros_like(relb),
                               where=rhob[:, None] > 1e-9)
                penb = np.where((rhob + self.r > R_bud) & self.escaped,
                                rhob + self.r - R_bud, 0.0)
                disp -= bp.relax_frac * penb[:, None] * ub
            self.pos = self.pos + disp
            self.escaped |= np.linalg.norm(self.pos - self.cav_center, axis=1) > self.R_cav

        self.sigma_mean = min(bp.c_wall * bp.E_ecm * self._bulk_overpack(), bp.sigma_cap)
        self.sigma_g = np.where(self.escaped, 0.0, self.sigma_mean)
        self.V = self.tab.volume(self.n_n, self.sigma_g)
        self.r = scm.radius_from_volume(self.V)

    # surface-tension capillary failure of the neck (the physical detach trigger)
    def _neck_update(self):
        bp = self.bp
        if self.detached or self.t < bp.defect_open_t or not self.escaped.any():
            return
        if self.R_bud <= 0.0:
            return
        # Laplace pressures: neck gamma/a vs bud gamma/R. When a < R the neck is
        # over-pressured and drains into the bud -> thins. d(a)/dt = -rate*gamma*(1/a - 1/R).
        # Thinning raises 1/a, so it accelerates: the Rayleigh-Plateau instability.
        drive = (1.0 / self.a_neck - 1.0 / self.R_bud)
        if drive > 0.0:
            self.a_neck -= bp.neck_rate * bp.gamma * drive * bp.dt
        if self.a_neck <= bp.a_min:
            self.detached = True

    def _divide(self):
        bp, p = self.bp, self.p
        if self.pos.shape[0] >= bp.n_max:
            return
        Pdiv = p.lam / (1.0 + np.exp(-bp.checkpoint_alpha * (self.V / p.V_div - 1.0)))
        rate = Pdiv * (bp.dt / bp.tau_div)
        idx = np.where((self.V >= p.V_div) & (self.rng.random(self.V.shape) < rate))[0]
        if idx.size == 0:
            return
        new_pos, new_nn, new_esc, keep = [], [], [], self.n_n.copy()
        for i in idx:
            if self.pos.shape[0] + len(new_pos) >= bp.n_max:
                break
            if self.escaped[i]:
                axis = self.rng.normal(size=3)
            else:
                rad = self.pos[i] - self.cav_center
                nr = np.linalg.norm(rad)
                rad = rad / nr if nr > 1e-6 else self.rng.normal(size=3)
                dvec = self.rng.normal(size=3); dvec /= np.linalg.norm(dvec) + 1e-12
                axis = dvec + bp.div_bias * rad
            axis /= np.linalg.norm(axis) + 1e-12
            off = 0.5 * self.r[i] * axis
            self.pos[i] = self.pos[i] + off
            new_pos.append(self.pos[i] - 2 * off)
            keep[i] = self.n_n[i] / 2.0; new_nn.append(self.n_n[i] / 2.0)
            new_esc.append(bool(self.escaped[i]))
        if new_pos:
            self.pos = np.vstack([self.pos, np.array(new_pos)])
            self.n_n = np.concatenate([keep, np.array(new_nn)])
            self.sigma_g = np.concatenate([self.sigma_g, np.zeros(len(new_pos))])
            self.escaped = np.concatenate([self.escaped, np.array(new_esc, bool)])
            self.V = self.tab.volume(self.n_n, self.sigma_g)
            self.r = scm.radius_from_volume(self.V)

    def run(self, record_every=1):
        bp, p = self.bp, self.p
        steps = int(round(bp.t_max / bp.dt))
        for k in range(steps + 1):
            self.t = k * bp.dt
            self.n_n = np.minimum(self.n_n + p.beta * bp.dt * 3600.0, p.n_sat)
            self._relax()
            self._divide()
            self._neck_update()                       # surface-tension neck thinning
            if self.detached:
                # the still-pressurised mother ejects the freed daughter outward
                self.detach_gap += bp.detach_eject * (self.sigma_mean / bp.sigma_cap) * bp.dt
            if k % record_every == 0:
                self.history.append(dict(t=self.t, pos=self.pos.copy(), r=self.r.copy(),
                                         sigma_g=self.sigma_g.copy(), N=self.pos.shape[0],
                                         budding=bool(self.escaped.any()),
                                         detached=self.detached, a_neck=self.a_neck))
        return self.history


def main():
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import EllipseCollection
    from matplotlib import cm, colors

    FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(FIGDIR, exist_ok=True)

    bp = BlebParams()
    s = BlebSpheroid(bp, VolumeTable(bp.cell))
    s.run()

    norm = colors.Normalize(0, 90); cmap = cm.inferno
    L = 1.05 * max((np.linalg.norm(f["pos"][:, :2] - f["pos"][:, :2].mean(0), axis=1)
                    + f["r"]).max() for f in s.history)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85); cb.set_label(r"$\sigma_g$ [Pa]")

    def draw(i):
        f = s.history[i]; ax.clear()
        pos = f["pos"] - f["pos"].mean(0)
        r, sg = f["r"], f["sigma_g"]
        order = np.argsort(pos[:, 2])
        ec = EllipseCollection(widths=2 * r[order], heights=2 * r[order], angles=0,
                               units="xy", offsets=pos[order, :2],
                               transOffset=ax.transData, cmap=cmap, norm=norm,
                               edgecolors="k", linewidths=0.25)
        ec.set_array(sg[order]); ax.add_collection(ec)
        if f["detached"]:
            state = "neck collapsed (surface tension) → detached"
        elif f["budding"]:
            state = f"budding · neck a={f['a_neck']:.0f}µm"
            ax.annotate("ECM defect", xy=(L * 0.92, 0), xytext=(L * 0.5, L * 0.82),
                        fontsize=9, color="#1a7d1a",
                        arrowprops=dict(arrowstyle="->", color="#1a7d1a", lw=1.5))
        else:
            state = "defect intact" if f["t"] < bp.defect_open_t else "defect open"
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"budding (geometric) → capillary neck failure\n"
                     f"t={f['t']/24:.1f} d,  N={f['N']},  {state}", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(s.history), blit=False)
    out = os.path.join(FIGDIR, "spheroid_bleb.gif")
    anim.save(out, writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)
    print(f"wrote {out}  ({len(s.history)} frames, final N={s.history[-1]['N']})")


if __name__ == "__main__":
    main()
