"""
Local ECM defect -> pressure-driven bleb / budding.

The trend model (spheroid_model.py) uses a *mean-field* confinement stress, which
is spatially ~uniform and so cannot produce a localised protrusion. Here we use a
genuine force-based confinement: cells are soft spheres with repulsion (volume
exclusion) and short-range cohesion (surface tension), held inside a spherical
matrix boundary of stiffness ~E_ECM. Growth pressurises the spheroid against that
boundary.

We then cut a **local defect**: an angular patch of the matrix where the confining
stiffness drops to near zero (a hole / soft spot in the ECM). The internal
hydrostatic pressure is now unbalanced there, so cells are squeezed out through
the defect and -- because the low stress there lets them pass the volume
checkpoint while confined cells arrest -- they keep proliferating. The escaped
cells fill their own growing spherical cavity, tangent to the mother at the hole,
so the bleb rounds up into a free daughter spheroid bulging into open space
(a confined mother spheroid with a proliferating, unconfined bud).

This reuses the single-cell mechano-osmotic volume law (L1) for cell size/growth;
only L2/L3 are replaced by the explicit force model.

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
    n_seed: int = 220           # initial cells (many small cells -> smooth sphere)
    cell_scale: float = 0.5     # shrink cells (scale-invariant; smoother-looking sphere)
    # confinement (force-based ECM boundary) --------------------------------
    E_ecm: float = 1100.0       # matrix stiffness [Pa]
    c_wall: float = 1.0         # confinement stress = c_wall * E_ecm * bulk overpacking
    sigma_cap: float = 320.0    # cap on confinement stress [Pa] (avoids runaway)
    phi_max: float = 0.64       # packing fraction (cavity capacity)
    compress: float = 0.985     # cavity radius = compress * seeded packing radius
    # the local defect ------------------------------------------------------
    defect_dir: tuple = (1.0, 0.0, 0.0)   # direction of the weak patch
    defect_halfangle: float = 0.45         # cone half-angle [rad] (~26 deg, small hole)
    defect_depth: float = 1.0              # 1 = stiffness -> 0 at the patch centre
    defect_open_t: float = 16.0            # [h] open the defect after this time
    div_bias: float = 1.6                  # outward bias for confined cells (extrusion)
    pinch_ratio: float = 1.9               # detach when R_bud > pinch_ratio * neck radius
    detach_speed: float = 0.45             # drift of the detached daughter [um/h]
    # cell-cell mechanics ---------------------------------------------------
    relax_frac: float = 0.30    # fraction of overlap/penetration resolved per sweep
    k_coh: float = 0.15         # cohesion (surface tension) relative to repulsion
    coh_range: float = 2.5      # cohesive range beyond contact [um]
    sub_steps: int = 9          # mechanical relaxation sweeps per growth step
    # growth ----------------------------------------------------------------
    dt: float = 2.0             # synthesis timestep [h]
    t_max: float = 240.0
    tau_div: float = 14.0       # mean delay to divide once past checkpoint [h]
    checkpoint_alpha: float = 30.0   # sharper tissue-scale sizing checkpoint
    n_max: int = 600
    seed: int = 1
    cell: Params = field(default_factory=default_params)

    def __post_init__(self):
        # shrink the cell (scale-invariant) so the spheroid is built from many
        # small cells; coh_range is a length, so it scales too.
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

        # seed a compact ball
        R = scm.radius_from_volume(self.V.sum() / 0.64)
        u = self.rng.normal(size=(n0, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        self.pos = u * (R * self.rng.random(n0)[:, None] ** (1 / 3))

        self.n_hat = np.array(bp.defect_dir, float)
        self.n_hat /= np.linalg.norm(self.n_hat)
        self.t = 0.0
        self.cav_center = self.pos.mean(0).copy()
        self.R_cav = 1e6                      # no confinement while seeding
        self.escaped = np.zeros(n0, bool)
        self._relax(40)

        # fix the matrix cavity about the seeded bulk (a structure fixed in space,
        # not following the cells); slightly pre-compressed so the bulk is confined
        self.cav_center = self.pos.mean(0).copy()
        self.R_cav = bp.compress * scm.radius_from_volume(self.V.sum() / bp.phi_max)
        self.escaped = np.zeros(self.pos.shape[0], bool)
        # thin-neck pinch-off state
        self.r_neck = self.R_cav * np.sin(bp.defect_halfangle)   # half-width of the hole
        self.detached = False
        self.detach_gap = 0.0          # extra separation of the daughter after pinch-off
        self.detach_t = None
        self.R_bud = 0.0
        self.history = []

    # geometry -------------------------------------------------------------
    def _packing_radius(self):
        c = self.pos.mean(0)
        return (np.linalg.norm(self.pos - c, axis=1) + self.r).max()

    def _confine_stiffness(self, u):
        """Per-cell relative confinement stiffness given each cell's outward unit
        vector u: ~1 (intact wall) everywhere except inside the opened defect
        cone, where it drops sharply to ~0 (a hole in the matrix)."""
        bp = self.bp
        if self.t < bp.defect_open_t or self.detached:
            return np.ones(u.shape[0])     # intact wall (or healed after pinch-off)
        cosang = u @ self.n_hat
        c0 = np.cos(bp.defect_halfangle)
        edge = max(0.03, 0.4 * (1.0 - c0))            # edge width scales with cone size
        t = np.clip((cosang - c0) / edge, 0.0, 1.0)   # 0 outside cone -> 1 toward axis
        return 1.0 - bp.defect_depth * t

    def _bulk_overpack(self):
        """Over-packing of the *confined* bulk (cells still inside the cavity)."""
        V_in = self.V[~self.escaped].sum()
        cap = self.bp.phi_max * 4.0 / 3.0 * np.pi * self.R_cav ** 3
        return max(0.0, V_in / cap - 1.0)

    # mechanical relaxation: geometry-based MOTION (stable, unit-free), with a
    # separate mean-field (Pa) confinement STRESS that is switched off inside
    # the defect cone so cells there can grow and be extruded -----------------
    def _relax(self, n=None):
        bp = self.bp
        n = bp.sub_steps if n is None else n
        for _ in range(n):
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            d = np.linalg.norm(diff, axis=2); np.fill_diagonal(d, np.inf)
            rs = self.r[:, None] + self.r[None, :]
            hat = np.divide(diff, d[:, :, None], out=np.zeros_like(diff),
                            where=d[:, :, None] < np.inf)
            overlap = np.where(d < rs, rs - d, 0.0)                 # volume exclusion
            gap = d - rs
            coh = np.where((gap > 0) & (gap < bp.coh_range), gap, 0.0)  # cohesion
            # cohesion acts only within a tissue (bulk-bulk or bud-bud), not across
            # them, so the bud rounds into its OWN spheroid instead of wetting the
            # mother; bud and mother still exclude each other by repulsion.
            same = self.escaped[:, None] == self.escaped[None, :]
            m = overlap - bp.k_coh * coh * same
            disp = bp.relax_frac * (m[:, :, None] * hat).sum(axis=1)

            rel = self.pos - self.cav_center
            rho = np.linalg.norm(rel, axis=1)
            u = np.divide(rel, rho[:, None], out=np.zeros_like(rel), where=rho[:, None] > 1e-9)
            sfac_hole = self._confine_stiffness(u)                  # 1 wall, 0 hole
            # matrix shell is a two-sided barrier with a hole: confined cells stay
            # inside the cavity, escaped cells stay outside it (can't wrap back in)
            # mother cavity: confine non-escaped cells (except at the hole)
            pen_in = np.where((rho + self.r > self.R_cav) & (~self.escaped),
                              rho + self.r - self.R_cav, 0.0) * sfac_hole
            disp -= bp.relax_frac * pen_in[:, None] * u
            # bud: escaped cells fill their OWN growing spherical cavity, tangent
            # to the mother at the hole, so the bleb is a round spheroid bulging
            # into open space (its radius and centre grow with the bud's volume).
            if self.escaped.any():
                R_bud = (self.escaped.sum() / bp.phi_max) ** (1.0 / 3.0) \
                    * self.r[self.escaped].mean()
                self.R_bud = R_bud
                bud_center = self.cav_center \
                    + (self.R_cav + R_bud + self.detach_gap) * self.n_hat
                relb = self.pos - bud_center
                rhob = np.linalg.norm(relb, axis=1)
                ub = np.divide(relb, rhob[:, None], out=np.zeros_like(relb),
                               where=rhob[:, None] > 1e-9)
                penb = np.where((rhob + self.r > R_bud) & self.escaped,
                                rhob + self.r - R_bud, 0.0)
                disp -= bp.relax_frac * penb[:, None] * ub
            self.pos = self.pos + disp
            # a cell that has crossed the wall through the hole is now in open
            # space and stays free thereafter
            self.escaped |= np.linalg.norm(self.pos - self.cav_center, axis=1) > self.R_cav

        # stress: the confined bulk is pressurised; escaped cells (open space) are
        # free (sigma_g = 0), so they grow, divide and round up into a spheroid.
        sigma_mean = min(bp.c_wall * bp.E_ecm * self._bulk_overpack(), bp.sigma_cap)
        self.sigma_g = np.where(self.escaped, 0.0, sigma_mean)
        self.V = self.tab.volume(self.n_n, self.sigma_g)
        self.r = scm.radius_from_volume(self.V)

    # division -------------------------------------------------------------
    def _divide(self):
        bp, p = self.bp, self.p
        if self.pos.shape[0] >= bp.n_max:
            return
        # sharper sizing checkpoint for the tissue-scale demo (cleanly separates
        # the arrested confined bulk from the proliferating unconfined bud)
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
                # in the bud's own cavity: divide isotropically -> fills a sphere
                axis = self.rng.normal(size=3)
            else:
                # confined: bias the daughter toward free space (out the hole),
                # which helps cells extrude through the defect
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

    # main loop ------------------------------------------------------------
    def run(self, record_every=1):
        bp, p = self.bp, self.p
        steps = int(round(bp.t_max / bp.dt))
        for k in range(steps + 1):
            self.t = k * bp.dt
            self.n_n = np.minimum(self.n_n + p.beta * bp.dt * 3600.0, p.n_sat)
            self._relax()
            self._divide()
            # thin-neck pinch-off: once the bud outgrows the neck set by the
            # defect, surface tension severs the connection. The daughter detaches
            # and drifts off as a free-floating spheroid; the mother's defect heals
            # (so no more cells are extruded and the mother stays confined).
            if (not self.detached) and self.escaped.any() \
                    and self.R_bud > bp.pinch_ratio * self.r_neck:
                self.detached = True
                self.detach_t = self.t
            if self.detached:
                self.detach_gap += bp.detach_speed * bp.dt
            if k % record_every == 0:
                self.history.append(dict(t=self.t, pos=self.pos.copy(),
                                         r=self.r.copy(), sigma_g=self.sigma_g.copy(),
                                         N=self.pos.shape[0], detached=self.detached,
                                         escaped=self.escaped.copy()))
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
    table = VolumeTable(bp.cell)
    s = BlebSpheroid(bp, table)
    s.run()

    norm = colors.Normalize(0, 90); cmap = cm.inferno
    # centre each frame on the spheroid's centre of mass (it drifts as it buds)
    L = 1.05 * max((np.linalg.norm(f["pos"][:, :2] - f["pos"][:, :2].mean(0), axis=1)
                    + f["r"]).max() for f in s.history)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85); cb.set_label(r"$\sigma_g$ [Pa]")

    def draw(i):
        f = s.history[i]
        ax.clear()
        pos = f["pos"] - f["pos"].mean(0)        # centre on centre of mass
        r, sg = f["r"], f["sigma_g"]
        order = np.argsort(pos[:, 2])
        ec = EllipseCollection(widths=2 * r[order], heights=2 * r[order], angles=0,
                               units="xy", offsets=pos[order, :2],
                               transOffset=ax.transData, cmap=cmap, norm=norm,
                               edgecolors="k", linewidths=0.25)
        ec.set_array(sg[order]); ax.add_collection(ec)
        if f.get("detached"):
            state = "neck pinched off → daughter detached"
        elif f["t"] >= bp.defect_open_t:
            state = "defect open"
            ax.annotate("ECM defect", xy=(L * 0.92, 0), xytext=(L * 0.55, L * 0.8),
                        fontsize=9, color="#1a7d1a",
                        arrowprops=dict(arrowstyle="->", color="#1a7d1a", lw=1.5))
        else:
            state = "defect intact"
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"ECM defect → bud → thin-neck pinch-off\n"
                     f"t={f['t']/24:.1f} d,  N={f['N']},  {state}", fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(s.history), blit=False)
    out = os.path.join(FIGDIR, "spheroid_bleb.gif")
    anim.save(out, writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)
    print(f"wrote {out}  ({len(s.history)} frames, final N={s.history[-1]['N']})")


if __name__ == "__main__":
    main()
