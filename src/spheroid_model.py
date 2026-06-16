"""
Simplified *open* reproduction of the multicellular spheroid result
(Senthilkumar et al., PNAS 2026, ~Fig. 4): stress-/stiffness-dependent tumour
growth that emerges from the single-cell mechano-osmotic sizing checkpoint.

Why this module exists
----------------------
The authors' own multicellular code (KU Leuven GitLab) runs on the proprietary
*Mpacts* discrete-element engine plus *Abaqus* and a neural-network surrogate for
the extracellular-matrix (ECM) finite-element response. None of that is openly
runnable. Here we keep the *physics the paper attributes the result to* -- a
mitotic volume checkpoint that a stiff ECM pushes cells below -- but replace the
heavy machinery with a lightweight, fully open pressure-coupled particle model:

  * Each cell is a soft sphere whose volume is set by the *single-cell*
    mechano-osmotic balance (single_cell_model.solve_state): given its
    biomolecule content n_n and the external compressive stress sigma_g acting
    on its surface, the cell takes its quasi-static hydromechanical volume.
  * Cells synthesise biomolecules (dn_n/dt = beta) and divide stochastically
    once they pass the volume checkpoint V_div (single_cell_model.p_division).
  * Cells repel on contact (linear soft-sphere) and are confined by an elastic
    ECM boundary whose stiffness is E_ecm. Contact + ECM forces give every cell
    an isotropic compressive stress sigma_g, which feeds straight back into its
    hydromechanical volume -- closing the mechano-osmotic loop at tissue scale.
  * The ECM boundary remodels plastically (slowly yields to the tumour), so a
    soft matrix is outgrown while a stiff matrix holds cells below V_div and
    arrests growth. This is the paper's mechanism, reproduced from the bottom up.

The interior pressure is *emergent*: the ECM only pushes the outermost cells,
and that load propagates inward through cell-cell contacts, so deep cells feel
the highest stress -- mirroring the paper's finding that confinement raises
hydrostatic pressure most in the spheroid core.

This is a reduced model: it is calibrated to reproduce the qualitative
stress-dependent-growth *trend*, not the paper's exact spheroid geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from parameters import Params, default_params
import single_cell_model as scm


# ==========================================================================
# fast volume lookup  V(n_n, sigma_g)
# ==========================================================================
class VolumeTable:
    """Tabulated quasi-static cell volume V(n_n, sigma_g) for fast, vectorised
    evaluation inside the spheroid loop (solve_state is a per-call root find)."""

    def __init__(self, p: Params, n_n_grid=None, sg_grid=None):
        self.p = p
        if n_n_grid is None:
            n_n_grid = np.linspace(0.5 * p.n_n0, p.n_sat, 70)
        if sg_grid is None:
            sg_grid = np.linspace(0.0, 3000.0, 80)
        self.n_n_grid = n_n_grid
        self.sg_grid = sg_grid
        V = np.empty((n_n_grid.size, sg_grid.size))
        for i, nn in enumerate(n_n_grid):
            for j, sg in enumerate(sg_grid):
                V[i, j] = scm.solve_state(float(nn), float(sg), p)["V"]
        self._V = V
        self._interp = RegularGridInterpolator(
            (n_n_grid, sg_grid), V, bounds_error=False, fill_value=None)

    def volume(self, n_n, sigma_g):
        n_n = np.clip(n_n, self.n_n_grid[0], self.n_n_grid[-1])
        sigma_g = np.clip(sigma_g, self.sg_grid[0], self.sg_grid[-1])
        return self._interp(np.column_stack([np.ravel(n_n), np.ravel(sigma_g)]))

    def radius(self, n_n, sigma_g):
        return scm.radius_from_volume(self.volume(n_n, sigma_g))


# ==========================================================================
# spheroid parameters
# ==========================================================================
@dataclass
class SpheroidParams:
    """Tissue-scale mechanics + growth controls for the particle model.

    Mechanical constants are calibrated (single set, shared across all ECM
    stiffnesses) so the three reported matrices 0.58 / 0.85 / 1.1 kPa span a
    clear growth -> arrest gradient. E_ecm is the only quantity changed between
    conditions; everything else is fixed.
    """

    # seeding ----------------------------------------------------------------
    n_seed: int = 24             # initial number of (desynchronised) cells

    # contact mechanics (positions / packing geometry) ----------------------
    relax_frac: float = 0.35     # fraction of overlap resolved per packing sweep
    relax_iters: int = 4         # packing relaxation iterations per growth step
    equil_passes: int = 2        # growth<->mechanics coupling passes per step

    # ECM confinement --------------------------------------------------------
    phi_max: float = 0.64        # random-close-packing fraction (cells per cavity vol)
    c_wall: float = 0.9          # ECM coupling: sigma = c_wall * E_ecm * overstrain
    plastic: float = 0.80        # fraction of tumour expansion the matrix accommodates
    radial_var: float = 0.6      # core/rim split of the confinement stress (spatial var.)

    # growth / division ------------------------------------------------------
    dt: float = 2.0              # synthesis timestep [h]
    t_max: float = 320.0         # total simulated time [h]
    tau_div: float = 3.0         # mean delay to divide once past checkpoint [h]
    n_max: int = 320             # stop subdividing past this many cells (runtime cap)
    split_offset: float = 0.6    # daughter separation as fraction of cell radius
    seed: int = 0

    # derived single-cell parameters (synthesis rate etc.) ------------------
    cell: Params = field(default_factory=default_params)


# ==========================================================================
# the spheroid simulation
# ==========================================================================
class Spheroid:
    """3-D soft-particle tumour spheroid with mechano-osmotic cell growth."""

    def __init__(self, E_ecm: float, sp: SpheroidParams, table: VolumeTable):
        self.E_ecm = float(E_ecm)          # ECM Young's modulus [Pa]
        self.sp = sp
        self.tab = table
        self.p = sp.cell
        self.rng = np.random.default_rng(sp.seed)

        # seed a small cluster of cells with desynchronised cell-cycle phases
        # (like a freshly seeded spheroid). Desynchronisation spreads divisions
        # in time, so the tumour grows smoothly instead of in doubling pulses.
        n0 = max(1, sp.n_seed)
        self.n_n = self.rng.uniform(0.5 * self.p.n_n0, 0.92 * self.p.n_sat, size=n0)
        self.sigma_g = np.zeros(n0)
        self.V = self.tab.volume(self.n_n, self.sigma_g)
        self.r = scm.radius_from_volume(self.V)
        # random close-ish initial packing inside a sphere
        Rseed = self.packing_radius()
        u = self.rng.normal(size=(n0, 3))
        u /= np.linalg.norm(u, axis=1)[:, None] + 1e-12
        rad = Rseed * self.rng.random(n0) ** (1.0 / 3.0)
        self.pos = u * rad[:, None]

        self.confined = self.E_ecm > 0.0
        self.V0_total = float(self.V.sum())                    # reference volume
        self.R_seed = self.packing_radius()                    # stress-free tumour radius
        self.R_wall = self.R_seed                              # accommodated cavity radius
        self._pack_positions()
        self.history = []                                      # per-frame snapshots

    # -- geometry ----------------------------------------------------------
    def _radial(self):
        c = self.pos.mean(axis=0)
        d = self.pos - c
        rho = np.linalg.norm(d, axis=1)
        hat = np.divide(d, rho[:, None], out=np.zeros_like(d), where=rho[:, None] > 1e-9)
        return c, rho, hat

    def tumour_radius(self):
        """Volume-equivalent radius of the whole spheroid."""
        return scm.radius_from_volume(self.V.sum())

    def packing_radius(self):
        """Radius of the sphere that holds the cells' *actual* volume at random
        close packing -- the physical outer radius of the spheroid."""
        return scm.radius_from_volume(self.V.sum() / self.sp.phi_max)

    # -- pack particles inside the ECM cavity (geometry only, for the spheroid
    #    shape and the animation) ------------------------------------------
    def _pack_positions(self):
        sp, N = self.sp, self.pos.shape[0]
        if N == 1:
            return
        for _ in range(sp.relax_iters):
            diff = self.pos[:, None, :] - self.pos[None, :, :]     # (N,N,3)
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            rsum = self.r[:, None] + self.r[None, :]
            overlap = np.where(dist < rsum, rsum - dist, 0.0)
            disp = np.zeros((N, 3))
            if overlap.any():
                hat = np.divide(diff, dist[:, :, None],
                                out=np.zeros_like(diff), where=dist[:, :, None] < np.inf)
                disp += sp.relax_frac * (overlap[:, :, None] * hat).sum(axis=1)
            # keep cells inside the spheroid's natural packing radius
            R_pack = self.packing_radius()
            c, rho, rhat = self._radial()
            overshoot = rho + self.r - R_pack
            wmask = overshoot > 0.0
            if wmask.any():
                disp -= sp.relax_frac * np.where(wmask, overshoot, 0.0)[:, None] * rhat
            self.pos = self.pos + disp

    # -- confinement stress from ECM elastic-plastic resistance ------------
    def _confinement_stress(self):
        """Mean-field compressive stress on each cell.

        As the tumour proliferates, its packing radius R_t grows above the
        stress-free seed radius R_seed. The surrounding matrix yields plastically
        but only accommodates a fraction `plastic` of that expansion, so the
        accommodated cavity is

            R_acc = R_seed * (1 + plastic * (R_t/R_seed - 1)).

        The remaining (elastic) mismatch strains the matrix, which resists with a
        pressure ~ E_ecm * (R_t/R_acc - 1) -- the persistent hydrostatic load
        that competes with osmotic growth in the paper. This is monotonic in both
        tumour size and ECM stiffness, so a stiffer matrix always suppresses
        growth more. A radial weighting makes core cells more compressed than rim
        cells (the paper's spatial volume variation).
        """
        sp = self.sp
        if not self.confined:
            self.R_wall = self.packing_radius()
            return np.zeros(self.n_n.shape)
        R_t = self.packing_radius()
        R_acc = self.R_seed * (1.0 + sp.plastic * (R_t / self.R_seed - 1.0))
        self.R_wall = R_acc
        overstrain = max(0.0, R_t / R_acc - 1.0)
        sigma_mean = sp.c_wall * self.E_ecm * overstrain          # Pa
        if sigma_mean <= 0.0 or self.pos.shape[0] == 1:
            return np.full(self.n_n.shape, sigma_mean)
        # core > rim: weight by depth (1 at centre -> down at rim), mean-normalised
        _, rho, _ = self._radial()
        depth = 1.0 - rho / (rho.max() + 1e-9)                    # 1 core, 0 rim
        w = 1.0 + sp.radial_var * (depth - depth.mean())
        return sigma_mean * np.clip(w, 0.0, None)

    # -- grow + mechanically equilibrate for one synthesis step ------------
    def _grow_and_equilibrate(self):
        # couple geometry <-> confinement <-> hydromechanical volume (few passes)
        for _ in range(self.sp.equil_passes):
            self._pack_positions()
            self.sigma_g = self._confinement_stress()
            self.V = self.tab.volume(self.n_n, self.sigma_g)
            self.r = scm.radius_from_volume(self.V)

    # -- division ----------------------------------------------------------
    def _divide(self):
        sp, p = self.sp, self.p
        if self.pos.shape[0] >= sp.n_max:
            return
        Pdiv = scm.p_division(self.V, p)                 # checkpoint pass prob.
        rate = Pdiv * (sp.dt / sp.tau_div)
        eligible = (self.V >= p.V_div) & (self.rng.random(self.V.shape) < rate)
        idx = np.where(eligible)[0]
        if idx.size == 0:
            return
        new_pos, new_nn = [], []
        keep_nn = self.n_n.copy()
        for i in idx:
            if self.pos.shape[0] + len(new_pos) >= sp.n_max:
                break
            d = self.rng.normal(size=3)
            d /= np.linalg.norm(d) + 1e-12
            off = sp.split_offset * self.r[i] * d
            self.pos[i] = self.pos[i] + off
            new_pos.append(self.pos[i] - 2 * off)
            keep_nn[i] = self.n_n[i] / 2.0               # split biomolecules
            new_nn.append(self.n_n[i] / 2.0)
        if new_pos:
            self.pos = np.vstack([self.pos, np.array(new_pos)])
            self.n_n = np.concatenate([keep_nn, np.array(new_nn)])
            self.sigma_g = np.concatenate([self.sigma_g, np.zeros(len(new_pos))])
            self.V = self.tab.volume(self.n_n, self.sigma_g)
            self.r = scm.radius_from_volume(self.V)

    # -- arrest bookkeeping ------------------------------------------------
    def arrested_mask(self):
        """Cells that cannot reach the checkpoint at their current confinement
        even with fully saturated synthesis (n_n = n_sat) -> permanently arrested."""
        V_cap = self.tab.volume(np.full(self.n_n.shape, self.p.n_sat), self.sigma_g)
        return V_cap < self.p.V_div

    # -- main loop ---------------------------------------------------------
    def run(self, record_every: int = 2):
        sp, p = self.sp, self.p
        n_steps = int(round(sp.t_max / sp.dt))
        for step in range(n_steps + 1):
            t = step * sp.dt
            # biomolecule synthesis (growth driver)
            self.n_n = np.minimum(self.n_n + p.beta * (sp.dt * 3600.0), p.n_sat)
            # mechanics + hydromechanical volume (sets sigma_g and R_wall)
            self._grow_and_equilibrate()
            # division
            self._divide()

            if step % record_every == 0 or step == n_steps:
                self.history.append(dict(
                    t=t, N=self.pos.shape[0],
                    pos=self.pos.copy(), r=self.r.copy(),
                    V=self.V.copy(), sigma_g=self.sigma_g.copy(),
                    V_total=float(self.V.sum()),
                    arrested=int(self.arrested_mask().sum()),
                    R_wall=self.R_wall,
                ))
        return self.history

    # -- convenience time series ------------------------------------------
    def series(self):
        h = self.history
        return dict(
            t=np.array([f["t"] for f in h]),
            N=np.array([f["N"] for f in h]),
            V_total=np.array([f["V_total"] for f in h]),
            arrested=np.array([f["arrested"] for f in h]),
            arrested_frac=np.array([f["arrested"] / max(f["N"], 1) for f in h]),
        )


# ==========================================================================
# driver: run the three reported ECM stiffnesses (+ unconfined control)
# ==========================================================================
# ECM Young's moduli used by the paper's AI surrogate (kPa -> Pa)
ECM_STIFFNESSES_KPA = (0.58, 0.85, 1.1)


def run_conditions(sp: SpheroidParams | None = None, include_free: bool = True):
    """Run the spheroid for each ECM stiffness. Returns {label: Spheroid}."""
    if sp is None:
        sp = SpheroidParams()
    table = VolumeTable(sp.cell)
    out = {}
    if include_free:
        s = Spheroid(E_ecm=0.0, sp=sp, table=table)
        s.run()
        out["unconfined"] = s
    for kpa in ECM_STIFFNESSES_KPA:
        s = Spheroid(E_ecm=kpa * 1000.0, sp=sp, table=table)
        s.run()
        out[f"{kpa:g} kPa"] = s
    return out


if __name__ == "__main__":
    sp = SpheroidParams()
    res = run_conditions(sp)
    print("ECM stiffness sweep (simplified open spheroid model)\n")
    print(f"{'condition':>12} | {'final N':>7} {'fold vol':>9} "
          f"{'arrested%':>9} {'<sig_g>Pa':>9}")
    for label, s in res.items():
        ser = s.series()
        sg = s.sigma_g.mean()
        print(f"{label:>12} | {ser['N'][-1]:7d} {ser['V_total'][-1]/s.V0_total:9.1f} "
              f"{100*ser['arrested_frac'][-1]:9.1f} {sg:9.1f}")
