"""
Single-cell mechano-osmotic growth model (Senthilkumar et al., PNAS 2026; Fig. 1).

State of the cell during the G1 -> M cycle:
    n_n   : amount of impermeable biomolecules (proteins) in the cytosol [amol]
            -- the slow growth driver, synthesised at rate beta (Eq. for dn_n/dt)
    V     : cell volume [um^3]            (set by water balance, Eq. 1)
    N     : amount of permeable ions [amol] (set by ion balance, Eq. 2)

Governing relations (paper Eqs. 1-2 + McEvoy et al. 2020 mechanics):

  Osmotic pressures (van't Hoff, dilute):
      Pi_n   = n_n R T / V          (impermeable biomolecules)
      Pi_ip  = N   R T / V          (permeable ions, internal)
      dPi_ip = Pi_ip - Pi_ext       (permeable-ion osmotic difference)
      dPi    = (n_n + N) R T / V - Pi_ext = Pi_n + dPi_ip   (total)

  Water flux across the semi-permeable membrane (Eq. 1):
      dV/dt = -A * Lp_m * (dP - dPi)
  Cortex mechanics (force balance for a sphere, McEvoy 2020 Eq. 3):
      sigma   = (K/2)(r^2/r0^2 - 1) + sigma_a          (cortical stress)
      dP      = sigma_g + 2 h sigma / r                 (hydrostatic difference)
  Permeable-ion transport (Eq. 2; mechanosensitive + leak + active pump):
      dN/dt = -A * [ (omega_ms(sigma) + omega_l + gamma) dPi_ip - gamma dPi_c ]
      omega_ms(sigma) = 0                       if sigma <= sigma_c
                      = omega_ms*(sigma-sigma_c) if sigma_c < sigma < sigma_s
                      = omega_ms*(sigma_s-sigma_c) if sigma >= sigma_s
  Biomolecule synthesis (growth driver):
      dn_n/dt = beta   while n_n < n_sat,  else 0

Separation of timescales
------------------------
Water equilibration (seconds-minutes) and ion equilibration are *fast*
compared with biomolecule synthesis (hours). The cell therefore tracks a
quasi-static balance dP = dPi at every instant, with the permeable-ion content
slaved to its pump-leak steady state. We integrate the single slow variable
n_n(t) and solve the fast (algebraic) balance for V at each time. A full stiff
ODE integrator (all three states) is also provided for verification and gives
the same trajectories.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

from parameters import Params, default_params


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def radius_from_volume(V: float) -> float:
    return (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)


def volume_from_radius(r: float) -> float:
    return 4.0 / 3.0 * np.pi * r ** 3


def surface_area(r: float) -> float:
    return 4.0 * np.pi * r ** 2


# --------------------------------------------------------------------------
# constitutive relations
# --------------------------------------------------------------------------
def cortical_stress(r: float, p: Params) -> float:
    """Cortical stress sigma = passive (area stretch) + active (myosin)."""
    sigma_p = 0.5 * p.K * (r ** 2 / p.r0 ** 2 - 1.0)
    return sigma_p + p.sigma_a


def hydrostatic_difference(r: float, sigma_g: float, p: Params) -> float:
    """dP = P_i - P_ext from cortex force balance plus external contact stress."""
    sigma = cortical_stress(r, p)
    return sigma_g + 2.0 * p.h * sigma / r


def omega_ms(sigma: float, p: Params) -> float:
    """Piecewise-linear mechanosensitive-channel permeability (McEvoy 2020 Eq. 5)."""
    if sigma <= p.sigma_c:
        return 0.0
    if sigma >= p.sigma_s:
        return p.omega_ms * (p.sigma_s - p.sigma_c)
    return p.omega_ms * (sigma - p.sigma_c)


def quasi_steady_dPi_ip(sigma: float, p: Params) -> float:
    """Permeable-ion osmotic difference at pump-leak-MS steady state.

    Setting dN/dt = 0 in Eq. 2 gives
        dPi_ip = gamma dPi_c / (omega_ms(sigma) + omega_l + gamma).
    """
    return p.gamma * p.dPi_c / (omega_ms(sigma, p) + p.omega_l + p.gamma)


# --------------------------------------------------------------------------
# quasi-static cell state for a given (n_n, sigma_g)
# --------------------------------------------------------------------------
def solve_state(n_n: float, sigma_g: float, p: Params) -> dict:
    """Solve the fast (water + ion) balance for the cell radius/volume.

    Finds r such that dP(r) = dPi_total(r), i.e. the water flux vanishes.
    dP increases with r (cortex stiffens) while dPi_total decreases with r
    (osmolytes dilute), so the root is unique. Returns all observables.
    """
    def imbalance(r: float) -> float:
        V = volume_from_radius(r)
        sigma = cortical_stress(r, p)
        dP = sigma_g + 2.0 * p.h * sigma / r
        dPi_ip = quasi_steady_dPi_ip(sigma, p)
        Pi_n = n_n * p.RT / V
        dPi_total = dPi_ip + Pi_n
        return dP - dPi_total

    # bracket the root. imbalance(r) is monotonically increasing (dP rises with
    # r, dPi falls with r), so the root is unique. Free cells sit above r0, but
    # strong external confinement can compress the equilibrium radius below r0,
    # so we expand the bracket in whichever direction is needed.
    f0 = imbalance(p.r0)
    if f0 < 0.0:                      # root above r0 (typical: cell is stretched)
        r_lo, f_lo = p.r0, f0
        r_hi = p.r0
        for _ in range(400):
            r_hi *= 1.05
            if f_lo * imbalance(r_hi) < 0:
                break
        else:
            raise RuntimeError("could not bracket quasi-static volume root (up)")
    else:                            # root below r0 (cell compressed by sigma_g)
        r_hi, f_hi = p.r0, f0
        r_lo = p.r0
        for _ in range(400):
            r_lo *= 0.97
            if f_hi * imbalance(r_lo) < 0:
                break
            if r_lo < 1e-2 * p.r0:
                break
        else:
            raise RuntimeError("could not bracket quasi-static volume root (down)")

    r = brentq(imbalance, r_lo, r_hi, xtol=1e-10, rtol=1e-12)

    V = volume_from_radius(r)
    A = surface_area(r)
    sigma = cortical_stress(r, p)
    dP = sigma_g + 2.0 * p.h * sigma / r
    dPi_ip = quasi_steady_dPi_ip(sigma, p)
    Pi_n = n_n * p.RT / V
    # permeable-ion amount slaved to its quasi-steady concentration
    N = (p.Pi_ext + dPi_ip) * V / p.RT
    return dict(
        r=r, V=V, A=A, sigma=sigma, dP=dP,
        dPi_ip=dPi_ip, Pi_n=Pi_n, dPi_total=dPi_ip + Pi_n,
        N=N, n_n=n_n, sigma_g=sigma_g,
    )


# --------------------------------------------------------------------------
# confinement protocol
# --------------------------------------------------------------------------
def sigma_g_of_t(t: float, confined: bool, p: Params) -> float:
    """Contact stress on the cell surface: 0 (free) or linear ramp to peak."""
    if not confined:
        return 0.0
    return p.sigma_g_peak * (t / p.t_end)


# --------------------------------------------------------------------------
# quasi-static time integration (slow variable n_n)
# --------------------------------------------------------------------------
def simulate(p: Params | None = None, confined: bool = False,
             n_points: int = 481) -> dict:
    """Integrate the cycle (0 -> t_end) in the quasi-static (fast-water) limit.

    Only n_n is dynamic (dn_n/dt = beta, capped at n_sat); V and N follow the
    algebraic balance. Returns time series of all observables.
    """
    if p is None:
        p = default_params()

    t = np.linspace(0.0, p.t_end, n_points)
    out = {k: np.zeros(n_points) for k in
           ("t", "V", "r", "dP", "Pi_n", "dPi_ip", "dPi_total",
            "sigma", "N", "n_n", "sigma_g")}
    out["t"] = t

    for i, ti in enumerate(t):
        n_n = min(p.n_n0 + p.beta * ti, p.n_sat)
        sg = sigma_g_of_t(ti, confined, p)
        s = solve_state(n_n, sg, p)
        for k in ("V", "r", "dP", "Pi_n", "dPi_ip", "dPi_total",
                  "sigma", "N", "n_n", "sigma_g"):
            out[k][i] = s[k]
    return out


# --------------------------------------------------------------------------
# full stiff-ODE integration (all three states) -- verification
# --------------------------------------------------------------------------
def simulate_full_ode(p: Params | None = None, confined: bool = False,
                      n_points: int = 481) -> dict:
    """Integrate the full 3-state ODE system (V, N, n_n) with a stiff solver.

    Demonstrates that explicitly resolving the fast water/ion dynamics
    reproduces the quasi-static trajectory. Uses the (rescaled) fast rate
    constants only to set the timescale separation.
    """
    if p is None:
        p = default_params()

    # initial state from the quasi-static balance at t=0
    s0 = solve_state(p.n_n0, 0.0, p)
    y0 = [s0["V"], s0["N"], p.n_n0]

    def rhs(t, y):
        V, N, n_n = y
        V = max(V, 1.0)
        r = radius_from_volume(V)
        A = surface_area(r)
        sigma = cortical_stress(r, p)
        sg = sigma_g_of_t(t, confined, p)
        dP = sg + 2.0 * p.h * sigma / r
        Pi_ip = N * p.RT / V
        Pi_n = n_n * p.RT / V
        dPi_ip = Pi_ip - p.Pi_ext
        dPi_total = Pi_n + dPi_ip
        dV = -A * p.Lp_m * (dP - dPi_total)
        dN = -A * ((omega_ms(sigma, p) + p.omega_l + p.gamma) * dPi_ip
                   - p.gamma * p.dPi_c)
        dn = p.beta if n_n < p.n_sat else 0.0
        return [dV, dN, dn]

    t_eval = np.linspace(0.0, p.t_end, n_points)
    sol = solve_ivp(rhs, (0.0, p.t_end), y0, method="BDF",
                    t_eval=t_eval, rtol=1e-8, atol=[1e-3, 1e-2, 1e-4])
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    V, N, n_n = sol.y
    r = radius_from_volume(V)
    sigma = np.array([cortical_stress(ri, p) for ri in r])
    sg = np.array([sigma_g_of_t(ti, confined, p) for ti in sol.t])
    dP = sg + 2.0 * p.h * sigma / r
    Pi_n = n_n * p.RT / V
    dPi_ip = N * p.RT / V - p.Pi_ext
    return dict(t=sol.t, V=V, r=r, dP=dP, Pi_n=Pi_n, dPi_ip=dPi_ip,
                dPi_total=Pi_n + dPi_ip, sigma=sigma, N=N, n_n=n_n, sigma_g=sg)


# --------------------------------------------------------------------------
# mitotic volume checkpoint
# --------------------------------------------------------------------------
def p_division(V, p: Params):
    """Probability of passing the mitotic sizing checkpoint (paper Methods).

    P_div(V) = lambda / (1 + exp(-alpha (V/V_div - 1)))
    """
    V = np.asarray(V, dtype=float)
    return p.lam / (1.0 + np.exp(-p.alpha * (V / p.V_div - 1.0)))


if __name__ == "__main__":
    p = default_params()
    free = simulate(p, confined=False)
    conf = simulate(p, confined=True)
    hpa = 3600.0
    print(f"{'time[h]':>7} | {'V_free':>8} {'dP_free':>8} {'Pin_free':>8}"
          f" | {'V_conf':>8} {'dP_conf':>8}")
    for i in range(0, len(free["t"]), len(free["t"]) // 8):
        print(f"{free['t'][i]/hpa:7.1f} | {free['V'][i]:8.0f} {free['dP'][i]:8.1f}"
              f" {free['Pi_n'][i]:8.1f} | {conf['V'][i]:8.0f} {conf['dP'][i]:8.1f}")
    print()
    print(f"Free  : V {free['V'][0]:.0f} -> {free['V'][-1]:.0f} um^3 "
          f"(x{free['V'][-1]/free['V'][0]:.2f}),  "
          f"dP {free['dP'][0]:.0f} -> {free['dP'][-1]:.0f} Pa")
    print(f"Conf. : V {conf['V'][0]:.0f} -> {conf['V'][-1]:.0f} um^3 "
          f"(x{conf['V'][-1]/conf['V'][0]:.2f}),  "
          f"dP {conf['dP'][0]:.0f} -> {conf['dP'][-1]:.0f} Pa")
    print(f"V_div = {p.V_div:.0f};  free reaches checkpoint: "
          f"{free['V'][-1] >= p.V_div};  confined reaches: {conf['V'][-1] >= p.V_div}")
