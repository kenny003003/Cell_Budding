"""
Quantitative validation of the single-cell replication against the anchors
reported in Senthilkumar et al. (PNAS 2026) for HeLa single-cell growth.

Checks (free / unconfined cell over the 24 h G1->M cycle):
  1. volume increases ~2-fold ("almost two-fold from early G1 to start of M")
  2. intracellular hydrostatic pressure rises ~210 -> ~340 Pa
  3. biomolecule osmotic pressure Pi_n is the dominant osmotic contribution
And (confined cell, sigma_g ramped to 250 Pa at 24 h):
  4. growth is suppressed and the cell stays below the mitotic checkpoint
  5. confinement raises intracellular hydrostatic pressure
Also checks that the full stiff-ODE integration agrees with the quasi-static
reduction.
"""

import numpy as np
from parameters import default_params
import single_cell_model as scm


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    p = default_params()
    free = scm.simulate(p, confined=False)
    conf = scm.simulate(p, confined=True)
    fo = scm.simulate_full_ode(p, confined=False)

    print("Validation against paper anchors (Fig. 1, single HeLa cell)\n")
    ok = []

    fold = free["V"][-1] / free["V"][0]
    ok.append(_check("free volume ~2-fold", 1.8 <= fold <= 2.15,
                     f"x{fold:.2f}  (target ~2.0)"))

    p0, p1 = free["dP"][0], free["dP"][-1]
    ok.append(_check("free hydrostatic pressure 210->340 Pa",
                     abs(p0 - 210) < 15 and abs(p1 - 340) < 15,
                     f"{p0:.0f} -> {p1:.0f} Pa  (target 210 -> 340)"))

    dom = np.all(free["Pi_n"] > 5 * np.abs(free["dPi_ip"]))
    ok.append(_check("biomolecule osmotic pressure dominates",
                     dom, f"Pi_n end={free['Pi_n'][-1]:.0f} Pa, "
                          f"|dPi_ip| end={abs(free['dPi_ip'][-1]):.2f} Pa"))

    conf_fold = conf["V"][-1] / conf["V"][0]
    suppressed = conf["V"][-1] < free["V"][-1] and conf["V"][-1] < p.V_div
    ok.append(_check("confinement suppresses growth below checkpoint",
                     suppressed,
                     f"confined x{conf_fold:.2f} -> {conf['V'][-1]:.0f} um^3 "
                     f"< V_div={p.V_div:.0f}"))

    ok.append(_check("confinement raises hydrostatic pressure",
                     conf["dP"][-1] > free["dP"][-1],
                     f"confined {conf['dP'][-1]:.0f} Pa > free {free['dP'][-1]:.0f} Pa"))

    pdiv_free = float(scm.p_division(free["V"][-1], p))
    pdiv_conf = float(scm.p_division(conf["V"][-1], p))
    ok.append(_check("checkpoint: free divides, confined arrested",
                     pdiv_free > 0.5 > pdiv_conf,
                     f"P_div free={pdiv_free:.2f}, confined={pdiv_conf:.2f}"))

    rel = np.max(np.abs(free["V"] - fo["V"]) / free["V"])
    ok.append(_check("full ODE agrees with quasi-static reduction",
                     rel < 0.02, f"max rel. diff in V = {rel:.2%}"))

    print(f"\n{sum(ok)}/{len(ok)} checks passed")
    return all(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
