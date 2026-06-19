"""
Parameters for the single-cell mechano-osmotic growth model.

This module reproduces the single-cell ("hydromechanical") growth model of:

  Senthilkumar, Vangheel, Kumar, McNamara, Smeets, Howley & McEvoy (2026),
  "Stress-dependent growth in breast cancer arises from a mechano-osmotic
  coupling and cell-sizing checkpoint", PNAS 123(10):e2523159123.
  doi:10.1073/pnas.2523159123
  (open-access preprint: bioRxiv 2025.07.29.667388, doi:10.1101/2025.07.29.667388)

The single-cell model in that paper extends the chemo-/mechano-osmotic cell model
introduced by the same group:

  McEvoy, Han, Guo & Shenoy (2020), "Gap junctions amplify spatial variations in
  cell volume in proliferating tumor spheroids", Nat. Commun. 11:6148.
  doi:10.1038/s41467-020-19904-5   (CC-BY, open access)

Because the PNAS Supplementary Information (parameter tables) and the authors'
code were not publicly available at the time of writing, parameter values here
are grounded as follows:

  * Membrane / ion-transport machinery and cortex mechanics:
    taken directly from McEvoy et al. (2020), Supplementary Table 1
    (the predecessor model that the 2026 paper builds on and cites).

  * Geometry, biomolecule-synthesis rate (beta, n_sat) and confinement
    protocol: CALIBRATED to the quantitative anchors that the 2026 paper
    reports for single HeLa-cell growth, namely
        - cell volume increases ~2-fold from early G1 to the start of M phase
          (calibrated to HeLa single-cell growth, Cadart et al. 2022);
        - intracellular hydrostatic pressure rises from ~210 Pa to ~340 Pa
          over the cycle (consistent with Fischer-Friedrich et al. 2014);
        - a confining contact stress ramped linearly to a peak of 250 Pa at
          24 h suppresses growth below the mitotic volume checkpoint.

Every value below is annotated with its source ("McEvoy2020 Table S1" or
"calibrated"). See README.md for the full justification.

Unit system
-----------
length      : micrometres (um)
time        : seconds (s)
pressure    : pascals (Pa)
amount      : attomoles (amol = 1e-18 mol)

With these units the van't Hoff product R*T has a convenient value and
osmolyte amount / volume comes out directly in millimolar:
    R*T = 8.314 J/mol/K * 310 K = 2577.3 Pa*m^3/mol
        = 2577.3 Pa*um^3/amol     (since 1 m^3 = 1e18 um^3 and 1 mol = 1e18 amol)
    1 amol/um^3 = 1 mM
"""

from dataclasses import dataclass, field


# Physical constant ---------------------------------------------------------
R_GAS = 8.314        # J / (mol K)
T_ABS = 310.0        # K  (37 C)
RT = R_GAS * T_ABS * 1.0  # Pa*um^3/amol  -> 2577.3 (see module docstring)
# numerically: 8.314*310 = 2577.34 Pa*m^3/mol = 2577.34 Pa*um^3/amol


@dataclass
class Params:
    """Single-cell mechano-osmotic growth parameters."""

    # --- thermodynamics -----------------------------------------------------
    RT: float = RT                      # Pa*um^3/amol
    Pi_ext: float = 0.67e6              # external osmotic pressure [Pa]
    #   McEvoy2020 Table S1: 0.67 MPa, from physiological ion concentrations
    #   (c_Na=145, c_K=5, c_Cl=110 mM) -> Pi_ext = R T (sum c) = 0.67 MPa.

    # --- cortex / membrane mechanics ---------------------------------------
    K: float = 2.485e3                 # effective cortical stiffness [Pa]
    #   McEvoy2020 reports 6 kPa (within the very broad reported 0.1-100 kPa
    #   range). Here calibrated to 2.49 kPa so that a ~2x volume increase maps
    #   onto the reported 210 -> 340 Pa hydrostatic-pressure rise for HeLa.
    sigma_a: float = 100.0             # active cortical (myosin) stress [Pa]
    #   McEvoy2020 Table S1 (after Jiang & Sun 2013).
    h: float = 0.6                     # cortex + membrane thickness [um]
    #   McEvoy2020 Table S1 (upper end of 0.1-0.6 um epithelial range).
    r0: float = 5.50                   # reference (unstretched) cell radius [um]
    #   Calibrated. The cell is always stretched above r0; chosen together with
    #   K so that V ~ 2000 um^3 at 210 Pa and V ~ 3900 um^3 at 340 Pa.

    # --- membrane water permeability ---------------------------------------
    Lp_m: float = 7.0e-6               # membrane water permeability [um/s/Pa]
    #   McEvoy2020 Table S1 lists 7e-12 for the *minutes-scale* loading
    #   response. Over the 24 h growth cycle the cell is in the quasi-static
    #   (fast-water) limit, so the absolute value is immaterial as long as
    #   water equilibration (seconds-minutes) is fast vs. synthesis (hours);
    #   value here only sets that separation of timescales for the full-ODE
    #   integrator. The quasi-static solver does not use it.

    # --- permeable-ion transport (mechanosensitive / leak / pump) ----------
    # All in attomole units (McEvoy2020 mol values * 1e18).
    omega_ms: float = 2.0e7            # MS-channel permeability slope
    #   [amol um^-2 s^-1 Pa^-2]  (McEvoy2020: 2e-11 mol ...)
    omega_l: float = 1.5e9             # leak-channel permeability
    #   [amol um^-2 s^-1 Pa^-1]  (McEvoy2020: omega_l = omega_ms*sigma_c)
    gamma: float = 2.25e5              # active-pump rate constant
    #   [amol um^-2 s^-1 Pa^-1]  (McEvoy2020: 2.25e-13 mol ...)
    dPi_c: float = 4.0e4               # critical osmotic diff. for pump [Pa]
    #   McEvoy2020 Table S1: 40 kPa (= Pi_ext * dG_ATP / RT).
    sigma_c: float = 75.0              # MS-channel threshold stress [Pa]
    sigma_s: float = 600.0            # MS-channel saturation stress [Pa]
    #   McEvoy2020 Table S1.

    # --- biomolecule synthesis (the growth driver) -------------------------
    beta: float = 4.07e-3             # synthesis rate of impermeable
    #   biomolecules [amol/s]. Calibrated so the free cell reaches the mitotic
    #   volume (~2x) at ~24 h.
    n_sat: float = 525.0              # synthesis saturation amount [amol]
    #   Calibrated (macromolecular-crowding cap, Alric et al. 2022). Limits the
    #   confined cell so its volume plateaus below the checkpoint.
    n_n0: float = 162.0              # initial impermeable-biomolecule amount [amol]
    #   Calibrated to set V(0) ~ 2000 um^3 at 210 Pa.

    # --- mitotic volume checkpoint -----------------------------------------
    V_div: float = 3600.0            # critical mitotic volume [um^3]
    #   Sizing-checkpoint threshold for mitotic entry. Calibrated just below the
    #   freely growing HeLa cell's M-phase volume (~3900 um^3, ~2x G1) so that
    #   the free cell surpasses it (divides) while the confined cell does not.
    lam: float = 1.0                 # proliferative capacity lambda (max prob.)
    alpha: float = 12.0              # checkpoint steepness (stochasticity)
    #   Functional form P_div(V) = lam / (1 + exp(-alpha (V/V_div - 1)))
    #   from the paper's Methods (Varsano et al. 2017 sizing checkpoint).

    # --- simulation / confinement protocol ---------------------------------
    t_end: float = 24 * 3600.0       # cell cycle duration [s] (24 h)
    sigma_g_peak: float = 250.0      # peak confining contact stress [Pa]
    #   Paper Methods: time-dependent uniform surface pressure ramped to a peak
    #   of 250 Pa at 24 h for the confined single-cell case.


def default_params() -> Params:
    return Params()


def scaled_params(p: "Params", s: float) -> "Params":
    """Return a geometrically down-scaled cell (linear scale factor s).

    Lengths scale by s and osmolyte amounts by s**3, which leaves every *pressure*
    (osmotic Pi=nRT/V, cortical sigma, hydrostatic dP=sigma_g+2h*sigma/r) and every
    ratio (V/V_div) invariant. So the cell behaves identically -- same fold-growth,
    same confinement-arrest threshold -- but is physically smaller. Used only by
    the illustrative multicellular models so a spheroid can be built from many
    small cells (smooth sphere) without touching the validated single-cell
    parameters.
    """
    from dataclasses import replace
    return replace(
        p,
        r0=s * p.r0,
        h=s * p.h,
        n_n0=s ** 3 * p.n_n0,
        n_sat=s ** 3 * p.n_sat,
        V_div=s ** 3 * p.V_div,
        beta=s ** 3 * p.beta,
    )
