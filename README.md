# Replication: single-cell mechano-osmotic growth and the volume checkpoint

A from-scratch reimplementation of the **single-cell hydromechanical growth
model** (Figure 1) of:

> I. Senthilkumar, J. Vangheel, V. Kumar, L. McNamara, B. Smeets, E. Howley,
> E. McEvoy, *"Stress-dependent growth in breast cancer arises from a
> mechano-osmotic coupling and cell-sizing checkpoint"*, **PNAS** 123(10):
> e2523159123 (2026). doi:[10.1073/pnas.2523159123](https://doi.org/10.1073/pnas.2523159123)
>
> Open-access preprint: *"Stress-dependent growth of breast cancer models
> arises from a cellular volume checkpoint"*, bioRxiv
> [2025.07.29.667388](https://doi.org/10.1101/2025.07.29.667388).

This repository reproduces the paper's central single-cell result: **cell-cycle
volume growth is driven by osmotic pressure from biomolecule synthesis, and
mechanical confinement disrupts the hydrostatic–osmotic balance, holding the
cell below a critical volume checkpoint for mitosis** (Fig. 1b–f).

![Figure 1 replication](figures/figure1_single_cell.png)

## Result

For a single HeLa cell over a 24 h G1→M cycle, the model reproduces the
paper's documented anchors:

| quantity | paper (Fig. 1) | this replication |
|---|---|---|
| free-cell volume increase | "almost two-fold" | **×1.95** |
| free-cell hydrostatic pressure | ~210 → ~340 Pa | **210 → 341 Pa** |
| dominant osmotic contribution | biomolecule synthesis Πₙ | **Πₙ ≫ ΔΠᵢ,ₚ** (340 vs 0.75 Pa) |
| confinement (σ_g → 250 Pa) | growth arrest below checkpoint | **×1.30, stays < V_div** |
| confinement effect on ΔP | raises hydrostatic pressure | **510 Pa vs 341 Pa** |

`python src/validate.py` checks all of these automatically (7/7 pass).

## The model

The cell is described during its cycle by three quantities: the amount of
impermeable biomolecules `n_n` (the slow growth driver), the cell volume `V`,
and the amount of permeable ions `N`.

**Osmotic pressures** (van't Hoff, dilute):

```
Πₙ      = n_n R T / V                     (impermeable biomolecules)
Πᵢ,ₚ    = N   R T / V                     (permeable ions, internal)
ΔΠᵢ,ₚ   = Πᵢ,ₚ − Π_ext                    (permeable-ion difference)
ΔΠ      = (n_n + N) R T / V − Π_ext = Πₙ + ΔΠᵢ,ₚ
```

**Water flux across the semi-permeable membrane** (paper Eq. 1):

```
dV/dt = − A · L_p,m · (ΔP − ΔΠ)
```

**Cortex mechanics** — force balance for a sphere of radius `r`
(McEvoy et al. 2020):

```
σ  = (K/2)(r²/r₀² − 1) + σ_a              (cortical stress: passive + active)
ΔP = σ_g + 2 h σ / r                       (intracellular hydrostatic pressure)
```
where `σ_g` is the external contact/confinement stress.

**Permeable-ion transport** — mechanosensitive + leak channels + active pump
(paper Eq. 2):

```
dN/dt = − A · [ (ω_ms(σ) + ω_l + γ) ΔΠᵢ,ₚ − γ ΔΠ_c ]
ω_ms(σ) = 0                         if σ ≤ σ_c
        = ω_ms (σ − σ_c)            if σ_c < σ < σ_s
        = ω_ms (σ_s − σ_c)          if σ ≥ σ_s
```

**Biomolecule synthesis** — the growth driver (saturating by macromolecular
crowding):

```
dn_n/dt = β     while n_n < n_sat,   else 0
```

**Mitotic sizing checkpoint** (paper Methods, after Varsano et al. 2017):

```
P_div(V) = λ / (1 + exp(−α (V/V_div − 1)))
```

### Why confinement arrests growth (the mechanism)

The permeable ions adjust quickly so that `ΔΠᵢ,ₚ` sits at its pump–leak steady
state (~1 Pa) — they osmotically balance the large external pressure `Π_ext`
(~0.67 MPa) almost exactly. Growth is therefore governed by the **small** (few
hundred Pa) balance between the biomolecule osmotic pressure `Πₙ`, the cortical
hydrostatic pressure, and any contact stress:

```
Πₙ(V) ≈ ΔP(V) − ΔΠᵢ,ₚ  =  σ_g + 2 h σ(r)/r − ΔΠᵢ,ₚ
```

Synthesis raises `n_n`, pulling water in and growing the cell; the cortex
stiffens as it stretches, raising `ΔP` (this is the observed 210→340 Pa rise).
Adding a confinement stress `σ_g ≈ 250 Pa` is **comparable to `Πₙ` itself**, so
it drives water efflux and holds the cell below the mitotic volume checkpoint —
even though 250 Pa is negligible against the 0.67 MPa osmotic background. This
is exactly the "competition between hydrostatic forces and osmotic pressure
from biomolecule synthesis" described in the paper.

### Numerics

Water and ion equilibration (seconds–minutes) are fast compared with synthesis
(hours), so the cell tracks a **quasi-static** balance `ΔP = ΔΠ` while only
`n_n` evolves slowly. `single_cell_model.simulate()` integrates `n_n(t)` and
solves the algebraic balance for `V` at each step. `simulate_full_ode()`
integrates the full stiff three-state ODE system and agrees with the
quasi-static reduction to ~1% (checked by `validate.py`).

## Parameters and provenance

The PNAS Supplementary Information (parameter tables) and the authors' code
were **not publicly available** at the time of writing (the preprint states
code/data will be released on GitLab at publication). Parameters are therefore
grounded as follows — see `src/parameters.py` for the per-value annotations:

* **Membrane/ion machinery and cortex mechanics** (`Π_ext`, `K`, `σ_a`, `h`,
  `ω_ms`, `ω_l`, `γ`, `ΔΠ_c`, `σ_c`, `σ_s`) are taken from **Supplementary
  Table 1 of McEvoy, Han, Guo & Shenoy (2020)**, *Nat. Commun.* 11:6148,
  doi:[10.1038/s41467-020-19904-5](https://doi.org/10.1038/s41467-020-19904-5)
  — the open-access predecessor model that the 2026 paper extends and cites.
* **Geometry (`r₀`), the synthesis parameters (`β`, `n_sat`, `n_n0`) and the
  confinement protocol** are **calibrated** to the single-HeLa-cell anchors the
  2026 paper reports (≈2-fold volume growth; ΔP ≈ 210→340 Pa; σ_g ramped to a
  250 Pa peak at 24 h). This mirrors the paper, which itself calibrates the
  growth curve to HeLa single-cell data (Cadart et al. 2022).

This is an honest, mechanism-faithful replication of the model and its central
result; absolute parameter values that depend on the unpublished SI are
calibrated to the paper's stated targets rather than copied.

## Usage

```bash
pip install -r requirements.txt
python src/single_cell_model.py   # prints the growth/pressure trajectory
python src/figure1.py             # writes figures/ (Fig. 1b–f + combined)
python src/validate.py            # checks the reproduction vs. paper anchors
```

## Scope

This replicates the **single-cell model (Fig. 1)** — the foundational,
self-contained result. The paper's later results (multicellular deformable-cell
"active foam" spheroids, the DNN-accelerated finite-element ECM solver, and the
T47D/4T1 experiments) build a large simulation stack on top of proprietary
frameworks (Abaqus, the KU Leuven deformable-cell code) and are out of scope
here. The single-cell mechano-osmotic coupling is the mechanism on which all of
those results rest.

## Repository layout

```
src/parameters.py          parameter set with per-value provenance
src/single_cell_model.py   the model (quasi-static + full-ODE integrators, checkpoint)
src/figure1.py             reproduces Figure 1b–f
src/validate.py            quantitative checks vs. the paper's anchors
figures/                   generated figures
```
