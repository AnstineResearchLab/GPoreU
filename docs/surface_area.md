# Accessible surface area (ASA / SASA)

*CLI method: `surface_area` (alias `sa`) · steps: `surface_area`
(operates on the prepared structure alone)*

## What is reported

IRMOF-1 (real run):

```
surface area       :     3578.4 A^2/cell    3498.89 m^2/g
```

* Geometric surface area of the *probe-accessible* molecular surface,
  per unit cell (Å²) and gravimetric (m²/g via
  `SA / (mass_cell[u] · 1.660539) · 10⁴`).

## Theory

### Molecular surface and probe-radius surfaces

For a rigid framework of van-der-Waals spheres, the surface a probe
"feels" is not the vdW surface: it is a **probe-radius surface** — the
locus of the probe *center* at contact. Classical definition
(Connolly; Richards):

* **van-der-Waals surface / SASA**: union of atom spheres; a probe of
  rolling radius r_p traces an *external* surface at distance r_p from
  the vdW envelope (solvent-accessible surface, SASA).
* A point of the SASA generated around atom *i* is **buried** if a
  neighbouring atom *j* comes within `r_p' = distance-at-which-j-shadows`
  — in a hard-sphere model, if |x − atom_j| < r_p + (σ_j/2).

For adsorption science the relevant probe here is the adsorptive. GPoreU
standardizes on the N2 parameter set and constructs the probe surface at

```
R_i = 1.122 · σ_mix(i, N2) = 1.122 · ½(σ_i + σ_N2)
```

The factor **1.122 = 2^(1/6)** is the Lennard-Jones *minimum* position:
placing the probe-center sphere at 2^(1/6)σ_mix is where the attractive
well bottoms — the distance at which an adsorbed molecule actually sits
against the surface, not the contact distance where the repulsion
starts. Sampling at the LJ-minimum radius systematically lowers the
surface toward the physically occupied shell and is the same convention
used by established surface-area tools derived from this formulation.

An area element of i's sphere is counted **accessible** when its probe
center is not inside any other atom j's shadow sphere

```
d(element_center, atom_j) ≥ 1.122 · ½(σ_j + σ_N2)
```

(again the LJ-minimum separation with respect to *j*). Note GPoreU's
shadow is evaluated against the same 1.122-scaled sphere about each
blocking atom — the "hard sphere at potential minimum" model throughout.

### Monte-Carlo area evaluation

Exact analytic SASA requires inclusion-exclusion over intersecting
spherical patches (quadratic pairwise geometry, numerically delicate and
awkward under periodicity). The classic Monte-Carlo alternative: the
accessible *area* of a sphere equals its full area times the fraction
of uniform surface samples that are unblocked. With S samples per atom,
per-atom variance is bounded (`p(1−p)/S` for the fraction); areas of
10⁵ Å²-scale cells print to 5-6 significant digits — the standard
Monte-Carlo SASA accuracy.

## GPoreU procedure

`gporeu.surface_area.GPoreU_SA`:

1. Per framework atom i, S = **512** surface samples (the pipeline
   `trials = 512·N_atoms` total draws, seeded). Uniform sphere sampling:
   θ_i, φ_i uniform → points
   `x_i = R_i · (sinθ cosφ, sinθ sinφ, cosθ)` on the i-th probe surface.
2. A sample is tested against all atoms j (periodic minimum image):
   blocked iff `d_minimage(sample, j) < 1.122 · ½(σ_j + σ_N2)` and
   `j ≠ i` (the host atom never masks its own surface).
3. Per atom: `accessible_area_i = 4π R_i² · (unblocked/S)`;
   sum → cell total.
4. Implementation: one Triton program per (atom, sample-chunk) on a 2D
   launch grid, atoms streamed in 128-wide tiles with early exit per
   sample row once blocked; per-atom unblocked counts accumulate with
   integer atomic adds, so the counts are exact integers — the MC
   result is a deterministic function of the seeded draw stream.
5. Periodicity enters by evaluating sample-to-atom distances minimum-
   image on the fractional–Cartesian transform (the sample itself sits
   untransformed; its *distance* to each atom wraps — the correct
   treatment for short-ranged (σ-scale) shadow interactions).

## Why performed this way

* **Sampling the LJ-minimum shell** rather than the contact sphere:
  aligns the computed surface with where adsorbed N2 actually resides,
  reducing the systematic overestimate of naive vdW-offset surfaces;
  this is the convention that reproduces the experimentally-comparable
  surface areas of the reference tools.
* **MC rather than analytic patches**: robust under any cell shape,
  needs no surface intersection bookkeeping, GPU-parallel across both
  dimensions (atoms × samples), and with seeded draws its reproducibility
  matches analytic methods while its accuracy (per-atom p(1−p)/512) is
  more than sufficient for the reporting precision used.
* **The j-shadow uses j's 1.122-sphere, hard cutoff**: matches the
  same model the accessible-volume masks use (probe identity
  consistency across methods); softer energetic shadow functions (Boltzmann
  masks) blur the surface definition without measurable benefit at these
  radii, and were considered and rejected as non-standard.
* **No connectivity gating of the surface**: surface area, unlike
  uptake, is a *local* geometric measure; inaccessible internal cages
  genuinely have area that contributes to e.g. powder packing
  diagnostics. If transport-relevant area is wanted, intersect
  `pore_labels`/`geo_access` masks with the accessible-sample geometry
  downstream — deliberately left to the consumer to avoid conflating
  conventions.

## Limitations

* Probe = N2 only; probe-size dependence of "surface area" (CO2 vs N2
  vs Kr) is not swept — re-run by editing the `n_sigma`/`n_epsilon`
  probe fields before the step if needed.
* Statistical noise ~ √(p(1−p)/512) per atom, sub-0.1% at cell scale,
  but individual narrow-pocket atoms can be off; do not interpret
  single-atom areas.
* Static framework: real gate-opening and adsorbate-induced surface
  restructuring dominate for flexible structures.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.surface_area.GPoreU_SA
  gporeu.surface_area.GPoreU_SA_Kernel      # Triton 2D (atom, sample-chunk) grid
gporeu.cli.format_summary
```
