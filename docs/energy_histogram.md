# 2D energy/gradient histogram (the "energy fingerprint")

*CLI method: `energy_histogram` (alias `ehist`) · steps: `grids →
energy_grid → energy_hist`*

## What is reported

Summary line (real run, IRMOF-1):

```
CH4 2D histogram   : 274625 grid points -> (5, 14) (E x |grad E|) bins;
                     normalized sum 1.0000
```

and the full 5 x 14 matrix in `energy_histogram` /
`energy_histogram_normalized` (normalized fractions): rows step the
gradient magnitude from 0 in 28 kJ mol⁻¹ Å⁻¹ intervals; columns step
the interaction energy from strong binding to unbound.

## Theory

### Why a fingerprint of (U, \|∇U\|)

Single numbers — surface area, pore volume, PLD — classify pores by
*shape extremes*. The pair (U(x), \|∇U(x)\|) at every point of the
void characterizes the *energetic texture* of the whole cell:

* **U(x)** — how strongly a molecule is bound at x. Deep negative
  bins come from overlapping potentials of closely-spaced walls
  (ultramicropores, interstitial sites in dense packing, open adsorption
  sites); bins near zero describe broad-pore interiors.
* **\|∇U(x)\|** — how steeply the binding changes at x: freely
  rotating/ translating guests in cavity centers have small
  gradients; wall-sliding states and window transits are steep.

The joint distribution thus separates material families that share any
one of SSA/PV/PLD: it is the grid-domain analogue of the experimental
"energy histogram" extracted from isosteric-heat data, and computing it
directly from structure is exactly what makes structure–adsorption
structure-function regressions (Block–Weymandt–Snurr-style joint
distribution fingerprints) possible at DB scale. Because the bins are
*fractions of cell volume*, the histogram is a normalized
volume-weighted descriptor: invariant to cell size, additive over
structure families, and machine-learning friendly.

In the dilute (Henry) limit the same fields give uptake analytically,

```
N(p)/p ∝ (1/V) ∫_cell exp( −U(x)/k_B T ) dx
```

i.e. the volume-weighted exp(-U/kT) average; the energy-binned marginal
of the histogram is the discretized integrand, so the histogram
doubles as an isotherm-shape predictor (micropore filling ≈ weight in
deep bins at their thermal factors). It is reported in kJ/mol to make
those Boltzmann factors directly evaluable.

### Bin definitions (constants of the method)

The grid domain is partitioned once, with fixed edges so histograms of
different structures are directly comparable:

* **Energy** (columns, 14): two open-ended extreme bins plus 12 bins
  of 2 kJ/mol over [−24, 0):
  `E < −24`, `−24…−22`, …, `−2…0`, `E ≥ 0`.
  (−24 kJ/mol ≈ 30·k_B T catches the strongest physisorption sites
  encountered in CH4 physisorption; ≥0 sites are unbound background.)
* **Gradient** (rows, 5): 28 kJ mol⁻¹ Å⁻¹ wide:
  `[0,28) [28,56) [56,84) [84,112) [112,∞)`.
  28 ≈ the slope scale of one LJ well over ~half a Å, so rows separate
  "flat well", "wall", and "hard overlap" texture.

*Normalization*: by the total grid-point count including framework-
interior points, so the matrix sums to exactly 1 over the cell — a
volume-fraction descriptor by construction (structure families then
compare as *textures*, not raw counts).

## GPoreU procedure

`gporeu.energy.build_2d_energy_histogram`, consuming the kJ/mol fields
from `calculate_energy_gradient`:

1. One Triton pass (`histogram_2d_kernel`) over all grid points:
   per-point bin index computed with the integer-safe form
   `col = clamp( floor((E + 24)/2) + 1, 0, 13 )` plus the explicit
   `E < −24 → 0`, `E ≥ 0 → 13` edges;
   `row = min( floor(|∇U|/28), 4 )`;
2. Flat `row·14 + col` atomic-adds into a 70-slot count buffer
   (fixed bin arithmetic means no search/sort — the histogram is a
   3-instruction per point mapping);
3. Reshape to (5, 14); return raw counts and counts/n_grid (all grid
   points counted, framework interior included).

The kernel runs in blocks over the flat grid with masks; atomics make
the result independent of scheduling order (integer additions are
associative), so the histogram is bit-identical across runs.

## Why performed this way

* **Fixed, published bin edges** rather than data-adaptive binning:
  the descriptor's entire purpose is cross-material comparability;
  percentiles or bin-width-by-range would make two structures' "same"
  bin mean different energies, destroying the regressions built on it.
* **Volume-fraction normalization over the whole cell** (not over void
  points only): keeps the framework-density signal — two materials of
  equal void texture but different packing density differ in
  gravimetric adsorption, and the histogram with cell-wide
  normalization encodes that directly. Users who want void-only
  textures can renormalize downstream (`/ (1 − framework fraction)`).
* **Energy from an exact-image, impulse-corrected field**: deep-bin
  counts concentrate in regions where walls are close — precisely
  where truncation/image errors would otherwise accumulate — and the
  gradient rows are populated by the \|∇U\| whose cutoff artifact was
  removed; the histogram is only as honest as the underlying field
  (see [energy_grid.md](energy_grid.md)).
* **Atomic flat-count kernel**: no per-block partial matrices to
  reduce (70 slots total — contention is negligible against memory
  traffic), one launch, order-independent.
* **Reported in kJ/mol**: keeps the Boltzmann factors
  `exp(−E/RT)` one mental step away; the K-domain intermediate stays
  inside the energy stage.

## Limitations

* Static field: temperature enters only via post-processing bin
  weights; quantum guest delocalization (real for H2, CH4 @ 77 K in
  tight pores) shifts deep-bin physics and is not modeled — the bins
  remain a classical classifier.
* No electrostatics (inherited from the LJ energy model) — for highly
  polar frameworks the energy axes miss the dominant site
  heterogeneity.
* Grid-resolution dependence: a 1 voxel wide ultramicropore may fall
  entirely in a deep bin or straddle two; for fingerprint regressions
  the *same* grid spacing must be used across the dataset
  (GPoreU's default 0.4 Å is the reference setting).

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc                   # evaluation lattice
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.energy.image_counts
gporeu.energy.calculate_energy_gradient     # kJ/mol E and |grad| fields
  gporeu.energy.energy_gradient_kernel
gporeu.energy.build_2d_energy_histogram
  gporeu.energy.histogram_2d_kernel         # (5,14) atomic binning
gporeu.cli.format_summary
```
