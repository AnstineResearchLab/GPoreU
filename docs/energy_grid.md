# Interaction-energy and gradient grids (CH4 probe)

*CLI method: `energy_grid` (alias `energy`) · steps: `grids →
energy_grid`*

## What is reported

A per-grid-point field (available in the pipeline dict and in
`NAME.grid` dumps):

| key | meaning |
|---|---|
| `ch4_energy_kJ_mol` | single-particle interaction energy U(x) of a CH4 probe at each grid point, kJ/mol |
| `ch4_gradient_kJ_mol_A` | magnitude \|∇U\| at each grid point, kJ mol⁻¹ Å⁻¹ |

Optionally the 2D histogram of the pair
([energy_histogram.md](energy_histogram.md)).

## Theory

### The rigid-adsorbate, site-energy model

The energetic landscape a guest molecule feels inside a rigid
framework, at the level of accuracy that matters for screening, is a
pairwise additive **Lennard-Jones 12-6** potential summed over
framework atoms, plus the lattice's periodic images:

```
U(x) = Σ over images, atoms i   4 ε_mix(i) [ (σ_mix(i)/r)¹² − (σ_mix(i)/r)⁶ ]
```

with Lorentz–Berthelot combination rules
`σ_mix = ½(σ_i + σ_CH4)`, `ε_mix = √(ε_i · ε_CH4)` and default
σ_CH4 = 3.73 Å, ε_CH4 = 148 K (the standard methane parameters used
with UFF in the MOF screening literature). Electrostatics are omitted
by design: they require partial charges (not always present/consistent
in CIFs) and the LJ part dominates the *shape* of the landscape in
non-ionic frameworks; the CLI documents this as a known scope choice.

Why this model is the right tool here:

* The **energy distribution** over the cell — not just its minimum —
  is the quantity that determines low-pressure uptake (Henry regime)
  and the Isosteric heat trend; a grid of U(x) is the input to those
  integrals (`⟨exp(−U/kT)⟩` is exactly the Henry constant's
  kernel). See [energy_histogram.md](energy_histogram.md).
* The **gradient magnitude** \|∇U\| measures how confined/pinned a
  molecule is at each site (steep walls) versus sitting in a flat
  well — the discriminant of the classic 2D (U, \|∇U\|) fingerprint
  used to separate ultra-microporous, microporous and mesoporous
  behaviour without assuming any pore shape.
* Single-particle (zero-loading) limit: no adsorbate–adsorbate terms;
  this is the definition of the *potential* landscape, valid at the
  low pressures where pore-filling mechanisms are structure-diagnostic.

### Truncation and the impulse correction

Evaluating U requires a cutoff — the sum over periodic images is only
absolutely convergent with the attractive tail removed (rcut = 12 Å
default; beyond it LJ's r⁻⁶ tail contributes < 0.01 kJ/mol in these
densities but the image lattice otherwise diverges). Truncation,
however, creates two artifacts:

1. an **energy step** at r = rcut: U jumps from U(rcut) ≈ −ε-scale
   value to 0 (if just zeroed) — the shift correction `−U(rcut)` fixes
   the *value* but then
2. the **force** still jumps: dU/dr(rcut) ≠ 0, so the gradient field
   has surface-delta errors exactly on the cutoff sphere, polluting
   the \|∇U\) statistics that the fingerprint relies on.

GPoreU uses the standard *switching/impulse* resolution over a thin
shell of half-width δ = 0.005 Å around rcut:

* `r < rcut − δ`: full LJ energy; analytic LJ force
  (factor `24ε(2s¹² − s⁶)/r²` along the pair vector, summed to the
  gradient);
* `rcut − δ ≤ r ≤ rcut`: LJ energy retained; force gets a constant
  additive offset of magnitude `|U(rcut)| / 2δ` directed so as to
  cancel the jump;
* `rcut < r ≤ rcut + δ`: energy 0; the same constant offset force;
* beyond: nothing.

The offset is the delta-function "impulse" that would be the gradient
of the energy step, smeared uniformly over a shell of width 2δ. Its
line integral across the shell equals exactly the truncation energy
`U(rcut)`, so energy and force become consistent in the
distributional sense and the \|∇U\| field is free of a spurious ridge
at the cutoff — a necessity because the histogram stage counts gradient
bin memberships directly. (For energies the shell's contribution is
O(δ), negligible; the correction exists for the gradient.)

### Exact image enumeration vs minimum image

Under periodicity the true sum is over **all lattice translations n**
of every atom. The common shortcut — keep only the *minimum image*
per pair — is exact only when the cutoff sphere around a grid point
contains no pair twice, i.e. when every perpendicular cell width
`w_i = V / |a_j × a_k|` exceeds **2·rcut**. Many real MOF cells are
*narrower than 24 Å along at least one axis*; there, the minimum-image
truncated sum silently drops pairs that are legitimately inside the
cutoff via a non-minimal translation, and the error is both structure-
dependent and discontinuous at the "just above/below 24 Å" boundary.

GPoreU therefore computes, per axis, the exact needed repetition

```
K_i = floor( (rcut + δ) / w_i )
```

(any image with nonzero contribution must have lattice-offset ≤ that
bound because displacements perpendicular to axis i are bounded by
w_i... formally |n_i|·w_i ≤ |n·A| ≤ rcut+δ), and the kernel enumerates
the `(2K_x+1)(2K_y+1)(2K_z+1)` translations around each pair's minimum
image, activating only those images with r ≤ rcut + δ. For a wide cell
`K = (0,0,0)` and the enumeration collapses to the plain minimum-image
sum at zero extra cost — which is why the rigorous route costs nothing
in the common case. (Starting the enumeration from the minimum image is
exact: minimum-image + all integer lattice vectors spans every image
of the atom exactly once.)

### Units

Internally ε are temperature units (K; UFF kcal/mol ÷ 1.987e-3),
so the kernel returns U in K and \|∇U\| in K/Å; the wrapper converts
by `k_B = 0.008314462618 kJ mol⁻¹ K⁻¹`, giving kJ/mol and
kJ mol⁻¹ Å⁻¹.

## GPoreU procedure

`gporeu.energy.calculate_energy_gradient`, after `grids`:

1. `image_counts(cell, rcut, delta)` → `(Kx, Ky, Kz)` as above
   (computed once, on CPU, double precision).
2. One Triton launch: 2D tiled loops — grid-point blocks
   (BLOCK_GRID) × atom blocks (BLOCK_ATOMS). Per pair:
   minimum-image displacement via the fractional wrap, then the image
   enumeration loop; per-image mask `r ≤ rcut + δ`;
   Lorentz–Berthelot σ/ε; region masks (normal / inner shell / outer
   shell); LJ energy and analytic gradient plus impulse term; atom-sum
   accumulated in registers (no atomics — each grid point owned by one
   program), gradient components stored as their norm.
3. K → kJ/mol conversion; the two arrays enter the pipeline dict, and
   the summary reports the energy statistics over the geometrically
   open grid points:

   ```
   CH4 energy grid    : accessible min -13.44 / mean 2112.60 /
                        max 231891.98 kJ/mol (rcut 12.0 A)
   ```

Everything is deterministic float32 arithmetic in fixed tile order —
no atomics on the energy path — so the fields are bit-reproducible.

## Why performed this way

* **CH4 default**: methane storage is the flagship application of the
  energy fingerprint and CH4 parameters are the most-transferable
  (well-characterized LJ pair); any probe is substitutable through the
  `energy` config block.
* **Exact image enumeration** rather than minimum-image: cells under
  24 Å across are the *rule* in MOFs, and a truncated-lattice error of
  a few kJ/mol distorts precisely the strong-binding tail the summary
  counts. The correctness is free when unnecessary (K=0 collapse).
* **Impulse-corrected gradient** instead of force-shifted LJ: force
  shifting (`U − U(rc) − (r−rc)U'(rc)`) changes energies measurably in
  the shell region; the impulse form leaves the energy definition
  untouched (truncated at rcut) and repairs only the gradient, which
  is exactly where the artifact mattered.
* **Analytic gradient** rather than finite differences: no Δx step
  choice, no cancellation noise, and one fused kernel computes both
  fields, halving the cost for the fingerprint stage.
* **Grid evaluation at the voxel lattice** (not random insertion MC):
  the same points already encode geometry for every other method, so
  the energy fields overlay the porosity/PLD maps voxel-for-voxel —
  essential for post-hoc joint analysis (e.g. " energies inside the
  percolating He-accessible region").

## Limitations

* No electrostatics/indentation/3-body terms; for charged frameworks
  or CO2 in open-metal sites the absolute energies are LJ-only
  estimates (comparisons across similarly-charged families remain
  meaningful).
* rcut 12 Å: tail contributions above it are analytically small
  (< 0.01 kJ/mol) but nonzero by convention; documented, consistent
  across structures.
* Float32 evaluation: energies agree with float64 references within
  ~1e-4 relative on the median (verified), adequate for binning and
  screening; the strong-overlap region (unphysical, r ≪ σ) saturates
  rather than resolving — irrelevant for adsorption.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md;
                                              epsilons in K, probe params)
gporeu.grid.all_grids_pbc                   # produces the evaluation lattice
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.energy.image_counts                  # exact per-axis image ranges
gporeu.energy.calculate_energy_gradient
  gporeu.energy.energy_gradient_kernel      # fused energy + |grad|, K units
  (K -> kJ/mol scaling by kB)
gporeu.cli.format_summary
```
