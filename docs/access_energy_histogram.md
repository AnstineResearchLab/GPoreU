# Accessible-point 2D energy/gradient histogram

*CLI method: `access_energy_histogram` (aliases `acchist`, `acc_ehist`) ·
steps: `grids → energy_grid → access_energy_hist`*

Companion to [energy_histogram.md](energy_histogram.md). One call,
same bin grid — but restricted to the space an actual probe can occupy,
with a gradient statistic that ignores directions blocked to the probe.

## What is reported

```
access CH4 2D hist   : accessible 128580/274625 points (46.8%,
                     E < 0.00 kJ/mol) -> (5, 14) bins; normalized
                     over accessible points, sum 1.0000
```

Pipeline keys: `access_energy_histogram` (int counts),
`access_energy_histogram_normalized`, `access_energy_accessible_points`.

## How it differs from the original histogram

Two changes, both aimed at describing only the *probe-relevant*
texture:

1. **Point selection.** The original bins every grid point of the cell
   (framework-interior included — its deep-repulsion points land in the
   E ≥ 0 columns as packing background). Here a point enters the
   histogram only if it is **accessible to the CH4 probe**, defined by
   its own interaction energy:

   ```
   accessible(g)  ⇔  U_CH4(g) < access_threshold   (default 0 kJ/mol)
   ```

   U < 0 means every framework atom is outside (or nearly outside) the
   repulsive core from that position — the probe center can sit there.
   The criterion is self-consistent with the energy model of the
   method (no extra probe mask needed: CH4 is its own occupancy test)
   and configurable via `energy.access_threshold`.

2. **Neighbour-gated gradient.** The original |∇U| column comes from
   the analytic pair-force sum. Analytic gradients are exact but they
   describe the mathematical field including directions the probe can
   never take: at a wall-facing point the gradient is dominated by the
   repulsion *into* the wall — information about solid the probe cannot
   enter, not about its landscape. This method instead estimates
   \|∇U\| by periodic **finite differences of the energy field over
   accessible grid neighbours only**:

   ```
   per lattice axis i:
     both neighbours accessible :  D_i = (U(g+h·ê_i) − U(g−h·ê_i)) / 2h
     exactly one accessible     :  D_i = one-sided difference with
                                   that neighbour  (…/ h)
     none accessible            :  D_i = 0        (axis drops out)

   |grad U| = | Σ_i D_i · reciprocal_cell_basis |     (triclinic-safe)
   ```

   The user-stated rule — *if a neighbour is inaccessible, it does not
   contribute to the gradient* — is implemented literally: the blocked
   direction is dropped from the difference rather than being replaced
   with a fictitious value. Physically: the gradient a roaming probe
   "feels" is the slope of the landscape it can actually sample; a
   difference across a wall it cannot cross measures two disconnected
   regions and belongs in neither descriptor.

The gradient is therefore a different (though correlated) statistic
from the original at near-wall points — deeper discussion below —
which is why this is a separate method, not a flag.

## Normalization

Counts are normalized by the number of **accessible** points, not the
whole cell: the matrix is the joint distribution of (U, |∇U|)
*conditioned on the probe being able to sit at the sampled point*.
This is the natural reading for an accessible-space descriptor; the
original
remains available for the whole-cell volume-fraction view (see
[energy_histogram.md](energy_histogram.md)). The summary reports the
accessible fraction so the two views can be combined downstream
(`access_hist × accessible_fraction` recovers a cell-normalized form).

## Accuracy of the finite-difference gradient

Central differences on a 0.4 Å lattice are second-order accurate where
the field is smooth: measured against the analytic gradients on
IRMOF-1 interior accessible points the median relative error is ~4 %
(worsening with local LJ curvature, as expected for O(h²) error on a
steep 1/r¹² field). This is a deliberate trade:

* deep-interior points (the physical content of the descriptor:
  binding-site texture) sit in smooth regions where the estimate is
  excellent;
* near-wall points suffer the O(h²) error, but there the *gating
  itself* is the point — the analytic value is dominated by forbidden
  directions anyway;
* the energy column is untouched (analytic field values, exact bin
  edges identical to the original histogram) so the two histograms'
  energy axes compare directly.

Bin edge constants (2 kJ/mol energy bins over −24…0 with open
extremes; 28 kJ mol⁻¹ Å⁻¹ gradient bins, 5 rows) are shared verbatim
with `histogram_2d_kernel` for cross-method comparability; both kernel
binnings must be kept in sync if this ever changes.

## Why defined this way

* **Energy-threshold accessibility instead of an N2/He mask**: the
  descriptor is built from the CH4 field and should answer for the CH4
  probe; the grid's hard occupancy masks are for other probes
  ([voxelization.md](voxelization.md)) and would mix conventions. The
  E < 0 test is the zero-crossing of the same pair potential the grid
  kernels use — a continuous, probe-consistent replacement for a hard
  sphere test, requiring no extra distance pass.
* **Periodic neighbours**: the field wraps, so neighbour access across
  cell faces is evaluated on the periodic image — a pore-mouth point
  at a face sees its true continuation on the other side. A
  neighbour blocked *across* a face is a real wall (in the replicated
  crystal), and correctly contributes nothing.
* **Opt-in, not replacing, the original**: the classic fingerprint is
  a literature-standard descriptor (comparability across structures
  and codes); this is a complementary, transport-flavoured view.
  DEFAULT_METHODS keeps the original; `acchist` is requested
  explicitly.
* **0-accessible is a valid result**: for a structure (or probe
  threshold) where no grid point is accessible the histogram is all
  zeros —
  the summary reports `0/… points` rather than failing; that itself
  is informative (e.g. CH4 sees no bindable space in WEI at the
  default grid, matching its impermeability).

## Limitations

* The finite-difference gradient has O(h²) error scaling with grid
  spacing near steep fields — do not read absolute gradient rows as
  analytic values; compare like-with-like (acchist vs acchist).
* Threshold semantics: `access_threshold = 0` bakes the "net
  non-repulsive" convention into the accessible set; raising it
  includes shallow-repulsion points (smoother gate, different
  texture) — record the setting when comparing structures.
* Everything static-framework and single-particle that limits the
  original ([energy_histogram.md](energy_histogram.md)).

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.energy.image_counts
gporeu.energy.calculate_energy_gradient     # the energy field consumed here
  gporeu.energy.energy_gradient_kernel
gporeu.energy.build_2d_energy_histogram_access
  gporeu.energy.histogram_2d_access_kernel    # gated FD gradient + (5,14) binning
gporeu.cli.format_summary
```
