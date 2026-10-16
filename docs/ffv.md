# Fractional free volume (FFV)

*CLI method: `ffv` · steps: `grids → psd`*

## What is reported

IRMOF-1 (real run):

```
FFV (N2 probe)       :     0.7673     occupiable 13226.3 A^3 (1.293 cm^3/g)
```

* **FFV** — the fraction of probe points placed uniformly at random in
  the unit cell that fall inside a pore open to an N2-type probe.
* **Probe-occupiable volume** — `FFV × V_cell`, the corresponding
  volumetric quantity.

## Theory

Fractional free volume is the classic single-number summary of "how much
of this material is empty as far as an adsorptive is concerned", written
for polymers by Bondi and carried into the MOF literature (Liumai,
Kitagawa, Yaghi lines of work) where FFV correlates with permeability
and low-pressure uptake. Formally:

```
FFV = V_free / V_total
```

where "free" must be defined with respect to a probe: a point is free if
a test sphere of the probe size can *sit* there without overlapping (in
the LJ sense) the framework vdW envelope. In GPoreU the probe is the N2
parameter set (σ = 3.31 Å exclusion convention, see
[voxelization.md](voxelization.md)).

Two classical ways to evaluate the ratio exist:

1. **Voxel counting** — fraction of grid points with a free-radius
   value above a threshold. Fast, but biased: grid points sample space
   with weights that do not correspond to the geometric test
   ("sphere fits at this point") they are used to answer, and the
   boundary voxel classification flips with sub-voxel placement.
2. **Monte-Carlo insertion** — insert probe centers at i.i.d. uniform
   positions and count the success fraction. This is exactly the
   Widom-type zero-bias insertion measure of free volume, has an
   unambiguous continuum definition, unbiasedness from random sampling,
   and its statistical error is the trivial `√(FFV(1−FFV)/n)`.

GPoreU evaluates a variant of (2) with the sampling cost amortized onto
the GPU's strength (distance-to-nearest-fitting-sphere queries over a
pre-filtered candidate set).

## GPoreU procedure

Implemented in `gporeu.psd_ffv.GPoreU_psd_ffv` (FFV shares the code of
the PSD stage; see [pore_size_distribution.md](pore_size_distribution.md)):

1. Draw `N_rp = psd_cycles` (default 10 000) probe centers at uniform
   random fractional coordinates of the cell, converted to Cartesian
   (seeded — deterministic across runs for a given seed).
2. Build the **candidate set**: all grid points that are
   *not* N2-occupied (`~n_grid_occ`) and have positive free radius. A
   probe center can only be declared free if some *large enough* free
   sphere covers it; prefiltering to N2-accessible candidates with
   r > 0 makes the set a strict superset of all covering spheres
   (points whose exclusion zone swallows any probe point are already
   excluded), while shrinking the search by the solid fraction.
3. The fused 2D kernel (`GPoreU_PSD_Kernel`) computes, for every probe
   point `rp`, the **pore radius**

   ```
   pore_d(rp) = max{ r_g : g ∈ candidates,  d_min-image(rp, g) ≤ r_g }
   ```

   i.e. the radius of the largest framework-fitting sphere that covers
   the probe point. A point is free for the probe iff such a sphere
   exists: `pore_d(rp) > 0`. (Proof sketch of equivalence: if some
   placement of the probe at `rp` avoids all exclusion spheres, the
   "clearance sphere" centered at rp with radius = min wall distance is
   a fitting sphere, and by grid discretization one of the candidate
   grid spheres covers rp; conversely any covering free sphere contains
   a valid probe placement. The `sqrt`-then-compare form is kept
   bit-identical to the boundary of the reference definition.)
4. `FFV = mean(pore_d > 0)` over the sampled points; the
   occupiable volume is `FFV·V_cell`. **Note GPoreU's definition:** the
   sampling is over the *whole* cell, so FFV here is "fraction of cell
   volume occupiable", not "occupiable / void" — dividing by void volume
   inflates the metric in dense materials.

The same kernel run supplies the PSD's raw radius population (see its
document), so requesting FFV and PSD together costs one pass, and
selecting either runs the identical sampling.

## Why performed this way

* **MC insertion over voxel thresholds**: unbiased continuum definition,
  no grid-alignment artifacts, per-point pore size comes for free.
* **Candidate pre-filtering**: reduces the inner loop from every grid
  point (~10⁵–10⁶) to the genuinely relevant free subset; the
  compaction cost is negligible, and correctness holds because the
  filter only removes points that *cannot* cover any probe position.
* **`~n_grid_occ` as the "free" criterion**: FFV's purpose is
  transport-relevant emptiness for a real adsorptive, so the same
  exclusion shell that gates N2 uptake is used — consistency between
  FFV, PSD and the percolation picture matters more than any one
  convention.
* **Determinism by seeded draws** — FFV, a MC estimate, is otherwise
  irreproducible at the third decimal; with the seed it is exactly
  reproducible, which matters for regression checks and publication.
* **Shared kernel with PSD**: both quantities are functions of the same
  radius population; computing them together eliminates sampling
  disagreement between the two reported metrics.

## Limitations

* Statistical error decreases as 1/√N_rp (default 10 000 → ±0.5 %
  absolute in mid-range FFV) — raise `psd_cycles` for tighter bounds.
* FFV here is *not* percolation-gated: an internal cavity free to the
  probe counts. For transport-relevant free volume combine with the
  percolation flags, or read the gated volumes from `porosity`.
* "Free" reflects static framework LJ exclusion only — no framework
  dynamics, no quantum probe effects.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.psd_ffv.GPoreU_psd_ffv
  gporeu.psd_ffv.GPoreU_PSD_Kernel          # fused 2D random-point x candidate kernel
gporeu.cli.format_summary
```
