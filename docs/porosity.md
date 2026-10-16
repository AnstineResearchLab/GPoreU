# Porosity: accessible pore volumes

*CLI method: `porosity` · steps: `grids → connectivity`*

## What is reported

IRMOF-1 (real run):

```
pore volume (He, probe 2.58 A) :  14231.62 A^3   1.392 cm^3/g
pore volume (geometric)        :  13794.83 A^3   1.349 cm^3/g
```

ZEOLITE WEI (the contrast case: percolating geometric void, almost no
helium-accessible volume):

```
pore volume (He, probe 2.58 A) :     76.60 A^3    0.038 cm^3/g
pore volume (geometric)        :    435.77 A^3    0.215 cm^3/g
```

* **Helium-accessible pore volume** — the Boltzmann-weighted free volume
  of all lattice regions that *percolate* through the crystal for a
  helium-sized probe.
* **Geometric pore volume** — the voxel-count free volume of the
  percolating geometric (point-probe) void.
* Both are also given gravimetrically: `cm^3/g = V[A^3] / (mass_cell[u]
  × 1.660539)`, the unit-cell mass from the atom table.

## Theory

Pore volume definitions differ by what "accessible" means, and the
differences are the reason GPoreU reports several volumes at once:

1. **Textbook probe-accessible volume.** A position is available to a
   probe of radius r_p when its center stays outside the framework's
   exclusion shells. The *volume a probe center can explore inside
   channels connected to infinity* is the operational quantity that
   correlates with low-pressure gas uptake plateaus. Helium
   (kinetic radius ~1.2–1.4 Å, σ = 2.58 Å) at 4 K is the classical
   probe because it accesses essentially every aperture open at all.
2. **Geometric void volume.** Counting the cell volume not excluded by
   the atom-centered spheres themselves (no finite probe) — a property
   of the framework alone; it exceeds any probe-based volume and is the
   "void" of crystallographic convention.
3. **Connectivity gating.** Both numbers only become "accessible pore
   volume" once a region has a path to the outside. Isolated cavities
   are thermodynamically irrelevant to flow/uptake on experimental
   timescales; percolation gating (see [percolation.md](percolation.md))
   removes them.

The thermal (Boltzmann) weighting follows the discussion in
[voxelization.md](voxelization.md): the classical fraction of probe
density at a grid point is `exp(−U(g)/k_BT)`; summing it over the cell
converts hard volume bookkeeping into a continuous, physically sourced
measure that agrees with hard-sphere counting where the latter is well
defined and interpolates gracefully where it is not.

## GPoreU procedure

### Step 1 — `grids` (`gporeu.grid.all_grids_pbc`)

Produces, for every grid point: `he_lj_energy` (¼ of the full LJ
potential, temperature units), `geo_grid_occ / he_grid_occ /
n_grid_occ`, `grid_radaii`, plus provisional unfiltered volumes
`he_pore_volume`, `geo_pore_volume` (computed as cell averages before
gating; not what gets reported).

### Step 2 — `connectivity` (`gporeu.percolate.GPoreU_percolate`)

1. Bond the free lattice: interior 6-neighbour pairs of geometrically
   free points (`ij_pairs`), plus wrap pairs across opposite faces
   (`periodic_ij_pairs`) for the geometric mask.
2. Connected components over these bonds (Triton union-find,
   `get_clusters` → `components`). Points belonging to no edge get the
   `−1` sentinel.
3. Per-axis percolation test (see [percolation.md](percolation.md)):
   clusters touching *both* opposite faces — with ≥ 2 distinct shared
   labels — mark that axis as percolating. The union of clusters that
   percolate **along any axis** becomes the boolean accessibility mask
   (`he_access`, `geo_access`, `n_access`).
4. **Volumes are then computed on those masks only:**

   *helium-accessible:* for at least one percolating axis,

   ```
   V_He = V_cell · Σ_{g ∈ he_access} exp( −4 · E_He(g) / 298 K ) / N_grid
   ```

   The `he_lj_energy` field already holds `LARGE` at hard-colliding
   points, so those sites contribute exactly ~0 without extra logic.

   *geometric:* `V_geo = V_cell · #geo_access / N_grid`
   (a plain voxel count — the geometric convention has no energetic
   gradient to weigh).

   If no axis percolates, both volumes are reported as 0: a material
   with only enclosed cavities is *non-porous to transport* by
   definition.

## Why performed this way

* **Boltzmann for He, counting for geometric.** The helium volume is a
  probe-thermodynamics quantity; the geometric volume is a framework
  quantity — mixing weightings would blur the two diagnostics. Their
  difference also flags structures with large excluded shells relative
  to channel size.
* **Percolation gating at face value, any-axis semantics.** A pore
  network connected in *any* crystallographic direction communicates
  with the (periodic) exterior; requiring per-axis connectivity would
  under-report intercrate diffusion paths. Gating with the union-of-
  percolating-clusters mask exactly reproduces the operational notion
  "connected to the outside under periodicity".
* **Isolated-cavity removal, not penalty.** In powder compaction or
  mixed-matrix membranes such voids are dead volume; reporting per-
  colated volume matches experiment (and the oracle PoreBlazer values,
  against which GPoreU's volumes agree within a few percent — see
  project verification notes).
* **298 K reference for the weight.** Kept deliberately as in the
  reference toolchain; helium probe volume is weakly temperature
  sensitive in this regime, and the fixed constant keeps results
  comparable across the literature it reproduces.

## Limitations

* Grid spacing discretizes the free-volume surface: expect ≈ O(Δ)
  boundary error; Δ = 0.4 Å is accurate to well within the ±1–2 %
  spread seen across probe definitions themselves.
* Rigidity and framework-breathing are not modeled (single CIF
  geometry) — an intrinsic limit of static characterization.
* The Boltzmann weight is classical; para-hydrogen/quantum delocalization
  corrections used in ultra-precise surface analyses are out of scope.

## Ordered function calls

```
gporeu.utils.structure.prepare_input
  gporeu.utils.structure.read_cif            [_fast_read_cif | fill_uc]
  gporeu.utils.structure.init_grid
  gporeu.utils.force_fields.UFF.forward
  gporeu.utils.force_fields.MASS.forward
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.percolate.GPoreU_percolate
  gporeu.percolate.ij_pairs
  gporeu.percolate.periodic_ij_pairs          # geometric bonds
  gporeu.percolate.get_clusters               # per mask (geo, He, N2)
    gporeu.utils.union_find.components
      union_find._cc_merge_kernel
      union_find._cc_final_kernel
  gporeu.percolate.he_pore_volume
  gporeu.percolate.geo_volume
gporeu.cli.format_summary                     # report construction
```
