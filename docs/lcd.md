# Largest cavity diameter (LCD)

*CLI method: `lcd` · steps: `grids → connectivity → pld`
(computed inside the PLD call)*

## What is reported

IRMOF-1 (real run):

```
LCD                :     15.045 A
```

The diameter of the largest sphere that fits anywhere inside the
framework void of the unit cell.

## Theory

### Cavity size as a covering radius

Distance-transform geometry gives the LCD directly. For the void region
Ω = complement of the (geometric) framework exclusion spheres, define
the *free radius* at a void point x:

```
r(x) = min over framework atoms i of ( |x − atom_i| − σ_i/2 )
```

the radius of the largest sphere centered at x contained in Ω. The
largest inscribed sphere of the unit cell then has radius `max_x r(x)`
and

```
LCD = 2 · max_x r(x)
```

with the sphere's center being the maximizer — a cavity center in the
distance-transform sense (Voronoi vertex of the atom spheres in the
continuum limit).

This is the standard "largest cavity" of MOF characterization: paired
with the PLD ([pld.md](pld.md)) it yields the classic cavity-vs-window
picture — LCD says how big the room is, PLD says how big the door is.
For a material whose largest cavity is isolated (e.g. sodalite-type
cages in a window-blocked framework) LCD > 0 while PLD = 0; the pair
of numbers immediately classifies that.

### On the probe convention

`grid_radaii` measures from *atom-center minus σ/2* spheres — the
geometric (point-probe) wall convention. Consequences:

* The LCD is a **framework property**, not tied to a particular
  adsorptive (unlike the Boltzmann-weighted He volume); it is the
  "space between van-der-Waals surfaces", which is what cavity-size
  tables in the literature report.
* Reported values depend on the σ scale used for the framework atoms
  (UFF here) — LCDs from force fields with different vdW radii differ
  systematically. This is a convention statement, not an error.

## GPoreU procedure

1. The `grids` stage stores, per void point, exactly
   `r(g) = min_i (d(g, i) − σ_i/2)` with periodic minimum-image
   distances; occupied points are pinned to 0.
2. `calculate_pld` (which already runs on the radius field) evaluates

   ```
   LCD = 2 · max { r(g) : g not geometrically occupied }
   ```

   as a single masked max-reduce — one kernel, no extra passes
   alongside the [pld.md](pld.md) computation that consumes the very
   same field.
3. No percolation gating: the LCD is by definition about the largest
   cavity *that exists*, whether or not a probe can reach it. (Pair
   with the percolation flags to tell whether it can.)

## Why performed this way

* **Grid maximum over the distance transform** is the honest discrete
  answer: the continuum maximizer is a Voronoi-vertex event whose
  determination under periodic triclinic lattices is intricate and,
  measured against the grid answer, lands within the same
  interpolation error already present in the discretization itself.
  At Δ = 0.4 Å the systematic underestimate (true center off-grid, so
  r(x\*) ≤ max_grid r + O(Δ)) is ≲0.4 Å absolute — within the
  σ-convention convention spread noted above.
* **Geometric rather than probe-softened wall**: a probe-specific
  weighting would make the LCD probe-dependent and destroy its role in
  the LCD–PLD pair — both must describe the same framework "skeleton"
  or their difference (cavity-vs-window contrast) is meaningless.
* **No connectivity filter**: encodes the physically useful asymmetry
  between "how big is the biggest hole" (this) and "how big a probe
  gets in" (PLD) — two numbers reporting different experiments.
* **Computed inside the PLD stage**: both consume only `grid_radaii`,
  so the CLI reports them per the user's toggle while the work happens
  once.

## Limitations

* The grid maximum is a lower bound of the continuum LCD (≤ ½Δ
  typical underestimation of the radius, worst case Δ); refine
  `grid_spacing` when sub-0.1 Å LCD matters.
* A spherical probe maximizer is not always the physically relevant
  "cavity" shape (e.g. slit pores: the LCD sphere under-describes a
  large-area slot); inspect the network basins and PSD for those.
* Static framework; σ-convention dependence as discussed.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel         # produces grid_radaii
gporeu.percolate.GPoreU_percolate           # (pulled in as PLD dependency)
gporeu.pld.calculate_pld
  gporeu.pld.pld_winding_fast               # shares the radius field
  (torch masked max: lcd = 2 * grid_radaii[~geo_grid_occ].max())
gporeu.cli.format_summary
```
