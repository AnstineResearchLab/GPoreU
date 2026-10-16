# Foundations: voxelization and the probe fields

*Used by: porosity, percolation, FFV, PSD, pore network, PLD/LCD (and
optionally the energy grid). This document describes the `grids` step —
the substrate every other method consumes.*

## What is computed

For a unit cell filled with a regular voxel lattice, GPoreU assigns to
every grid point **g**:

| field (data key) | meaning |
|---|---|
| `grid_radaii[g]` | **free radius**: `min_i ( d(g, atom_i) − σ_i/2 )` — distance from the grid point to the nearest atom *surface* (Lennard-Jones σ/2 as vdW radius); 0 at geometrically occupied points |
| `geo_grid_occ[g]` | geometric occupancy flag (point-probe convention) |
| `he_grid_occ[g]`, `n_grid_occ[g]` | helium / N2 hard-exclusion occupancy flags |
| `he_lj_energy[g]`, `n_lj_energy[g]` | soft probe–framework LJ energy sums (temperature units, see below) |
| `he_pore_volume`, `geo_pore_volume` | first-pass Boltzmann-weighted and counting volumes (later gated on percolation by the connectivity stage) |

## Theory

### The probe-particle picture

Porous-material characterization is built on the *probe particle*
idealization: a pore is characterized by the motion a small spherical
test particle (He, N2, CH4) could perform inside it. Because quantum
effects at cryogenic temperatures are well approximated by classical
statistical mechanics for these gases, the practical criterion becomes
geometrical/ energetic at the atom level:

* the probe center may sit anywhere whose **distance to every framework
  atom** exceeds a probe-dependent exclusion distance;
* a pore is *accessible* if a chain of admissible positions connects it
  to the outside (or wraps the crystal, see [pld.md](pld.md)).

Discretizing position space into a grid converts these existence
questions into lattice problems of the kind this package solves.

### Why a voxel grid

The alternative to grid characterization — analytical geometry over
Voronoi/Delaunay decompositions — either scales poorly or cannot express
Boltzmann-weighted accessibility. A regular grid over the cell:

* makes every quantity a **field over N = nx·ny·nz points**, allowing a
  single fused GPU kernel to compute all distance and energy tests in
  O(N_atoms · N) work per structure (trivially parallel, cache-friendly
  in atom-blocked tiles);
* gives every derived quantity (volume, surface, percolation, PLD) a
  common language on the *same* lattice, so the outputs of different
  methods stay mutually consistent;
* has a *controlled and visible* approximation error: grid spacing Δ
  bounds every reported length to ~Δ, and refining Δ tightens all methods
  at once. Δ = 0.4 Å is the default: helium-sized probes are ~2.6–3 Å,
  so at least six voxels span the narrowest channels of interest.

### Lattice placement

`init_grid` walks each **lattice vector row**: `nx = floor(|a|/Δ)` etc.
Using row *lengths* (not Cartesian extents) is essential for tilted
(triclinic) cells: the Cartesian diagonal entries are projections, which
would coarsen one axis and silently drop lattice planes. Points are
uniform in fractional coordinates, `linspace(0, 1 − underfrac, n+1)`
with `underfrac` absorbing the remainder; the plane at fractional 1 is
*excluded*, which is exactly right under periodicity (it coincides with
plane 0 of the next image and would otherwise double-count).

Return values: Cartesian point array `grid` (N, Å), the plane counts
`nx+1, ny+1, nz+1`, and `trials = 512 × N_atoms` (the number of random
draws the surface-area stage standardizes on).

## The kernel: `GPoreU_Lattice_Kernel`

One program per grid point; the atom list is streamed in blocks of 128.
For each point it tracks: `geo_check / he_check / n_check` flags and the
running LJ energy sums, early-exiting each probe's loop once the point
is decided for that probe. Distances are computed minimum-image:
displacement → fractional (row-lattice convention `v = f @ C`,
`f = d @ inv(C)`) → wrap `f − floor(f + 0.5)` → back to Cartesian.
This is exact for *any* cell shape and is the reason GPoreU handles
triclinic cells without special-casing.

For each atom i (with grid-point distance d):

1. **Geometric mask** — occupied iff `d < σ_i / 2`.
   This is the *point-probe* convention: atoms occupy disks of radius
   σ_i/2 and the probe has no size. It defines the "geometric" free
   volume used for framework packing diagnostics, and — crucially — the
   **free radius field** `d − σ_i/2` (the space available to a
   center sitting at g measured from van-der-Waals surfaces).
2. **Helium mask** — occupied iff `d < σ_mix(He,i) / 2`,
   σ_mix = ½(σ_He + σ_i). Otherwise the LJ term is accumulated:
   `E_He += ε_mix( (σ_mix/d)^12 − (σ_mix/d)^6 )`
   (note: without the conventional factor 4 — see below).
3. **N2 mask** — occupied iff `d < σ_mix(N2,i)` — a *doubling* of the
   He convention's exponent-free threshold: nitrogen is treated as the
   much more exclusionary probe, matching the reference toolchain
   behavior GPoreU reproduces (e.g. N2-impermeability of narrow-pore
   structures at 77 K while He still samples the volume).
4. **Free radius** — `min_d = min_i ( d − σ_i/2 )` over all atoms,
   stored as `grid_radaii` for geometrically free points.

Occupied points are stored with `grid_radaii = 0` and LJ sums
`LARGE = 1e10` — "make the Boltzmann weight numerically zero" sentinels
rather than branchy special-casing downstream.

### The stored energy is one quarter of the full LJ potential

The accumulated He/N2 sums omit the `4·` of the textbook 12-6 form. The
volume integrals elsewhere multiply the stored sum by 4 again in the
Boltzmann exponent:

```
w(g) = exp( −4 · E_soft(g) / 298 K ) = exp( −U_LJ(g) / (k_B · 298 K) )
```

so the physics is the canonical room-temperature Boltzmann weight of the
full potential at a reference temperature of 298 K (the temperature the
reference implementations standardize on). Keeping the raw ε(σ¹²/r¹² −
σ⁶/r⁶) sum in the field and the factor inside the exponent preserves
this without recomputation.

## Why a *soft* (Boltzmann) volume at all?

A purely binary accessible/occupied grid makes pore volume discontinuous
in the probe radius: a grid point pops in and out as `d` crosses the
exclusion distance, and a structure's volume jumps with quantization
steps of the lattice. The Boltzmann weight `exp(−U/kT)` gives every grid
point a fractional contribution that dies out smoothly as the probe is
pressed into the wall (repulsion) and saturates at 1 deep in free space.
Two consequences:

* sub-voxel information survives discretization (a channel slightly
  narrower than nominal still contributes a *fraction*);
* the result is directly interpretable as the classical, temperature-
  resolved probe availability rather than a hard geometric accident.

The hard mask (`he_grid_occ`) is retained in parallel because *graph*
methods (percolation, PLD) require binary adjacency, and because points
in true collision (r < σ_mix/2) must contribute strictly zero, never a
small number.

## Outputs of note

`he_pore_volume = V_cell · Σ_g exp(−4·E_He(g)/298) / N` and
`geo_pore_volume = V_cell · (# geometric-free points) / N` are computed
here as *unfiltered* cell averages — every grid point, percolating or
not, isolated cavities included. The connectivity stage replaces them
with the percolation-gated "accessible" volumes that are actually
reported for porosity (see [percolation.md](percolation.md)).

## Justification summary

| choice | rationale |
|---|---|
| single fused kernel, all fields at once | one streamed pass produces every quantity; atom-block early-exit cuts cost where probes are already excluded |
| minimum-image everywhere except the energy stage | for pair-distance *tests* (r < few Å ≪ half-cell in the vast majority of real cells) the minimum image is exactly the nearest image; the expensive image-sum requirement is unique to long-range cutoff sums ([energy_grid.md](energy_grid.md) handles it rigorously) |
| σ_i/2 vdW radius convention | matches the reference toolchain (PoreBlazer/IRMOF-era characterization practice), so absolute numbers align with the literature |
| geometric mask independent of probes | keeps a probe-free "solid vs void" definition usable by the pore-graph and LCD stages, which describe framework topology, not one gas |
| row-length grid, boundary plane excluded | correct sampling of tilted cells without duplicate periodic planes |

## Ordered function calls

```
gporeu.grid.all_grids_pbc
└── gporeu.grid.GPoreU_Lattice_Kernel          # Triton, 1 program / grid point
```

(Executed by the `grids` step; input dict produced by
`gporeu.utils.structure.prepare_input` — see index.)
