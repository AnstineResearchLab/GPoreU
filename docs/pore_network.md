# Pore network: H-maxima and watershed partitioning

*CLI method: `pore_network` (aliases `poregraph`, `network`) · steps:
`grids → connectivity → network`*

## What is reported

IRMOF-1 (real run):

```
pore network       : 68 pore(s); largest pore radius ~7.52 A
     floods 46 sweeps; 27 periodic center merges
```

plus, in the pipeline dict: the retained pore centers
(`markers`/`maxima_ids`, with `maxima_radii`), the per-voxel assignment
(`pore_labels`: 0 = unassigned/background, k = basin k), `num_pores`,
`flood_iterations`, `periodic_pairs`, and the safe-region bookkeeping.

## Theory

### Pores as H-maxima of the free-radius field

A porous framework is not a list of spheres connected by cylinders, but
the *free-radius field* `r(x)` (distance to the framework surface)
encodes exactly that intuition: local maxima of r are *cavities*;
saddles between maxima are *windows*. The rigorous version is morphological:
treating the superlevel sets

```
S_t = { x : r(x) ≥ t }
```

as water is drained (t lowered), connected components of S_t *merge* at
each saddle; the local maximum whose component survives longest
absorbs the others. The maxima that never get absorbed — those with
**prominence** above a threshold — are the significant pores, and the
basins they sweep up as the water rises partition the void. This is
exactly the **H-maxima / watershed** transform from mathematical
morphology (Vincent & Soille; Meyer; the "flooding" metaphor is literal),
and it is the same construction PoroNet uses to obtain its pore graph.

Why prominence rather than raw maxima? Because a discretized r(x) field
covered in tiny local bumps would otherwise report thousands of
"pores" per structure. The prominence parameter h answers "how much
deeper than its col is this lake" — a robust, physical significance
test.

### Maximum spanning tree: the right skeleton for H-maxima

Computing mergers directly over the 6-neighbour lattice graph is
sequential and awkward; the classical simplification: for a vertex-weighted
graph with edge weight `w(u,v) = min(r(u), r(v))`, the merge structure of
all superlevel sets S_t is carried entirely by any **maximum spanning
tree (maxST)** of the graph — Kruskal's criterion: u, v lie in the same
component of S_t iff the path between them in the maxST has every edge
≥ t (the path maximizes the minimum edge — the widest-path property).
So: build one maxST (cheap, parallel Borůvka), run *reconstruction*
on the tree instead of the mesh. Besides being fast, this makes the
combinatorics tree-simple and the result deterministic up to ties.

Note GPoreU builds the tree on **interior edges only** (`ij_geo`):
periodicity is folded in at the end by the periodic-merge stage, which
is more robust than threading wrap edges through the tree (a tree on a
torus has nontrivial cycle structure; the merge stage handles it
directly).

## GPoreU procedure

`gporeu.pore_graph.GPoreU_poreGraph`, on top of grids + connectivity:

1. **Local maxima detection** (`_local_maxima_kernel`): over the bond
   graph, a point is kept if no neighbour has a larger radius (atomic
   OR-flag elimination of non-maxima), **and** `r > 1.0 Å` — the
   minimum-cavity criterion that discards surface noise in dense regions
   where `r ≈ 0`. Candidates are sorted by radius descending.
2. **Maximum spanning forest** (`maximum_spanning_tree` →
   `gporeu.utils.union_find.max_st`): parallel Borůvka over
   `w = min(r_i, r_j)` edges; the implementation guarantees one link
   per merge phase chosen by a deterministic ballot, so the forest is a
   reproducible function of the input (ties resolved by vertex id).
3. **H-max reconstruction** (`_hmax_kernel`, iterated until fixed
   point): start from `reconstructed = r − h` (h = **0.5 Å**, the
   prominence threshold) and propagate upward along tree edges the
   value `min(level_neighbor, edge_w)` with atomic max — "raise the
   water; every vertex takes the highest level at which the drained
   flow from its own peak reaches it, capped where the terrain is lower
   (col crossing)". At convergence, a maximum *survives* reconstruction
   iff its peak value `r − reconstructed > h`; those surviving maxima
   (`maxima_height ≥ h − 1e-6`) are the H-maxima / candidate pore
   centers.
4. **Adjacency merge** (`components` over the maxima-induced subgraph):
   two retained maxima joined by a grid edge are one pore; per cluster
   the widest vertex (first occurrence of the cluster maximum radius)
   remains — two maxima cannot be one cavity without a wall between
   them anyway, and the wider center anchors the basin.
5. **Watershed / safe-region flooding** (`GPoreU_safe_watershed`):
   * *Safe regions*: each center i keeps radius
     `safe_i = max(0, min(r_i, min_{j≠i}(d_ij − r_j)) − safe_buffer)`
     with `safe_buffer = 6.0 Å` (configurable). Voxels claimed by
     exactly one safe sphere become that pore's seed with capacity
     `r(voxel)` (centers themselves are always seeds). The buffer
     guarantees seeds are uncontested by definition — flooding starts
     from unambiguous territory, which is what makes the competitive
     basin assignment well posed. (Centre distances are Euclidean,
     within-cell — cross-cell twins are dealt with in step 6, not by
     wrapping these pairs.)
   * *Competitive flooding* (`GPoreU_Flood_Sweep_Kernel`, iterated):
     every **competitive** voxel (void, not safe) repeatedly adopts the
     best proposal from its six non-periodic neighbours: proposal
     capacity `min(capacity_neighbour, r(voxel))` — the bottleneck of
     extending the neighbour's path. Higher capacity wins; ties go to
     the *smaller* label; out-of-cell neighbours propose −∞ — so
     basins grow from the safe cores through the pore-interior
     bottleneck-geometry, and voxels reached only by paths below every
     center's level stay unassigned (walls / dead ends, label 0).
     Sweeps run in six directed passes per global iteration,
     double-buffered, until a full iteration changes nothing
     (`flood_iterations` = sweep count). Each voxel's final label is
     the basin whose widest path into it carries the highest bottleneck.
6. **Periodic consolidation** (`GPoreU_apply_pbc`): a pore touching a
   cell face is plausibly the same pore reappearing on the opposite
   face. In each dimension, take the `wall_cells = ceil(wall_thickness
   / grid_spacing)` voxel layers at both faces (default wall thickness
   1.0 Å), count label co-occurrences across the two layers, and accept
   a merge when one label's fraction exceeds 50% on either side — a
   genuine continuation, not a grazing touch. Then the merge direction
   is decided physically: only the *smaller-radius* pore may be absorbed,
   and only if its center is actually close to the face in question;
   merges applied sequentially keeping the *larger-radius* surviving
   center (ties → smaller label). Final labels are renumbered 1..num_pores.

## Why performed this way

* **H-maxima on the maxST** instead of raw local maxima or spherical
  clustering: the watershed definition is mesh-faithful (mergers occur
  at real saddles of the discrete field), scale-controlled by a single
  interpretable parameter h, and stable to grid noise below h. The maxST
  reduction turns an O(V-E) sequential flooding into a few parallel
  tree sweeps with provably identical merge structure.
* **Interior-only tree + explicit periodic merge**: keeping wrap edges
  out of the tree preserves the tree (vs. torus-cycle) reconstruction
  argument, and the wall-overlap criterion is strictly stronger than
  edge-identity: two basins are merged only when they demonstrably
  describe the same continuous object across the face, judged by
  voxel-countable overlap rather than coordinate equality.
* **Safe regions with a 6 Å buffer**: seeds that are themselves near
  basin maxima prevent flooding wars at the cores and suppress the
  classic watershed problem of basin pinching; the buffer also matches
  one of PoroNet's hyperparameters so pore *counts* remain comparable
  to that ecosystem.
* **Capacity = path bottleneck, smaller-label tie-break**: the
  assignment rule is the graph-theoretic "widest path" (maximin) — the
  natural diffusion-limited allocation of a void voxel to a pore — and
  ties broken by id make it total and deterministic. The six ordered
  directed passes with double buffering make each sweep an exact,
  race-free snapshot (no sequential-ghost updates).
* **Absorbing only the smaller pore on periodic merges**: prevents the
  "two big cavities claiming each other's territory" artifact — the
  small pore is the one plausibly clipped by the cell boundary;
  additionally requiring its center near the face prevents merges of
  genuinely distinct pores that merely graze shared walls.

## Limitations

* Pore *counts* depend on (h=0.5 Å, r > 1 Å, safe_buffer 6 Å,
  wall thickness 1 Å) — meaningful as relative comparisons at fixed
  settings (PoroNet-comparable), not absolute.
* Basins are geometric, not chemical: no adsorbate specificity beyond
  the radius field's N2-ish wall convention.
* Unassigned void voxels (label 0) are legitimate wall voxels, not an
  error; a large unassigned fraction flags a structure whose pores are
  separated by bottlenecks below h.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.percolate.GPoreU_percolate           # supplies ij_geo + access masks
  gporeu.percolate.ij_pairs
  gporeu.percolate.periodic_ij_pairs
  gporeu.percolate.get_clusters
    gporeu.utils.union_find.components
gporeu.pore_graph.GPoreU_poreGraph
  gporeu.pore_graph._local_maxima_kernel
  gporeu.pore_graph.maximum_spanning_tree
    gporeu.utils.union_find.max_st
      union_find._ms_find
      union_find._ms_best
      union_find._ms_ballot
      union_find._ms_hook
      union_find._ms_edges
  gporeu.pore_graph._hmax_kernel            # fixed-point reconstruction loop
  gporeu.utils.union_find.components        # adjacent-center merge
  gporeu.pore_graph.GPoreU_safe_watershed
    gporeu.pore_graph.GPoreU_SafeRegion_Kernel
    gporeu.pore_graph.GPoreU_Flood_Sweep_Kernel   # sweep loop
    gporeu.pore_graph.GPoreU_apply_pbc
gporeu.cli.format_summary
```
