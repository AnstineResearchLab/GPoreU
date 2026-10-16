# Percolation: direction-resolved connectivity of the void

*CLI method: `percolation` · steps: `grids → connectivity`*

## What is reported

```
percolates He        : [yes,yes,yes]  (x,y,z)
percolates N2        : [no ,no ,no ]  (x,y,z)
percolates geometric : [yes,yes,yes]  (x,y,z)
```

Three per-axis yes/no flags — one per probe convention — stating
whether the probe-accessible void of the framework forms an *infinite
connected cluster* along each crystallographic direction. Internally
the stage also produces the accessibility masks (`he_access`,
`n_access`, `geo_access`, boolean point fields over the grid) that
gate the reported volumes, and the bond lists (`ij_geo`,
`ij_geo_periodic`) that the network and PLD stages re-use.

## Theory

### Why percolation, not just connected components

"Does this material have pores" is not a question about components —
every structure has void pockets. It is a question about *infinity*: an
adsorptive can enter a region only if the region connects to the
outside of the crystal. For a single crystal modelled under periodic
boundary conditions, "connects to the outside of every crystal" means
"belongs to a cluster of infinite extent", and the theory of when such
clusters exist is percolation theory on periodic graphs (Stauffer &
Aharony): above a critical void fraction a spanning cluster appears;
its transport dimensionality (1/2/3) is what separates channel
zeolites from layered clays from 3D-interconnected nets.

Under PBC there is no literal exterior to touch, but there is an exact
equivalent test: a cluster in the periodic quotient graph extends to
infinity in direction **d** iff it connects a face of the cell to the
opposite face in d — equivalently (graph-theoretically) iff it contains
a cycle with nonzero winding vector along d (the homological view used
by [pld.md](pld.md); the two tests agree because crossing a face *is*
winding ±1). The face test is what GPoreU uses for the boolean flags:
it is one pass over two cross-sections and needs no cycle basis.

### Why three probes

The connectivity answer is probe-dependent by construction — the void
graph is built from the probe's exclusion mask:

* **He** — the "can a molecule get in at all" answer at the smallest
  classical probe size; matches helium pycnometry semantics.
* **N2** — the adsorption-temperature workhorse: a material that
  percolates to He but not to N2 (e.g. zeolite WEI at the default
  grid) reports the gating that makes it invisible to
  77 K nitrogen uptake, even though its geometric void percolates.
* **geometric** — probe-free framework connectivity, used as the
  transport skeleton for the pore-network stage and as the
  "is there connected void at all" reference.

The triple of flags is therefore a genuinely informative signature
(3 binary numbers spanning "dense", "He-open but gated", "N2-open 1D /
2D / 3D"), not redundant output.

## GPoreU procedure

`gporeu.percolate.GPoreU_percolate`, on top of the grid fields:

1. **Bond construction** (Triton lattice kernels, one launch per
   mask): `ij_pairs` joins each accessible grid point to its +x/+y/+z
   accessible neighbours (interior adjacency, deduplicated);
   `periodic_ij_pairs` additionally joins the last plane to the first
   along each axis — the wrap bonds that make "opposite faces" the
   test for infinity.
2. **Connected components** (`get_clusters` → Triton union-find
   `components`): every point receives its component root. Points in
   **no** edge (occupied points and isolated free points) carry the
   sentinel label `−1`.
3. **Face-percolation test**, per axis d, per mask:
   take the component labels on the two opposite faces and intersect
   them:
   `labels shared between face(0) and face(last)`; percolates along d
   iff that shared-label set has **more than one** member. The `> 1`
   (not `> 0`) condition is a deliberate sentinel guard: if one entire
   face is occupied, its only label is `−1`, which trivially appears on
   the other face too; requiring at least one *real* shared cluster
   prevents an all-wall cross-section from counting as a connection.
   With the sentinel occupying at most one slot of the set, more than
   one member ⇔ at least one genuine cluster touches both faces.
4. **Access masks**: for each probe, the boolean mask marks all points
   whose cluster percolates along *at least one* axis — the region a
   probe entering the (infinitely replicated) crystal can roam.
5. **Volumetry on the masks**: `he_pore_volume` applies the Boltzmann
   weights and `geo_volume` the plain count, but now only over the
   percolating clusters; the per-probe accessible volumes reported for
   porosity are replaced here ([porosity.md](porosity.md)). If no axis
   percolates, the volumes are set to 0 — non-porous to transport.

## Why performed this way

* **Union-find components once, everything else read off it**: face
  tests, access masks, and the volumetry all reference the same label
  field; the expensive global operation (labeling ~10⁵–10⁶ lattice
  points) happens exactly once per probe, and the derived products are
  linear passes. The same label machinery underpins the PLD/engine
  stages ([pld.md](pld.md)), so GPoreU keeps one shared `utils/
  union_find` implementation.
* **Face test for the booleans** (winding test stays in the PLD
  engine): for a yes/no per-axis answer, cross-section intersection is
  cheaper and easier to audit than cycle homology, and it is the
  criterion the reference tools' logs use, keeping flag semantics
  identical.
* **Any-axis union for the masks**: a cluster winding along *any* of
  the three lattice directions is infinite; intersecting with the union
  is exactly "the infinite cluster(s)" under PBC. (Per-axis masks are
  available from the flags if a user needs direction-specific access.)
* **Volume zeroing on no-percolation, not "volume of isolated
  pockets"**: matches uptake physics — closed cavities never fill from
  the gas phase; a material reporting He-pocket volumes as
  "accessible pore volume" would mislead screens.
* **Interior bonds separate from wrap bonds** (`ij_geo` vs
  `ij_geo_periodic`): downstream consumers choose whether wrapping is
  admissible — the pore-graph floods non-periodically (basins are
  consolidated afterwards by the explicit periodic-merge stage, which
  needs the two bond classes apart), while the PLD engine must include
  wrap edges to see windings. Producing both once here avoids each
  stage rebuilding lattice adjacency.

## Limitations

* Flags inherit the grid discretization: an aperture of one grid
  spacing's width is open or closed depending on where Δ = 0.4 Å
  happens to place voxels; near-threshold gating decisions can flip
  with `grid_spacing` (this is physically honest — such apertures are
  at the probe scale).
* Rigidity: thermal gate-opening (why real WEI-class materials take up
  some gas despite closed windows) is outside the static model.
* The face test certifies the *cluster percolates*, not its aperture —
  the aperture is the PLD's job; combining the boolean flags with the
  per-axis PLD gives the same information as a per-threshold
  filtration scan, at two independent-computation cost.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.percolate.GPoreU_percolate
  gporeu.percolate.ij_pairs                 # interior bonds (geo, He, N2)
  gporeu.percolate.periodic_ij_pairs        # wrap bonds (geo)
  gporeu.percolate.get_clusters             # per probe mask
    gporeu.utils.union_find.components
  gporeu.percolate.he_pore_volume           # gated Boltzmann volume
  gporeu.percolate.geo_volume               # gated voxel-count volume
gporeu.cli.format_summary
```
