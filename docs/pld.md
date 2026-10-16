# Pore limiting diameter (PLD)

*CLI method: `pld` · steps: `grids → connectivity → pld`*

## What is reported

Zeolite WEI (real run):

```
PLD                :      1.852 A (dimension D2)
     per axis: X 1.852   Y 1.432   Z 1.852 A
LCD                :      2.574 A
```

* **PLD** — the diameter of the largest spherical probe that can
  travel infinitely far through the framework, under periodic
  boundary conditions. A material with no open channel reports 0.
* **Per-axis PLD** — the same quantity restricted to channels that
  wind along each crystallographic direction (the engine additionally
  returns the winding vector of each critical cycle in its result
  dict: `[1,0,0]`-type vectors are straight channels, `[1,1,1]` a body
  diagonal).
* **Dimension** — the number of independent directions that remain
  percolating at the reported PLD (PoreBlazer's dimensional
  classification: 0 = non-porous, 1 = channel material, 2 = layered
  channel system, 3 = 3D-interconnected).

## Theory

### PLD as a percolation-with-aperture problem

The pore limiting diameter is the crossover quantity of transport in
rigorous porous solids:

```
PLD = sup { d : a probe sphere of diameter d has an unbounded
             path through the periodic framework }
```

It is *not* the narrowest constriction near the largest cavity (that is
the "cavity + window" decomposition of the pore network) and it is *not*
the largest inscribed sphere (that is [lcd.md](lcd.md)); it is a
global, percolation-theoretic aperture. On the voxel lattice with free
radius field `r(g)`, define, for a probe radius t, the sub-lattice

```
L_t = { g : r(g) ≥ t }      with the 6-neighbour periodic bonds
```

Then PLD/2 = the largest t at which `L_t` percolates (contains a
component of infinite extent). This is a **filtration**: as t decreases
from the max radius, clusters merge and at some critical t an
*infinite cluster* appears. Computing the PLD is locating that
threshold.

### Detecting infinite clusters: winding vectors

A finite periodic cell contains no literal infinity — the test is
homological. A cluster in the quotient graph (the cell graph with wrap
edges) corresponds to infinite clusters of the covering graph **iff it
contains a cycle with nonzero winding / holonomy vector**

```
g(e) ∈ Z³,   g = (net crossings of the x, y, z face families)
```

Concretely, assign every bond its displacement `W ∈ {−1,0,+1}³` (nonzero
only for wrap bonds); extend additively along paths: a cycle's `g` is
the lattice translation it realizes. Such a cycle, repeated, escapes to
infinity in direction g — and only such cycles do. This is the standard
"topological criterion for percolation in periodic graphs" underlying
PoreBlazer's PLD scan and is equivalent to asking whether the cluster's
image in the maximal abelian cover remains connected (i.e. the cluster's
spanning-tree potential differences are nontrivial).

### The filtration and widest paths

The naive scan — rebuild connected components at every candidate t,
test winding — needs 10s–100s of union-finds. Two classical facts
collapse it to essentially one:

1. **Percolation events live on a maximum spanning tree.**
   Build the maxST of the periodic bond graph with edge weights
   `w(u,v) = min(r(u), r(v))` (the bottleneck the edge admits). The
   maxST path between any two vertices is a *widest path* (maximizes
   the minimum edge), so u, v lie in the same component of `L_t` iff
   every edge on their maxST path weighs ≥ t. The entire filtration —
   including when infinite clusters appear — is readable from the tree
   (and the edges outside it).
2. **Critical edges are exactly the winding edges of the tree.**
   Give the maxST, root it and propagate to each vertex its accumulated
   winding potential `P(v) ∈ Z³` (tree-path sum of W). For every bond
   e = (u,v) — tree or not — the fundamental-cycle winding is

   ```
   g(e) = P(v) − P(u) − W(e)  ∈ Z³
   ```

   (the cycle "tree path u→v plus the chord e"). Then the punchline:

   ```
   PLD_d = 2 · max { w(e) : g(e)[d] ≠ 0 }
   ```

   **Proof sketch.** Take the exact threshold t\* = PLD/2: at t\* some
   cluster C of `L_{t*}` percolates along axis d, and any winding cycle
   e inside the cluster has both endpoints in the same `L_{t*}` tree
   component, i.e. w(e) ≥ t\* and g(e) along d. Conversely, an edge
   whose fundamental cycle has `g(d) ≠ 0` and weight w is, together
   with the tree path (all edges ≥ w), a winding cycle entirely inside
   `L_w` — hence L_w percolates along d and the PLD along d is ≥ w.
   Maximizing over edges proves equality, and the per-axis values /
   winding vectors come out of the same scan.

So the PLD reduces to: **one maxST + one potential sweep + one edge
scan** — exact (no threshold bisection), and every number carries a
certificate (the critical edge and its winding vector, which the CLI
prints).

### Why this replaced an MST-two-stage scheme

Earlier grid codes (and GPoreU's ancestor) ran a *binary search over
t*, rebuilding components each step, or scanned the maxST
edge-ascending and re-tested winding after each hook (PoreBlazer-style
filter-walk, which needs a per-step winding oracle). Those agree with
the formula above but spend 10–20× the work and their accuracy is
bounded by the scan granularity. The one-shot fundamental-cycle
identity is exact, parallelises to a handful of Triton sweeps, and —
because the maxST itself is the only sorted structure — runs in
O(E · α) per Borůvka phase with no per-threshold rebuilds.

## GPoreU procedure

Driver: `gporeu.pld.calculate_pld` → engine
`gporeu.pld.pld_winding_fast`, on grids + connectivity:

1. **Bond assembly** (`bonds_from_pairs`): the geometric free-lattice
   bonds from the connectivity stage (`ij_geo`) plus the wrap bonds
   (`ij_geo_periodic`), with the 3-vector `W` attached to every edge
   (zero for interior edges). Edge weights
   `w(e) = min(r(u), r(v))`; geometrically occupied voxels carry
   `r = 0` and are in no bond.
2. **Maximum spanning forest** (Borůvka, Triton
   `_find` / `_best` / `_ballot` / `_hook`): deterministic per-phase
   ballots (max weight, ties by edge id) so the tree — and every
   number derived from it — is a reproducible function of the
   structure. Contraction between phases keeps rounds O(log V).
3. **Final find + winding potentials**: after the last phase, a
   pointer-chasing find fixes final roots; edge winding residuals
   `g(e) = P(v) − P(u) − W(e)` are computed in one pass
   (`_score_kernel`) with P accumulated during the Borůvka hooking
   steps (potentials ride on the same unions — no separate DFS pass).
4. **Per-axis scan**: for each axis d, the max-weight edge with
   `g(e)[d] ≠ 0` yields `PLD_d = 2w` and the winding vector of the
   critical cycle; no such edge → `PLD_d = 0`.
5. **Overall PLD and dimension**: `PLD = max_d PLD_d` (a probe of that
   diameter has *some* infinite path — the PoreBlazer semantics of a
   porous material); the dimension counts axes still percolating at
   the overall PLD (`PLD_d = PLD`, i.e. ≥ PLD − ε): those and only
   those axes have infinite clusters when the probe is exactly at the
   limiting size.
6. (The LCD, `2 · max r(g)` over void points, is computed in the same
   call — see [lcd.md](lcd.md).)

## Why performed this way

* **Homological criterion over geometric window search**: window-hunting
  (intersecting Voronoi faces, sphere-sphere tangent geometry) either
  misses "sneaky" diagonal channels or needs probe-MC with
  finite-precision pass/fail. The winding test is combinatorially
  exact on the lattice, handles all lattice geometries (triclinic
  included) without special cases, and gives per-axis + dimension
  answers from the same computation.
* **Geometric (point-probe) bond lattice**: PLD is about the channel
  *geometry*, so the graph should be the framework void itself, with
  the probe size entering through the radius filtration — not baked
  into the adjacency. Using the `geo` bonds with weight `min(r_u, r_v)`
  implements exactly "the probe must fit at every point of its path".
* **One-shot exactness over threshold scans**: eliminates the classic
  trade-off between scan accuracy and cost; the reported value is the
  exact lattice answer, which sits within one grid spacing (~0.4 Å,
  measured ≤ 0.07 Å) of continuum references.
* **Deterministic maxST**: PLD is used in benchmarks and screens;
  bit-level reproducibility (fixed tie-breaking) matters as much as
  speed.
* **Certificate printing**: the engine returns the critical cycle's
  winding vector; inspecting `winding_x/y/z` distinguishes e.g. a true
  3D net from a channel along a body diagonal (`[1,1,1]`) — physics
  that a single number hides.

## Limitations

* The path lattice is 6-connected; a 26-connected adjacency admits
  slightly wider diagonal bypasses (up to √3 spacing) — the chosen
  convention is conservative and matches the reference tools.
* Grid discretization bounds absolute accuracy to ~one Δ (default
  0.4 Å); the discretized PLD is a strict lower bound of the lattice
  continuum value (any true aperture must be sampled by grid points).
* Static framework — gating from thermal motion (the reason experimental
  "effective apertures" differ) is outside any CIF-based method.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.percolate.GPoreU_percolate           # supplies ij_geo / ij_geo_periodic
  gporeu.percolate.ij_pairs
  gporeu.percolate.periodic_ij_pairs
  gporeu.percolate.get_clusters
    gporeu.utils.union_find.components
gporeu.pld.calculate_pld
  gporeu.pld.pld_winding_fast
    gporeu.pld.bonds_from_pairs
    gporeu.utils.union_find.max_st          # Triton Boruvka:
      _find / _best / _ballot / _hook       #   forest + winding potentials
    gporeu.pld._find_kernel                 # final positions
    gporeu.pld._score_kernel                # g(e) = P(v) - P(u) - W(e)
    (per-axis torch.where/argmax scan)      # PLD_d = 2 max{w : g[d] != 0}
  (lcd = 2 * max grid_radaii over ~geo_grid_occ)
gporeu.cli.format_summary
```
