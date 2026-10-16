# Pore size distribution (PSD)

*CLI method: `pore_size_distribution` (alias `psd`) · steps: `grids → psd`*

## What is reported

IRMOF-1 (real run):

```
PSD: max sampled pore diameter 7.52 A (bins 0.25 A to 20.0 A, 10000 cycles)
```

with the full distributions available in the pipeline dict:

| key | contents |
|---|---|
| `pore_out` | per sampled probe point, its pore diameter (Å); 0 where no sphere fits |
| `norm_cumul_psd` | normalized complementary cumulative distribution over the diameter bins |
| `psd` | the density form: central difference of the complementary CDF |

## Theory

### What a PSD means for a rigid framework

For a flexible/probed material the differential pore size distribution
f(d) is derived from adsorption isotherms (Horvath–Kawazoe, DFT, NLDFT,
QSDFT...) and is model-dependent. On a *rigid known structure* the PSD
can instead be computed from the geometry directly. Two established
geometric definitions exist:

1. **Accessibility-based** ("pore that a probe can enter"): distribution
   of bottleneck radii — closely related to the PLD picture.
2. **Cavity-based** ("space between walls"): at each interior point, the
   radius of the largest sphere *contained in the void* that covers the
   point; the PSD is the volume distribution of these radii. This is the
   PoroNet/PoreBlazer lineage (the "pore size" of an insertion point),
   and it is what GPoreU computes.

Formally: place probe centers at i.i.d. uniform positions p in the unit
cell and define

```
R(p) = max{ r : the sphere centered at p with radius r lies in free space }
```

The PSD is the distribution of `2R(p)` weighted by volume — the volume
fraction of cell space lying inside a void sphere of diameter d.

### Geometric discretization

Continuum containment is evaluated on the voxel lattice: a grid point g
has precomputed free radius `r_g = grid_radaii[g]`, meaning the sphere
`S(g, r_g)` is the largest sphere centered at g contained in void. Then

```
R(p) = max{ r_g : g candidate, |p − g| ≤ r_g }     (minimum image)
```

i.e. over the pre-computed *fitspheres*, take the largest covering
sphere. As the voxel spacing → 0 this converges to the continuum R(p)
from below; with Δ = 0.4 Å the sampling grid resolution error is small
compared to the bin size (0.25 Å typical) and the smoothing effect of
the volume weighting.

Note the probe convention: `grid_radaii` measures from vdW surfaces
(σ/2 spheres). PSD diameters therefore describe *hard-wall* cavity
sizes. The candidate pre-filter (`~n_grid_occ & r > 0`) fixes the probe
identity to the geometric/N2 wall convention used throughout GPoreU
(see [voxelization.md](voxelization.md), [ffv.md](ffv.md)).

## GPoreU procedure

`gporeu.psd_ffv.GPoreU_psd_ffv`, after `grids`:

1. **Sampling**: `N = psd_cycles` (default 10 000) probe centers,
   uniform in fractional coordinates, `rand @ cell` — seeded.
2. **Candidate compaction**: keep grid points with `~n_grid_occ` and
   `r > 0`; store indices in a compact `int32` list. Only these can
   cover a probe point with positive radius (a point whose exclusion
   zone contains p proves nothing positive; a largest covering sphere
   always has `r_g ≥ |p−g| > 0` so its center is a candidate).
3. **Fused 2D kernel search**: program grid (probe-chunk ×
   candidate-chunk); each program computes minimum-image |p − g| for
   its tile of pairs (fractional wrap — correct in tilted cells),
   accepts spheres covering the probe, reduces its tile max into the
   shared `pore_out[p]` with `atomic_max`. One launch, no
   Python-side loop; the sqrt form is preserved so boundary ties match
   the reference bit-for-bit.
4. **Histogramming**: nonzero diameters binned `0 : bin_size :
   max_pore` (0.25 Å / 20 Å defaults). `torch.histogram` runs on CPU
   because the CUDA backend rejects tensor bin edges.
5. **Distribution forms**: the *complementary cumulative* count
   `cumul[k] = Σ_{j≥k} hist[j]` is formed (flip/cumsum/flip),
   normalized to 1 at the first bin, and differentiated by a *central*
   difference

   ```
   psd[k] = −( norm_cumul[k+1] − norm_cumul[k−1] ) / (2·bin_size)
   ```

   which is smoother than forward differences and avoids doubling the
   endpoint spikes.

## Why performed this way

* **Insertion-point definition** ties the PSD to a measurable
  volume-weighted quantity, is probe-size explicit through the wall
  convention, and — unlike surface-integral definitions — needs no
  surface patching between neighbouring voxels.
* **Complementary cumulative form** is the standard MOF PSD plotting
  convention (PoreBlazer/PoroNet lineage): cumulative % of void volume
  above a given diameter, which reads directly as "fraction of the pore
  volume accessible to a sphere of size d". Differentiating it back is
  exactly what the differential PSD is.
* **Central differencing** of a discrete step-function CDF: forward
  differencing injects shot noise from bin-edge quantization and
  aliases spikes into the first bin; the central form is the standard
  remedy.
* **Compacted candidates**: the maximum of covering spheres is
  unaffected by removing non-covering candidates, yet the candidate set
  is normally 3–10× smaller than the grid — a free-order win on the
  dominant inner loop.
* **Probe sampling on free radius only where nonzero**: zero-radius
  candidates contribute nothing to a max over `r>0` outputs; excluding
  them makes `pore_out = 0` exactly equal the "no fitting sphere" case
  that FFV consumes ([ffv.md](ffv.md)), keeping the two stages
  definitionally consistent.

## Limitations

* Rigid framework; no probe relaxation: a sphere is *placed* at p, not
  allowed to shift to the local optimum (that would convert the
  insertion PSD into a local-maxima "cavity size" spectrum — the LCD-
  style view, see [lcd.md](lcd.md)).
* Statistical convergence at the bin level: 10⁴ draws spread over ~80
  bins give noisy tails; both `psd_cycles` and `psd_bin_size` are
  configurable.
* Surface roughness of real activated frameworks (gating windows that
  collapse on desolvation) cannot appear — intrinsic to CIF-based
  characterization.

## Ordered function calls

```
gporeu.utils.structure.prepare_input        (... as in porosity.md)
gporeu.grid.all_grids_pbc
  gporeu.grid.GPoreU_Lattice_Kernel
gporeu.psd_ffv.GPoreU_psd_ffv
  gporeu.psd_ffv.GPoreU_PSD_Kernel          # 2D probe x candidate fused kernel
gporeu.cli.format_summary
```
