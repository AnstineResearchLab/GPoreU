# GPoreU

GPU-accelerated pore characterization for porous crystals (MOFs and
related framework materials). From a CIF file, GPoreU computes:

* accessible **pore volumes** (helium soft probe + geometric) and
  gravimetric uptakes
* **percolation** flags / dimensionality for He, N2 and geometric probes
* **fractional free volume (FFV)** and **pore size distribution** (PSD)
* accessible **surface area** (Monte-Carlo SASA)
* **pore network**: H-maxima + watershed flooding, merged pore centers
* **PLD** (pore limiting diameter, per axis + connectivity dimension) and
  **LCD** (largest cavity diameter) via a winding-vector union-find on a
  Triton maximum spanning forest
* **CH4 interaction energy + gradient grids** with exact periodic image
  enumeration, and a 2D (energy, |gradient|) histogram

All stages run on CUDA (Triton kernels); a full characterization of a
~2k-atom supercell completes in well under a second on a modern GPU.

## Installation

Requires an NVIDIA GPU. Linux/x86 (Triton), Python >= 3.9, CUDA-enabled
PyTorch already installed or installable on your platform.

```bash
pip install .                    # console entry point: gporeu
# optional fallback CIF parser for symmetry-expanded CIFs:
pip install '.[cif-sym]'
```

> In an existing CUDA PyTorch environment whose Triton version has been
> set deliberately (e.g. a nightly pairing), install with
> `pip install --no-deps .`: dependency resolution otherwise re-locks
> Triton to the version pinned by the installed torch's metadata.
> GPoreU needs only `numpy` and CUDA-enabled `torch` (whose install
> itself provides a matching Triton).

## Command line

```bash
gporeu structure.cif                        # one structure
gporeu ./cifs/ -o results/                  # whole directory (batch)
gporeu ./cifs/ -m pld,porosity              # subset of methods
gporeu ./cifs/ -c settings.json --save-grids
gporeu --list-methods                       # methods + dependency graph
gporeu cif_files/IRMOF1.cif -m ehist --explain   # show resolved plan only
gporeu --init-config my_settings.json       # documented settings template
```

`INPUT` may mix CIF files and directories of CIF files; every structure is
processed independently (failures are reported, the batch continues).

Outputs per structure `NAME.cif` (in the output directory):

| file | contents |
|---|---|
| `NAME.summary` | formatted characterization report (also printed) |
| `NAME.grid`    | grid property arrays -- **only** with `--save-grids` (npz default, `--grid-format pt`, `compress` in settings) |

Grid arrays include: grid points, geo/He/N2 occupancies and access masks,
free-radius field, probe LJ energies, pore labels, pore-out field, CH4
energy/gradient fields when computed.

### Methods and automatic dependency resolution

User-facing methods are backed by pipeline steps; each method pulls in
exactly the steps it needs, executed in topological order:

```
grids -> connectivity -> psd        (pore size distribution / FFV)
      ->            -> network      (H-max + watershed pore network)
      ->            -> pld          (PLD + LCD, winding union-find)
      -> surface_area               (MC SASA)
      -> energy_grid -> energy_hist (CH4 grids; 2D histogram)
```

Names (aliases in brackets): `porosity`, `percolation`, `ffv`,
`pore_size_distribution [psd]`, `surface_area [sa]`, `pore_network
[poregraph, network]`, `pld`, `lcd`, `energy_grid [energy]`,
`energy_histogram [ehist, 2d_energy_histogram]`. Default: full
characterization. Example: `-m ehist` runs only
`grids -> energy_grid -> energy_hist`; the pore graph is never built.

### Settings file

`gporeu --init-config my_settings.json` writes a fully documented
template (`settings.example.json` is also shipped in the package).
It defines method selection, grid spacing, RNG seed, probe overrides
(He / N2 / CH4 sigma & epsilon), PSD cycles/bins, pore-network wall
thickness and safe buffer, energy `rcut`/`delta`/`images`, and output
behavior. Command-line flags override the file.

Important: `energy.images` must stay `true` when any cell width is below
`2 * rcut` (exact image enumeration; minimum-image-only is wrong there).

### Exit codes

`0` success, `1` at least one structure failed, `2` no CUDA GPU found.

## Programmatic use

```python
from gporeu.cli import run_one, DEFAULT_CONFIG, format_summary
from pathlib import Path

data, steps, why, timings = run_one(Path('IRMOF1.cif'), DEFAULT_CONFIG)
print(format_summary(data, Path('IRMOF1.cif'), DEFAULT_CONFIG,
                     DEFAULT_CONFIG['methods'], steps, timings))
```

`data` is the pipeline dict (all intermediate grids and results).
The legacy single-structure driver remains available as `gporeu.run`.

## Reproducibility & diagnostics

Stochastic stages (PSD sampling, surface-area Monte-Carlo) are seeded per
structure (`seed`, default 0). `GPOREU_VERBOSE=1` re-enables stage
diagnostics; `GPOREU_PROFILE=1` prints pore-graph section timings.

## Verification

This implementation was verified against PoreBlazer logs, an external 2D
energy-histogram reference, and PoroNet structural outputs (PLD agreement
<= 0.07 A vs PoreBlazer on the validated set; pointwise CH4 energies to
~1e-5 relative; acceptance suites green, including bit-identical kernel-
optimization gates). The full verification harness lives in the
development tree (`verification/`, REPORT_A_B.md, TOLERANCES.md).

## Tests

```bash
pytest tests/                      # plan/logic tests always run;
                                   # end-to-end GPU test needs CIFs:
GPOREU_TEST_CIF_DIR=/path/to/cifs pytest tests/test_end_to_end.py
```
