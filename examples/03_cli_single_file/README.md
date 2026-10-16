# 03 · A single CIF, command line

The simplest use of the `gporeu` command: one structure, all default
characterizations (porosity, percolation, FFV, PSD, surface area, pore
network, PLD/LCD, CH4 2D energy histogram).

## Run

```bash
./run_single.sh                # uses the bundled cifs/IRMOF1.cif
./run_single.sh my_mof.cif     # or any CIF of your own
```

equivalent to typing:

```bash
gporeu cifs/IRMOF1.cif -o out_single
```

## What happens

- The full report is printed and written to `out_single/IRMOF1.summary`.
- No grid arrays are written (in-memory only) — see example 06 for that.
- Exit codes: `0` success, `1` at least one structure failed,
  `2` no CUDA GPU found.

## Sample output

```
--- [1/1] IRMOF1.cif --------------------------------------------
====================================================================
 GPoreU characterization summary - IRMOF1.cif
====================================================================
  atoms            : 424
  cell             : a=25.832  b=25.832  c=25.832  alpha=90.00 beta=90.00 gamma=90.00
  cell volume      : 17237.5 A^3     density: 0.593 g/cm^3
  grid             : 65 x 65 x 65 @ 0.400 A (274625 points)
--------------------------------------------------------------------
  pore volume (He, probe 2.58 A) :   14231.62 A^3    1.392 cm^3/g
  pore volume (geometric)               :   13794.83 A^3    1.349 cm^3/g
  ...
  PLD                :      7.771 A (dimension D3)
  LCD                :     15.045 A
  pipeline (auto-resolved): grids -> connectivity -> psd -> surface_area -> network -> pld -> energy_grid -> energy_hist
```

Useful flags: `-m pld,lcd` (select methods, example 05),
`-g 0.5` (grid spacing in Å), `-s 1` (RNG seed), `--save-grids`
(example 06), `-c settings.json` (full JSON control), `--list-methods`,
`-h`.
