# 06 · Writing grid output for a small CIF

Grid arrays (occupancy masks, free-radius field, probe LJ energies, ...)
are normally kept in memory only.  `--save-grids` dumps them per
structure to `NAME.grid`, which is convenient for inspection, plotting
or downstream analysis in NumPy/PyTorch.

## Run

```bash
./save_grids.sh               # uses the bundled small cifs/WEI.cif
./save_grids.sh small.cif     # any small structure
```

equivalent to typing:

```bash
gporeu cifs/WEI.cif -m porosity,pld --save-grids -o out_grids
```

A small cell is recommended for this output: dump size grows with the
number of grid points (cell volume / spacing^3; WEI at the default
0.4 Å spacing is ~19k points ≈ a few MB, a large cubic MOF would be ~10x
that or more).

## What you get

`NAME.grid` is a numpy `.npz` by default (`--grid-format pt` produces a
`torch.save` payload instead).  It contains whichever arrays the
selected methods produced:

| array | meaning |
|---|---|
| `points` | Cartesian grid-point coordinates (N, 3) |
| `occ_geo`, `occ_helium`, `occ_nitrogen` | solid-occupancy masks per probe |
| `access_geo`, `access_helium`, `access_nitrogen` | accessibility masks |
| `free_radius` | distance to framework minus probe radius (the radius field all stages use) |
| `lj_helium_K`, `lj_nitrogen_K` | probe LJ energy fields (K) |
| `pore_out` | pore-out field (with `ffv`/`psd` selected) |
| `pore_labels`, `safe_mask` | watershed label / safe-core mask per point (with `pore_network`) |
| `ch4_energy_kJ_mol`, `ch4_gradient_kJ_mol_A` | CH4 energy/gradient fields (with `energy_grid`) |
| `grid_shape` | `(nx, ny, nz)` of the voxel grid |

## Sample output

```
--> WEI.grid written to out_grids/. Contents:
  points           (18750, 3)     float32
  occ_geo          (18750,)       bool
  occ_helium       (18750,)       bool
  occ_nitrogen     (18750,)       bool
  free_radius      (18750,)       float32
  lj_helium_K      (18750,)       float32
  lj_nitrogen_K    (18750,)       float32
  access_geo       (18750,)       bool
  access_helium    (18750,)       bool
  access_nitrogen  (18750,)       bool
  grid_shape       (3,)           int64

  free radius: min 0.00 A  max 1.29 A
```

From Python the same dump is `gporeu.cli.save_grid(data, path, cfg)`.

## Energy / gradient grid as XYZ (`--save-xyz`)

The CH4 energy and gradient fields can also be written as an extended
XYZ point set -- one line per grid point -- for direct visualization:

```bash
gporeu cifs/WEI.cif -m energy --save-xyz -o out_grids
```

`NAME.energy.xyz`:

```
18750
GPoreU CH4 energy grid: 18750 points; columns: energy_kJ_mol grad_energy_kJ_mol_A
X 0.0000 0.0000 0.0000 1595.45 7492.08
...
```

The dummy element `X` marks non-atomic sample points, and the two
trailing columns (`energy_kJ_mol`, `|grad energy|`) are read as the
occupancy / beta-factor by VMD (color by beta to paint the energy
landscape) and as named columns by OVITO's extended-XYZ reader.
