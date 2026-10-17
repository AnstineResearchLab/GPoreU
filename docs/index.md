# GPoreU Method Documentation

Detailed expositions of the scientific methods behind every GPoreU
characterization output: the theory, the exact routines (constants,
thresholds, kernels) as implemented, the justification for each design
choice, and an ordered list of the GPoreU functions each method calls.

## Documents

| method (CLI name) | document | one-line description |
|---|---|---|
| foundations | [voxelization.md](voxelization.md) | probe fields on the voxel lattice: free radius, geometric / He / N2 occupancy |
| `porosity` | [porosity.md](porosity.md) | helium and geometric accessible pore volumes |
| `percolation` | [percolation.md](percolation.md) | direction-resolved percolation flags and accessible-channel topology |
| `ffv` | [ffv.md](ffv.md) | fractional free volume for an N2-type probe |
| `pore_size_distribution` (`psd`) | [pore_size_distribution.md](pore_size_distribution.md) | PSD from random probe-point sampling |
| `surface_area` (`sa`) | [surface_area.md](surface_area.md) | Monte-Carlo accessible surface area |
| `pore_network` (`poregraph`, `network`) | [pore_network.md](pore_network.md) | pore centers, watershed basins, periodic pore merging |
| `pld` | [pld.md](pld.md) | pore limiting diameter via winding-vector union-find |
| `lcd` | [lcd.md](lcd.md) | largest cavity diameter |
| `energy_grid` (`energy`) | [energy_grid.md](energy_grid.md) | CH4 interaction-energy and gradient fields |
| `energy_histogram` (`ehist`) | [energy_histogram.md](energy_histogram.md) | 2D (energy, \|gradient\|) histogram |
| `free_energy_histogram` (`fehist`) | [free_energy_histogram.md](free_energy_histogram.md) | accessible-point-restricted 2D histogram with neighbour-gated finite-difference gradient |

Each document is self-contained: repetition of shared background is
intentional so that any method can be read in isolation.

---

## Pipeline architecture (shared background)

GPoreU characterizes a crystalline porous material (a MOF, zeolite, CSP,
...) from its **CIF** file plus a periodic lattice. Every method builds on
a shared discretization:

1. **`grids`** — the unit cell is filled with a regular voxel grid
   (default spacing 0.4 Å). For every grid point, GPoreU evaluates,
   against all framework atoms under minimum-image periodicity:
   - `grid_radaii[g]` — the *free radius*: distance from the grid point
     to the nearest atom **surface** (center distance minus σ/2);
   - three occupancy masks (geometric, helium, N2) — hard exclusion
     spheres per probe convention (see [voxelization.md](voxelization.md));
   - soft LJ energy fields for the He and N2 probes (Boltzmann weights).
2. **`connectivity`** — adjacent free grid points are bonded (6-
   neighbour lattice, including bonds across periodic faces), bonded
   points are grouped by connected components (union-find), and
   opposite-face cluster matching yields per-axis percolation flags and
   accessibility masks (`he_access`, `n_access`, `geo_access`).
3. The individual methods then run on top of these fields; the CLI
   resolves which of them execute from the requested method list, e.g.
   `energy_histogram` pulls in `grids -> energy_grid -> energy_hist` and
   never touches the connectivity stage.

### Conventions used in every document

* **Lennard-Jones parameters.** Framework atom parameters are UFF σ (Å)
  and ε (converted internally to *temperature units*, K:
  `ε[K] = ε[kcal/mol] / 1.987e-3`). Probe parameters (defaults):
  | probe | σ (Å) | ε (K) | role |
  |---|---|---|---|
  | He | 2.58 | 10.22 | free-volume / accessible-volume probe |
  | N2 | 3.314 | 10.22 | PSD / FFV / surface-area probe |
  | CH4 | 3.73 | 148.0 | the energetic probe (storage applications) |
  Pair parameters follow **Lorentz–Berthelot**:
  σ_mix = ½(σ_i + σ_p), ε_mix = √(ε_i·ε_p), 12-6 form.
* **Periodicity.** Everything is evaluated in the unit cell with
  three-periodic boundary conditions, using
  fractional wrapping `f ← f − floor(f + 0.5)` (minimum image) or
  explicit image enumeration where required (`energy_grid`).
* **Diameters vs radii.** All reported pore sizes are *diameters*; the
  internal radius field is `grid_radaii`.
* **Grid discretization error.** Every voxel method is accurate to
  roughly one grid spacing; finer spacing costs memory as O(Δ⁻³) and
  time as O(N_atoms · Δ⁻³).
* **Determinism.** Stochastic stages (PSD sampling, surface-area
  Monte-Carlo) are seeded (`seed`, default 0); identical inputs and seed
  give bit-identical outputs.

### Function-call spine common to all methods

Every characterization begins with the same preparation, performed by
`gporeu.cli.run_one`:

```
gporeu.cli.run_one
└── gporeu.utils.structure.prepare_input        # CIF -> pipeline dict
    ├── gporeu.utils.structure.read_cif         # fast P1 parser
    │   ├── gporeu.utils.structure._fast_read_cif
    │   └── (fallback) pybel + gporeu.utils.structure.fill_uc
    ├── gporeu.utils.structure.init_grid        # voxel lattice (nx,ny,nz),
    │                                           #  trials = 512 x N_atoms
    └── gporeu.utils.force_fields.UFF.forward   # per-atom sigma/epsilon
        gporeu.utils.force_fields.MASS.forward  # framework mass
```

Method-specific chains, in execution order, are listed at the end of
each document.
