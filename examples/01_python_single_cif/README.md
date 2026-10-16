# 01 · One CIF, Python interface

Characterize a single structure entirely from Python — no subprocess, no
CLI. Three calls do the whole job:

```python
from pathlib import Path
from gporeu.cli import DEFAULT_CONFIG, format_summary, run_one

data, steps, why, timings = run_one(Path('cifs/IRMOF1.cif'), DEFAULT_CONFIG)
print(format_summary(data, Path('cifs/IRMOF1.cif'), DEFAULT_CONFIG,
                     DEFAULT_CONFIG['methods'], steps, timings))
```

## Run

```bash
python process_single_cif.py            # uses the bundled cifs/IRMOF1.cif
python process_single_cif.py my_mof.cif # or any CIF of your own
```

## What happens

`run_one()` takes a CIF path and a config dict (here the full default set
of characterizations) and returns four things:

| return | contents |
|---|---|
| `data` | the pipeline dict: every intermediate grid and every result |
| `steps` | the steps actually executed, in order |
| `why` | per step, *why* it ran — requested directly, or pulled in as a dependency |
| `timings` | wall time per step in milliseconds |

`DEFAULT_CONFIG['methods']` is the full set (porosity, percolation, FFV,
PSD, surface area, pore network, PLD, energy histogram).  Each method
automatically pulls in the pipeline steps it needs, in dependency order —
you never schedule stages yourself.

## Sample output (tail, IRMOF1)

```
  PLD = 7.771 A   LCD = 15.045 A
  execution plan:
    grids             441.6 ms   (percolation + accessible volumes -> lattice grid construction)
    connectivity      106.5 ms   (method "porosity")
    psd                51.0 ms   (method "ffv")
    surface_area       33.3 ms   (method "surface_area")
    network            60.7 ms   (method "pore_network")
    pld                21.0 ms   (method "pld")
    energy_grid         7.5 ms   (2D energy histogram (E, |grad E|) -> CH4 energy + gradient grid)
    energy_hist         1.4 ms   (method "energy_histogram")
```

Numbers are in the pipeline dict for further use, e.g.
`data['pld']`, `data['lcd']`, `data['ffv']`, `data['total_surface_area']`,
`data['pore_labels']`.
