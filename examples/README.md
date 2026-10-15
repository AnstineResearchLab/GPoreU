# GPoreU examples

Six small, self-contained examples: two Python scripts, four shell
scripts, one settings file.  Each demo can also be typed by hand -- the
shell scripts are just grouped, commented commands.

**Requirements:** a CUDA GPU and `pip install .` of this repository (which
provides the `gporeu` command and the importable `gporeu` package).

**Test data:** the examples default to the CIFs in `$GPOREU_TEST_CIF_DIR`
(or `./cif_files`), matching this layout:

```
cif_files/
  WEI.cif  IRMOF1.cif  ZIF8.cif ...
```

Every example also takes an explicit file/directory argument instead:

```bash
./03_cli_single_file.sh /path/to/my_mof.cif
python 02_process_directory.py /path/to/dir_of_cifs
```

---

## 1 · One CIF, Python interface — `01_process_single_cif.py`

The whole workflow in three calls:

```python
from gporeu.cli import DEFAULT_CONFIG, format_summary, run_one

data, steps, why, timings = run_one(Path('IRMOF1.cif'), DEFAULT_CONFIG)
print(format_summary(data, Path('IRMOF1.cif'), DEFAULT_CONFIG,
                     DEFAULT_CONFIG['methods'], steps, timings))
```

* `run_one()` resolves the requested *methods* into pipeline *steps*,
  runs them in dependency order on the GPU, and returns everything.
* `data` is the pipeline dict -- every intermediate grid and result
  (e.g. `data['pld']`, `data['lcd']`, `data['ffv']`, `data['pore_labels']`).
* `why` explains each step (`method "porosity"`, or
  `2D energy histogram -> CH4 energy grid` when it was pulled in as a
  dependency).

Sample tail of the output (WEI):

```
  execution plan:
    grids             444.2 ms   (percolation + accessible volumes -> lattice grid construction)
    connectivity       89.5 ms   (method "porosity")
    psd                29.6 ms   (method "ffv")
    surface_area       28.1 ms   (method "surface_area")
    network            54.3 ms   (method "pore_network")
    pld                15.8 ms   (method "pld")
    energy_grid         9.1 ms   (2D energy histogram (E, |grad E|) -> CH4 energy + gradient grid)
    energy_hist         1.4 ms   (method "energy_histogram")
```

## 2 · A directory of CIFs, Python interface — `02_process_directory.py`

Batching without the CLI.  `expand_inputs()` turns files and directories
into a sorted, de-duplicated CIF list; the loop calls `run_one()` per
structure inside `try/except`, so one bad structure reports a failure
but never kills the batch — exactly what `gporeu DIR` does internally.
This example also shows a lean configuration via `deep_merge()`: only
`porosity` and `pld` are requested (their dependencies are still added
automatically), and reports go to `py_batch/NAME.summary`.

## 3 · One CIF, command line — `03_cli_single_file.sh`

```bash
gporeu IRMOF1.cif -o out_single
```

Full characterization; report to `out_single/IRMOF1.summary` (also
printed).  Exit code `0` success, `1` if any structure failed, `2` no
CUDA GPU.

## 4 · A directory, command line — `04_cli_directory.sh`

```bash
gporeu /path/to/cifs/ -o out_directory
```

Every CIF below the path is characterized independently; positional
arguments may mix files and directories.

## 5 · Turning calculations on/off — `05_cli_turn_off_calcs.sh` + `05_turn_off_calcs_settings.json`

Control which characterizations run at *method* granularity
(`gporeu --list-methods` for the table and aliases).  Two equivalent
ways:

```bash
# only PLD: PSA, PSD, FFV, pore network and energy grids are skipped
gporeu IRMOF1.cif -m pld

# the same choice in a settings file
gporeu IRMOF1.cif -c 05_turn_off_calcs_settings.json
```

* The settings file's `"methods"` list is the master switch (see the
  JSON's `_turned_off` comment for what disabling buys you).
* Command-line flags always override the file, so `-m` wins over
  `"methods"`.
* Add `--explain` to preview the resolved step plan without running.
  For instance, a run with only `energy_histogram` selected expands to
  `grids -> energy_grid -> energy_hist`; the pore graph is never built.
  Dependencies are added *only* as far as they are needed.

## 6 · Writing grid output for a small CIF — `06_cli_save_grids.sh`

Grid arrays are in-memory by default (only `NAME.summary` is written).
With `--save-grids` each structure also gets `NAME.grid`:

```bash
gporeu WEI.cif -m porosity,pld --save-grids -o out_grids
```

`NAME.grid` is a numpy `.npz` (use `--grid-format pt` for `torch.save`)
containing whichever arrays the selected methods produced — grid point
coordinates, geometric/He/N2 occupancy and accessibility masks, the
free-radius field, probe LJ energies, pore-out/pore-label fields
(network selected) and the CH4 energy/gradient fields (energy
selected) — plus `grid_shape = (nx, ny, nz)`.  The example loads the
file and prints every array name, shape, dtype and the free-radius
range.  (Same arrays from Python: `gporeu.cli.save_grid(data, path, cfg)`.)

---

### Which script for what

| goal | script |
|---|---|
| characterize one CIF from Python | `01_process_single_cif.py` |
| batch a folder from Python | `02_process_directory.py` |
| one CIF on the command line | `03_cli_single_file.sh` |
| a folder on the command line | `04_cli_directory.sh` |
| disable specific calculations | `05_cli_turn_off_calcs.sh` (also the `.json`) |
| dump grids for a small CIF | `06_cli_save_grids.sh` |