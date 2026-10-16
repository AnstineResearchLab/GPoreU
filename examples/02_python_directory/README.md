# 02 · A directory of CIFs, Python interface

Batch-characterize every CIF in a directory from Python — the same
mechanics that `gporeu DIR` uses internally.

## Run

```bash
python process_directory.py               # uses the bundled cifs/ (4 CIFs)
python process_directory.py /path/to/dir  # or any directory / mixed files
```

## What it demonstrates

- **`expand_inputs(paths)`** — turns a list of files and/or directories
  into a sorted, de-duplicated list of `.cif` paths (broken symlinks and
  non-regular entries are skipped).
- **Independent structures** — each file goes through `run_one()`
  separately inside `try/except`: one unreadable structure marks itself
  `FAILED` but the batch continues.
- **Selective processing** — the config used here requests only
  `porosity` and `pld` via

  ```python
  cfg = deep_merge(DEFAULT_CONFIG, {'methods': ['porosity', 'pld']})
  ```

  so the PSD / surface-area / pore-graph / energy stages never run,
  while their PLD dependencies (lattice grids, connectivity) are added
  automatically.
- Reports land in `./py_batch/NAME.summary`.

## Sample output

```
BHP               PLD   5.365 A   LCD   8.389 A   -> BHP.summary
STO               PLD   2.916 A   LCD   5.194 A   -> STO.summary
WEI               PLD   1.852 A   LCD   2.574 A   -> WEI.summary
ZIF8              PLD   2.549 A   LCD  11.421 A   -> ZIF8.summary

4 processed, 0 failed  (reports in py_batch/)
```
