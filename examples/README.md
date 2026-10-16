# GPoreU examples

Six self-contained examples, one directory each.  Every directory ships
its own README and the CIF file(s) it needs under `cifs/`, so you can run
anything immediately:

```
examples/
├── 01_python_single_cif/       one CIF via the Python interface
├── 02_python_directory/        a directory of CIFs via Python (batch)
├── 03_cli_single_file/         one CIF on the command line
├── 04_cli_directory/           a directory on the command line
├── 05_turn_off_calculations/   disabling specific characterizations
└── 06_save_grids/              dumping grid arrays for a small CIF
```

**Requirements:** a CUDA GPU and this repository installed (`pip install .`),
which provides the importable `gporeu` package and the `gporeu` command.

## Quick reference

| goal | directory | key ingredients |
|---|---|---|
| characterize one CIF from Python | `01_python_single_cif/` | `run_one()`, `format_summary()`, pipeline dict |
| batch a folder from Python | `02_python_directory/` | `expand_inputs()`, per-file error handling, config override |
| one CIF on the command line | `03_cli_single_file/` | `gporeu FILE -o OUT` |
| a folder on the command line | `04_cli_directory/` | `gporeu DIR -o OUT` |
| disable specific calculations | `05_turn_off_calculations/` | `-m/--methods`, JSON `"methods"`, `--explain` |
| dump grids for a small CIF | `06_save_grids/` | `--save-grids [--grid-format npz|pt]`, npz inspection |

Each directory is meant to be read on its own; open its `README.md` for
the walkthrough and sample output.

## Overview of the three knobs

- **What to compute** — the *methods* list (`-m` or `"methods"` in a
  settings JSON; `gporeu --list-methods`, `gporeu --init-config FILE`).
  Selected methods automatically pull in their pipeline dependencies
  and nothing else runs.
- **How to compute it** — settings JSON (`-c`): grid spacing, RNG seed,
  probe overrides, PSD sampling, pore-network buffers, energy cutoff and
  image enumeration, output behavior.  Command-line flags
  (`-g`, `-s`, ...) override the file.
- **What to store** — reports are always written (`NAME.summary`); grid
  arrays only with `--save-grids` (`NAME.grid`), format chosen with
  `--grid-format`.

Run any script directly (`./run_single.sh`, `python process_directory.py`,
...) to use its bundled CIFs, or pass your own files/directories as
arguments.
