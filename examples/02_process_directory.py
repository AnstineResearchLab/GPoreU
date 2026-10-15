#!/usr/bin/env python
"""Process a directory of CIFs through the Python interface (batch).

Replicates what `gporeu DIR` does: expand_inputs() turns files and/or
directories into a sorted, de-duplicated list of CIF paths, each
structure is processed independently, and a failure in one file never
aborts the batch.  Summaries are written to ./py_batch/NAME.summary.

Usage:
    python 02_process_directory.py DIR [more_files_or_dirs ...]

Without an argument, uses $GPOREU_TEST_CIF_DIR (or ./cif_files).
"""

import os
import sys
from pathlib import Path

from gporeu.cli import (DEFAULT_CONFIG, deep_merge, expand_inputs,
                        format_summary, run_one)

OUTDIR = Path('py_batch')


def main():
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        paths = [Path(os.environ.get('GPOREU_TEST_CIF_DIR', './cif_files'))]

    # A lean settings override: porosity + PLD only.  Dependencies
    # (grids, connectivity) are still pulled in automatically.
    cfg = deep_merge(DEFAULT_CONFIG, {'methods': ['porosity', 'pld']})

    OUTDIR.mkdir(exist_ok=True)
    cifs = expand_inputs(paths)
    failures = []

    for cif in cifs:
        try:
            data, steps, why, timings = run_one(cif, cfg)
        except Exception as exc:          # keep the batch alive
            failures.append(cif.name)
            print(f'{cif.stem:16s}  FAILED: {exc}')
            continue

        text = format_summary(data, cif, cfg, cfg['methods'], steps, timings)
        (OUTDIR / f'{cif.stem}.summary').write_text(text)
        print(f'{cif.stem:16s}  PLD {float(data["pld"]):7.3f} A   '
              f'LCD {float(data["lcd"]):7.3f} A   -> {cif.stem}.summary')

    print(f'\n{len(cifs) - len(failures)} processed, '
          f'{len(failures)} failed  (summaries in {OUTDIR}/)')


if __name__ == '__main__':
    main()
