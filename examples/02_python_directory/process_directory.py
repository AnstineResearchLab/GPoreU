#!/usr/bin/env python
"""Process a whole directory of CIFs through the Python interface.

Replicates what `gporeu DIR` does: expand_inputs() turns files and/or
directories into a sorted, de-duplicated CIF list; each structure is
processed independently; and a failure in one file is reported without
aborting the batch.  Reports are written to ./py_batch/NAME.summary.

Usage:
    python process_directory.py [DIR_or_file ...]

Without arguments, uses the cifs/ directory shipped next to this script
(BHP, STO, WEI, ZIF8).
"""

import sys
from pathlib import Path

from gporeu.cli import (DEFAULT_CONFIG, deep_merge, expand_inputs,
                        format_summary, run_one)


def main():
    here = Path(__file__).resolve().parent

    # Files and/or directories; default: the bundled batch.
    paths = [Path(a) for a in sys.argv[1:]] or [here / 'cifs']

    # A lean settings override: only porosity and PLD are requested.
    # Their dependencies (grids, connectivity) are still pulled in
    # automatically, and everything else (PSD, SA, pore graph, energy)
    # is skipped.
    cfg = deep_merge(DEFAULT_CONFIG, {'methods': ['porosity', 'pld']})

    outdir = Path('py_batch')
    outdir.mkdir(exist_ok=True)

    cifs = expand_inputs(paths)
    failures = []

    for cif in cifs:
        try:
            data, steps, why, timings = run_one(cif, cfg)
        except Exception as exc:              # keep the batch alive
            failures.append(cif.name)
            print(f'{cif.stem:16s}  FAILED: {exc}')
            continue

        text = format_summary(data, cif, cfg, cfg['methods'], steps, timings)
        (outdir / f'{cif.stem}.summary').write_text(text)
        print(f'{cif.stem:16s}  PLD {float(data["pld"]):7.3f} A   '
              f'LCD {float(data["lcd"]):7.3f} A   -> {cif.stem}.summary')

    print(f'\n{len(cifs) - len(failures)} processed, '
          f'{len(failures)} failed  (reports in {outdir}/)')


if __name__ == '__main__':
    main()
