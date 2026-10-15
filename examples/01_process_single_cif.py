#!/usr/bin/env python
"""Process a single CIF through the Python interface.

Runs the full auto-resolved characterization pipeline on one structure
and prints the same report the CLI writes to NAME.summary, then shows
how to reach into the pipeline dict for raw numbers and per-step
timings/reasons.

Usage:
    python 01_process_single_cif.py [structure.cif]

Without an argument, uses WEI.cif from $GPOREU_TEST_CIF_DIR (or ./cif_files).
"""

import os
import sys
from pathlib import Path

from gporeu.cli import DEFAULT_CONFIG, format_summary, run_one


def main():
    if len(sys.argv) > 1:
        cif = Path(sys.argv[1])
    else:
        root = Path(os.environ.get('GPOREU_TEST_CIF_DIR', './cif_files'))
        cif = root / 'WEI.cif'

    # run_one() resolves the method list into pipeline steps, executes
    # them in dependency order and returns:
    #   data    -- the pipeline dict (grids + every stage result)
    #   steps   -- actual execution order
    #   why     -- why each step ran (requested / pulled in by which one)
    #   timings -- wall time per step, milliseconds
    data, steps, why, timings = run_one(cif, DEFAULT_CONFIG)

    print(format_summary(data, cif, DEFAULT_CONFIG,
                         DEFAULT_CONFIG['methods'], steps, timings))

    # Raw results live in the pipeline dict:
    print(f'  PLD = {float(data["pld"]):.3f} A   '
          f'LCD = {float(data["lcd"]):.3f} A')

    # Execution plan with millisecond timings:
    print('  execution plan:')
    for step in steps:
        print(f'    {step:14s} {timings[step]:8.1f} ms   ({why[step]})')


if __name__ == '__main__':
    main()
