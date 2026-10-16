#!/usr/bin/env python
"""Process a single CIF through the GPoreU Python interface.

Runs the full auto-resolved characterization pipeline on one structure,
prints the report that the CLI would write to NAME.summary, then shows
the raw numbers, the execution plan and per-step timings.

Usage:
    python process_single_cif.py [structure.cif]

Without an argument, uses cifs/IRMOF1.cif shipped next to this script.
"""

import sys
from pathlib import Path

from gporeu.cli import DEFAULT_CONFIG, format_summary, run_one


def main():
    here = Path(__file__).resolve().parent
    cif = Path(sys.argv[1]) if len(sys.argv) > 1 else here / 'cifs' / 'IRMOF1.cif'

    # run_one() resolves the requested methods into pipeline steps,
    # executes them in dependency order on the GPU, and returns:
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
