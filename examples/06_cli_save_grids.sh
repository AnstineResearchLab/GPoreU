#!/usr/bin/env bash
# Writing grid outputs for a small CIF.
#
# By default GPoreU keeps all grid arrays in memory only and writes just
# NAME.summary.  --save-grids additionally dumps NAME.grid (numpy npz by
# default, torch pt with --grid-format pt) containing the grid point
# coordinates, solid/occupancy masks, accessibility masks, free-radius
# field, probe LJ energies and (when computed) pore labels and CH4
# energy/gradient fields -- plus 'grid_shape' = (nx, ny, nz).
#
# WEI is used here as the small example cell.
#
# Usage:  ./06_cli_save_grids.sh [small.cif]
set -euo pipefail

CIF="${1:-${GPOREU_TEST_CIF_DIR:-./cif_files}/WEI.cif}"

# PLD needs the lattice grids and connectivity, so those arrays get
# produced and dumped; keep it lean -- no energy grids requested.
gporeu "$CIF" -m porosity,pld --save-grids -o out_grids

echo
echo "--> $(basename "${CIF%.cif}").grid written to out_grids/. Contents:"
python - "$(basename "${CIF%.cif}")" <<'PY'
import sys
import numpy as np

name = sys.argv[1]
with np.load(f'out_grids/{name}.grid') as z:
    for k in z.files:
        a = z[k]
        print(f'  {k:14s} {str(a.shape):14s} {a.dtype}')
    r = z['free_radius']
    print(f'\n  free radius: min {r.min():.2f} A  max {r.max():.2f} A')
PY

echo
echo '# torch format instead (needs no unpacking in Python):'
echo '#   gporeu "$CIF" -m pld --save-grids --grid-format pt -o out_grids'
