#!/usr/bin/env bash
# Writing grid outputs for a small CIF.
#
# By default GPoreU keeps all grid arrays in memory only and writes just
# NAME.summary.  --save-grids additionally dumps NAME.grid (numpy npz by
# default, torch pt with --grid-format pt) containing the grid point
# coordinates, occupancy/accessibility masks, the free-radius field,
# probe LJ energies and -- when those stages are selected -- pore labels
# and the CH4 energy/gradient fields, plus grid_shape = (nx, ny, nz).
#
# WEI (small unit cell) is used as the example structure.
#
# Usage:  ./save_grids.sh [small.cif]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CIF="${1:-$HERE/cifs/WEI.cif}"
OUT=out_grids

# PLD needs the lattice grids and connectivity, so those arrays get
# produced and dumped; keep it lean -- no energy grids requested.
gporeu "$CIF" -m porosity,pld --save-grids -o "$OUT"

NAME="$(basename "${CIF%.cif}")"
echo
echo "--> $NAME.grid written to $OUT/. Contents:"
python - "$OUT/$NAME.grid" <<'PY'
import sys
import numpy as np

with np.load(sys.argv[1]) as z:
    for k in z.files:
        a = z[k]
        print(f'  {k:16s} {str(a.shape):14s} {a.dtype}')
    r = z['free_radius']
    print(f'\n  free radius: min {r.min():.2f} A  max {r.max():.2f} A')
PY

echo
echo '# torch format instead (loads with torch.load):'
echo "#   gporeu \"\$CIF\" -m pld --save-grids --grid-format pt -o $OUT"

# The CH4 energy / |gradient| grid can also be dumped as extended XYZ
# for visualization (VMD/OVITO); requires the energy_grid method.
gporeu "$CIF" -m energy --save-xyz -o "$OUT"
echo
echo "--> $NAME.energy.xyz written to $OUT/ (first point):"
head -3 "$OUT/$NAME.energy.xyz"
