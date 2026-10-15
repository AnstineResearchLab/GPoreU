"""
Pore Limiting Diameter (PLD) and largest cavity diameter (LCD).

The PLD is computed with the winding-vector union-find of
:mod:`gporeu.winding_pld`:

    PLD_d = 2 x (highest bottleneck radius admitting a non-contractible
                 cycle of free volume along lattice axis d)

which is the exact "largest sphere with a periodic transport pathway"
criterion.  It consumes the pipeline fields produced by the grid and
connectivity stages (``grid_radaii``, ``ij_geo``, ``ij_geo_periodic``,
``geo_grid_occ``, ``nx``, ``ny``, ``nz``) and returns per-axis
diameters, the overall PLD (PoreBlazer semantics: percolation along at
least one direction), LCD, winding vectors and a connectivity
dimension count.
"""

import time

import torch

from .winding_pld import pld_winding_fast as pld_winding


def calculate_pld(_input):
    """Per-axis winding PLD for the GPoreU pipeline dict.

    Returns pld_x/pld_y/pld_z (diameter units, 0.0 when no channel winds
    along that axis), their max as 'pld' (PoreBlazer semantics: percolation
    along at least one direction), 'lcd', per-axis winding vectors, and the
    percolation-style dimension count 'pld_dimension'.
    """
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    # ---------- INPUTS ----------
    grid_radaii = _input['grid_radaii']

    nx = _input['nx']
    ny = _input['ny']
    nz = _input['nz']

    # ---------- WINDING UNION-FIND ----------
    res = pld_winding(grid_radaii,
                      _input['ij_geo'],
                      _input['ij_geo_periodic'],
                      nx, ny, nz,
                      verbose=True)

    pld_x = res['pld_x']
    pld_y = res['pld_y']
    pld_z = res['pld_z']

    # ---------- OVERALL PLD (percolation along any axis) ----------
    per_axis = [p for p in (pld_x, pld_y, pld_z) if p > 0.0]
    pld = max(per_axis) if per_axis else 0.0
    # PoreBlazer-style dimension: axes still percolating AT the overall PLD
    dimension = sum(1 for p in per_axis if p >= pld - 1e-9)

    # ---------- LARGE CAVE DIAMETER ----------
    lcd = 2.0 * float(grid_radaii[~_input['geo_grid_occ']].max())

    torch.cuda.synchronize()
    total_end = time.perf_counter()

    print("==========================================")
    print("FINAL PLD RESULTS (winding union-find)")
    print("==========================================")
    print("bonds processed:", res['n_edges'])
    print("PLD X:", pld_x, ' winding:', res['winding_x'])
    print("PLD Y:", pld_y, ' winding:', res['winding_y'])
    print("PLD Z:", pld_z, ' winding:', res['winding_z'])
    print("PLD:", pld, " (dimension", dimension, ")")
    print("LCD:", lcd)
    print("TOTAL PLD TIME:", (total_end - total_start) * 1000, "ms")
    print("==========================================")

    return {
        'pld_x': pld_x,
        'pld_y': pld_y,
        'pld_z': pld_z,
        'pld': pld,
        'lcd': lcd,
        'pld_dimension': dimension,
        'winding_x': res['winding_x'],
        'winding_y': res['winding_y'],
        'winding_z': res['winding_z'],
    }
