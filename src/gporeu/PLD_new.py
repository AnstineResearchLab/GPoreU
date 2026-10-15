"""
PLD_new.py -- Pore Limiting Diameter (production entry point).

The historical two-MST algorithm (no periodic bonds in the spanning forest,
face-to-face widest path, and a -1.0-weight closing edge that clamped the
reported PLD to 2.0 A whenever the true bottleneck radius exceeded 1 A) has
been replaced by the winding-vector union-find of uf_pld.py, executed on
the GPU via the Triton kernels of uf_pld_triton.py (exact parallel-Boruvka
maximum spanning forest + one-pass winding scoring; 25-165x faster than the
vectorized reference, bit-identical results):

  PLD_d = 2 x (highest bottleneck radius admitting a non-contractible cycle
               of free volume along lattice axis d)

which is the exact "largest sphere with an infinite periodic pathway"
criterion.  It consumes the same GPoreU pipeline fields as before
(grid_radaii, ij_geo, ij_geo_periodic, geo_grid_occ, nx, ny, nz) and
returns the same keys (pld_x, pld_y, pld_z) plus extras; see
PLD_new_mst_legacy.py for the replaced implementation.

Validation (verification/ in the dev tree): PoreBlazer-log agreement within
0.07 A threshold-to-threshold on the validated set {WEI (tilted triclinic,
D=2), IRMOF1 (cubic, D=3)}; LCD within 0.15 A; regression battery
test_uf.py green, including an independent lifted-space BFS oracle.
"""

import time

import torch

from .uf_pld_triton import pld_winding_fast as pld_winding


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
