"""
Pore Limiting Diameter (PLD) and largest cavity diameter (LCD).

The PLD is computed with a winding-vector union-find: an exact maximum
spanning forest of the grid radius field is built with parallel Boruvka
steps, and winding residuals are then scored once:

    PLD_d = 2 * max{ w_e : g_e[d] != 0 },
    g_e   = P(v) - P(u) - W_e      (potentials over the maxST)

i.e. twice the highest bottleneck radius that admits a non-contractible
cycle of free volume along lattice axis d -- the exact "largest sphere
with a periodic transport pathway" criterion.  This stage consumes the
pipeline fields of the grid and connectivity stages (``grid_radaii``,
``ij_geo``, ``ij_geo_periodic``, ``geo_grid_occ``, ``nx``, ``ny``,
``nz``) and returns per-axis diameters, the overall PLD (PoreBlazer
semantics: percolation along at least one direction), LCD, per-axis
winding vectors and a connectivity dimension count.

Why this equals the sequential-Kruskal first-event PLD:
  (>=) A maximum spanning tree is a widest-path structure, so the cycle
       formed by a non-tree edge e has every tree edge >= w_e; g_e != 0
       makes it a genuine winding cycle at threshold w_e.
  (<=) Any winding cycle C at threshold t decomposes into fundamental
       cycles of non-tree edges contained in C (cycle space; g is
       additive over homology), so winding(C) != 0 forces some e in C
       with g_e != 0 and w_e >= t.

Boruvka phase (five kernels, one tiny host sync per phase):

  1. _find_kernel     per-edge find IN-KERNEL.  Branch-free fixed-hop
                      walk (finished lanes self-load stable entries), so
                      there is no while-loop condition, no cross-warp
                      reduction, and identical semantics at any
                      BLOCK/num_warps.  A chain longer than `maxh` hops
                      sets an overflow flag the driver turns into a
                      budget doubling and retry.
  2. _best_kernel     scatter each edge's packed (weight_bits, edge_id)
                      via atomic_max into BOTH endpoint roots: the
                      heaviest crossing edge per component.  Also counts
                      crossing edges (the exact termination predicate).
  3. _ballot_kernel   every component r holding a best edge claims the
                      CHILD SLOT c = min(r, partner) with the value
                      (weight_bits << 32) | r via atomic_max.  Highest
                      weight wins ties by component id, so exactly one
                      claimant per slot, deterministically.
  4. _hook_kernel     the ballot winner performs par[c]: c -> p (p = max
                      of the pair) with a CAS (guaranteed to succeed:
                      c was a root at find time and only its ballot
                      winner touches par[c]), storing the potential
                      constraint of ITS edge.  Every ballot winner is a
                      guaranteed merge, so components contract fast.
                      Edges that lose a ballot are re-offered next phase
                      as long as they still cross; the min->max
                      convention makes concurrent hooks structurally
                      incapable of forming parent cycles.

Termination: zero crossing edges in phase 2's counter.
"""

import time

import torch
import triton
import triton.language as tl

from .utils import prof

@triton.jit
def _find_kernel(u_ptr, v_ptr, par_ptr, pot_ptr,
                 ru_ptr, rv_ptr, pu_ptr, pv_ptr,
                 E, nstride, maxh, ofl_ptr, BLOCK: tl.constexpr):
    """Roots and potentials for every edge endpoint.

    Branch-free fixed-hop walk: a lane that reached its root keeps
    self-loading the (stable) identity entry while tl.where freezes its
    state.  Sets *ofl_ptr if any chain outlives maxh.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < E

    ru = tl.load(u_ptr + i, mask=m, other=0)
    rv = tl.load(v_ptr + i, mask=m, other=0)
    au0 = tl.zeros([BLOCK], dtype=tl.int32)
    au1 = tl.zeros([BLOCK], dtype=tl.int32)
    au2 = tl.zeros([BLOCK], dtype=tl.int32)
    av0 = tl.zeros([BLOCK], dtype=tl.int32)
    av1 = tl.zeros([BLOCK], dtype=tl.int32)
    av2 = tl.zeros([BLOCK], dtype=tl.int32)
    du0 = ~m
    dv0 = ~m

    for _hop in range(maxh):
        pru = tl.load(par_ptr + ru, mask=m, other=0)
        prv = tl.load(par_ptr + rv, mask=m, other=0)
        du = (pru == ru) | (~m)
        dv = (prv == rv) | (~m)
        au0 = tl.where(du | du0, au0,
                       au0 + tl.load(pot_ptr + ru, mask=m, other=0))
        au1 = tl.where(du | du0, au1,
                       au1 + tl.load(pot_ptr + nstride + ru, mask=m,
                                     other=0))
        au2 = tl.where(du | du0, au2,
                       au2 + tl.load(pot_ptr + 2 * nstride + ru, mask=m,
                                     other=0))
        av0 = tl.where(dv | dv0, av0,
                       av0 + tl.load(pot_ptr + rv, mask=m, other=0))
        av1 = tl.where(dv | dv0, av1,
                       av1 + tl.load(pot_ptr + nstride + rv, mask=m,
                                     other=0))
        av2 = tl.where(dv | dv0, av2,
                       av2 + tl.load(pot_ptr + 2 * nstride + rv, mask=m,
                                     other=0))
        ru = tl.where(du | du0, ru, pru)
        rv = tl.where(dv | dv0, rv, prv)
        du0 = du0 | du
        dv0 = dv0 | dv

    ofl = tl.max(tl.where((~du0 | ~dv0) & m, 1, 0).to(tl.int32), axis=0)
    tl.atomic_max(ofl_ptr, ofl)

    tl.store(ru_ptr + i, ru, mask=m)
    tl.store(rv_ptr + i, rv, mask=m)
    tl.store(pu_ptr + i, au0, mask=m)
    tl.store(pu_ptr + E + i, au1, mask=m)
    tl.store(pu_ptr + 2 * E + i, au2, mask=m)
    tl.store(pv_ptr + i, av0, mask=m)
    tl.store(pv_ptr + E + i, av1, mask=m)
    tl.store(pv_ptr + 2 * E + i, av2, mask=m)


@triton.jit
def _best_kernel(ru_ptr, rv_ptr, w_ptr, best_ptr, cnt_ptr, E,
                 BLOCK: tl.constexpr):
    """Heaviest crossing edge per component; count crossing edges."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < E
    ru = tl.load(ru_ptr + i, mask=m, other=0)
    rv = tl.load(rv_ptr + i, mask=m, other=0)
    w = tl.load(w_ptr + i, mask=m, other=0.0)
    cross = m & (ru != rv)
    tl.atomic_add(cnt_ptr, tl.sum(cross.to(tl.int32), axis=0))
    bits = w.to(tl.uint32, bitcast=True).to(tl.int64)
    pack = (bits << 32) | i.to(tl.int64)
    # weights are provably >= 0 (accessible points), so signed ordering of
    # `pack` matches weight ordering. Neutral lanes contribute -1: a no-op.
    pack = tl.where(cross, pack, -1)
    tl.atomic_max(best_ptr + ru, pack)
    tl.atomic_max(best_ptr + rv, pack)


@triton.jit
def _ballot_kernel(best_ptr, ru_ptr, rv_ptr, ballot_ptr, N,
                   BLOCK: tl.constexpr):
    """Claim child slot c=min(r, partner) with (weight_bits << 32) | r."""
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < N
    b = tl.load(best_ptr + r, mask=m, other=-1)
    has = b >= 0
    e = (b & 0xFFFFFFFF).to(tl.int32)
    e = tl.where(has, e, 0)
    eru = tl.load(ru_ptr + e, mask=has, other=0)
    erv = tl.load(rv_ptr + e, mask=has, other=0)
    partner = eru + erv - r
    c = tl.where(has, tl.minimum(r, partner), 0)
    claim = (b >> 32 << 32) | r.to(tl.int64)
    claim = tl.where(has, claim, -1)
    tl.atomic_max(ballot_ptr + c, claim)


@triton.jit
def _hook_kernel(par_ptr, pot_ptr, best_ptr, ballot_ptr, prog_ptr,
                 W_ptr, ru_ptr, rv_ptr, pu_ptr, pv_ptr,
                 N, E, nstride, BLOCK: tl.constexpr):
    """Ballot winners link their pair: par[c] = p with potential value."""
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < N
    b = tl.load(best_ptr + r, mask=m, other=-1)
    has = b >= 0
    e = (b & 0xFFFFFFFF).to(tl.int32)
    e = tl.where(has, e, 0)
    eru = tl.load(ru_ptr + e, mask=has, other=0)
    erv = tl.load(rv_ptr + e, mask=has, other=0)
    partner = eru + erv - r
    c = tl.where(has, tl.minimum(r, partner), 0)
    p = tl.where(has, tl.maximum(r, partner), 0)

    claim = (b >> 32 << 32) | r.to(tl.int64)
    won = has & (tl.load(ballot_ptr + c, mask=has, other=-1) == claim)

    # potential constraint for attaching c under p using this edge:
    # c == eru  ->  P(ru)-P(rv) = pv - pu - W ;  else reversed
    pu0 = tl.load(pu_ptr + e, mask=won, other=0)
    pu1 = tl.load(pu_ptr + E + e, mask=won, other=0)
    pu2 = tl.load(pu_ptr + 2 * E + e, mask=won, other=0)
    pv0 = tl.load(pv_ptr + e, mask=won, other=0)
    pv1 = tl.load(pv_ptr + E + e, mask=won, other=0)
    pv2 = tl.load(pv_ptr + 2 * E + e, mask=won, other=0)
    w0 = tl.load(W_ptr + e, mask=won, other=0)
    w1 = tl.load(W_ptr + E + e, mask=won, other=0)
    w2 = tl.load(W_ptr + 2 * E + e, mask=won, other=0)
    c_is_ru = eru <= erv
    v0k = tl.where(c_is_ru, pv0 - pu0 - w0, pu0 - pv0 + w0)
    v1k = tl.where(c_is_ru, pv1 - pu1 - w1, pu1 - pv1 + w1)
    v2k = tl.where(c_is_ru, pv2 - pu2 - w2, pu2 - pv2 + w2)

    # CAS guaranteed to succeed (c is a root and its unique winner).
    # Non-winners CAS slot 0 with an unmatchable comparison (always
    # fails, never writes): Triton atomic_cas has no mask argument.
    tgt = tl.where(won, c, 0)
    cmpv = tl.where(won, c, -1)
    old = tl.atomic_cas(par_ptr + tgt, cmpv, p)
    tok = won & (old == c)
    bad = tl.sum((won & ~tok).to(tl.int32), axis=0)
    tl.atomic_add(prog_ptr, tl.sum(tok.to(tl.int32), axis=0))
    if bad != 0:
        tl.atomic_add(prog_ptr + 1, bad)

    tl.store(pot_ptr + c, v0k, mask=tok)
    tl.store(pot_ptr + nstride + c, v1k, mask=tok)
    tl.store(pot_ptr + 2 * nstride + c, v2k, mask=tok)


@triton.jit
def _score_kernel(pu_ptr, pv_ptr, W_ptr, g_ptr, E, BLOCK: tl.constexpr):
    """g = (P(v) - P(u)) - W for every edge (endpoints already at roots)."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < E
    for k in tl.static_range(3):
        pu = tl.load(pu_ptr + k * E + i, mask=m, other=0)
        pv = tl.load(pv_ptr + k * E + i, mask=m, other=0)
        w = tl.load(W_ptr + k * E + i, mask=m, other=0)
        tl.store(g_ptr + k * E + i, pv - pu - w, mask=m)



def bonds_from_pairs(ij_geo, ij_geo_periodic, nx, ny, nz, device='cuda'):
    """Winding constraints for the periodic pair lists.

    Interior pairs (percolate.ij_pairs) are oriented i -> i + e_axis and
    wrap pairs (percolate.periodic_ij_pairs) (n_axis - 1) -> 0; in the
    lifted lattice both advance the axis index by +1, hence W = +e_axis
    for both classes, recovered from the (signed) linear-index difference.
    """
    e_int = ij_geo.to(device=device, dtype=torch.int64)
    e_per = ij_geo_periodic.to(device=device, dtype=torch.int64)
    both = torch.cat([e_int, e_per], dim=0) if e_per.numel() else e_int
    # The kernels take raw pointers, so require contiguous storage:
    # plain [:, 0] views carry stride 2 and would be misread.
    u, v = both[:, 0].contiguous(), both[:, 1].contiguous()
    d = v - u
    sx, sy, sz = ny * nz, nz, 1
    W = torch.zeros(u.numel(), 3, device=device, dtype=torch.int32)
    m0 = (d == sx) | (d == -(nx - 1) * sx)
    m1 = (d == sy) | (d == -(ny - 1) * sy)
    m2 = (d == sz) | (d == -(nz - 1) * sz)
    W[m0, 0] = 1
    W[m1, 1] = 1
    W[m2, 2] = 1
    if not bool((m0 | m1 | m2).all()):
        raise RuntimeError(
            'bonds_from_pairs: unexpected index difference (pairs must be '
            '6-neighbour lattice bonds)')
    return u, v, W


# ---------------------------------------------------------------- driver ---
def pld_winding_fast(radii, ij_geo, ij_geo_periodic, nx, ny, nz,
                     max_rounds=None, verbose=False):
    """Return pld_x/pld_y/pld_z, winding_x/y/z and bookkeeping fields."""

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    device = radii.device
    radii = radii.float()

    u, v, W = bonds_from_pairs(ij_geo, ij_geo_periodic, nx, ny, nz,
                               device=device)
    # Triton kernels take raw pointers: refuse strided inputs loudly rather
    # than misreading memory.
    assert u.is_contiguous() and v.is_contiguous() and W.is_contiguous()
    E = u.numel()
    w = torch.minimum(radii[u], radii[v])

    # W as [3, E] planes for coalesced kernel access
    Wp = torch.empty(3, E, device=device, dtype=torch.int32)
    Wp[0] = W[:, 0]
    Wp[1] = W[:, 1]
    Wp[2] = W[:, 2]
    del W

    n = nx * ny * nz
    par = torch.arange(n, device=device, dtype=torch.int64)
    pot = torch.zeros(3, n, device=device, dtype=torch.int32)
    ru = torch.empty(E, device=device, dtype=torch.int64)
    rv = torch.empty(E, device=device, dtype=torch.int64)
    pu = torch.empty(3, E, device=device, dtype=torch.int32)
    pv = torch.empty(3, E, device=device, dtype=torch.int32)
    best = torch.empty(n, device=device, dtype=torch.int64)
    ballot = torch.empty(n, device=device, dtype=torch.int64)
    cnt = torch.zeros(1, device=device, dtype=torch.int32)
    prog = torch.zeros(2, device=device, dtype=torch.int32)
    ofl = torch.zeros(1, device=device, dtype=torch.int32)

    BLOCK = 512
    grid = (triton.cdiv(E, BLOCK),)
    gridn = (triton.cdiv(n, BLOCK),)

    torch.cuda.synchronize()
    t_prep = time.perf_counter()

    if max_rounds is None:
        max_rounds = 200
    rounds = 0

    def _find():
        # Adaptive hop budget: chains grow <= 1 link/phase, so 16 is usually
        # plenty; the overflow flag doubles the budget and retries.
        nonlocal hcap
        while True:
            ofl.zero_()
            _find_kernel[grid](u, v, par, pot, ru, rv, pu, pv, E, n, hcap,
                               ofl, BLOCK=BLOCK, num_warps=4)
            if int(ofl[0]) == 0:
                return
            hcap *= 4
            if hcap > 8192:
                raise RuntimeError('find chain overflow in Boruvka forest')

    hcap = 16
    while True:
        _find()
        cnt.zero_()
        best.fill_(-1)
        _best_kernel[grid](ru, rv, w, best, cnt, E, BLOCK=BLOCK,
                           num_warps=4)
        ncross = int(cnt[0])
        if ncross == 0:
            break
        ballot.fill_(-1)
        _ballot_kernel[gridn](best, ru, rv, ballot, n, BLOCK=BLOCK,
                              num_warps=4)
        prog.zero_()
        _hook_kernel[gridn](par, pot, best, ballot, prog, Wp, ru, rv, pu,
                            pv, n, E, n, BLOCK=BLOCK, num_warps=4)
        merges, failed = int(prog[0]), int(prog[1])
        rounds += 1
        if failed:
            raise RuntimeError(f'{failed} ballot winners lost their CAS '
                               '(forest invariant violated)')
        if merges == 0:
            raise RuntimeError(f'Boruvka stalled: {ncross} crossings but no '
                               f'merge after {rounds} rounds')
        if rounds >= max_rounds:
            raise RuntimeError(f'Boruvka PLD exceeded {max_rounds} rounds')
    torch.cuda.synchronize()
    t_kruskal = time.perf_counter()

    # final positions + winding residuals
    _find()
    g = torch.empty(3, E, device=device, dtype=torch.int32)
    _score_kernel[grid](pu, pv, Wp, g, E, BLOCK=BLOCK, num_warps=4)
    torch.cuda.synchronize()
    t_score = time.perf_counter()

    result = {'n_edges': int(E), 'events': []}
    for k, name in enumerate(('x', 'y', 'z')):
        mk = g[k] != 0
        if bool(mk.any()):
            ws = torch.where(mk, w, torch.full_like(w, -1.0))
            j = int(torch.argmax(ws))
            result[f'pld_{name}'] = 2.0 * float(ws[j])
            result[f'winding_{name}'] = [int(g[0, j]), int(g[1, j]),
                                         int(g[2, j])]
        else:
            result[f'pld_{name}'] = 0.0
            result[f'winding_{name}'] = None

    result['rounds'] = rounds
    result['time_s'] = t_score - t0
    if verbose:
        prof.vprint(f'  triton: rounds={rounds} '
                    f'prep={1000*(t_prep-t0):.1f}ms '
                    f'boruvka={1000*(t_kruskal-t_prep):.1f}ms '
                    f'score={1000*(t_score-t_kruskal):.1f}ms')
    return result


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
    res = pld_winding_fast(grid_radaii,
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
