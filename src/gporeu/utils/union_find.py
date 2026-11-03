"""
GPU union-find primitives for the lattice graph (Triton).

    components(N, u, v) -> int64 root per node (connected components)
    max_st(N, u, v, w)  -> (src, dst, w) links of a maximum spanning
                           forest (one link per non-root node)

Both use a branch-free fixed-hop find and the provably cycle-free
min->max CAS linking invariant: for connected components, a plain
per-edge CAS on slot c = min(ru, rv) suffices (the CAS winner makes
guaranteed progress; losers remain crossing edges next round), while
max_st adds a deterministic ballot election to pick one winning edge
per component pair.  Termination for both is the exact predicate
"zero crossing edges".
"""

import torch
import triton
import triton.language as tl


# -------------------------------------------------------------- CC paths ---
# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["E", "maxh"])
def _cc_merge_kernel(u_ptr, v_ptr, par_ptr, prog_ptr, ofl_ptr, E, maxh,
                     BLOCK: tl.constexpr):
    """Find both endpoint roots; crossing edges link min-root under
    max-root via CAS.  Counts crossings (termination) and successes."""
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < E

    ru = tl.load(u_ptr + i, mask=m, other=0)
    rv = tl.load(v_ptr + i, mask=m, other=0)

    au0 = tl.zeros([BLOCK], dtype=tl.int32)   # finished-lane flags
    av0 = ~m
    for _hop in range(maxh):
        pru = tl.load(par_ptr + ru, mask=m, other=0)
        prv = tl.load(par_ptr + rv, mask=m, other=0)
        du = (pru == ru) | (~m)
        dv = (prv == rv) | (~m)
        ru = tl.where(du | au0, ru, pru)
        rv = tl.where(dv | av0, rv, prv)
        au0 = au0 | du
        av0 = av0 | dv

    tl.atomic_max(ofl_ptr, tl.max(tl.where((~au0 | ~av0) & m, 1, 0).to(tl.int32),
                                  axis=0))

    # Act only on lanes whose walks actually terminated this launch;
    # truncated walks would fabricate crossings and stale links.
    conv = au0 & av0 & m
    cross = conv & (ru != rv)
    tl.atomic_add(prog_ptr + 1, tl.sum(cross.to(tl.int32), axis=0))

    c = tl.minimum(ru, rv)
    p = tl.maximum(ru, rv)
    # CAS slot c: par[c] = p.  Non-crossing lanes CAS slot 0 against -1
    # (provably fails: every root != -1): Triton atomics take no mask.
    tgt = tl.where(cross, c, 0)
    cmpv = tl.where(cross, c, -1)
    old = tl.atomic_cas(par_ptr + tgt, cmpv, p)
    ok = cross & (old == c)
    tl.atomic_add(prog_ptr, tl.sum(ok.to(tl.int32), axis=0))


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["N", "maxh"])
def _cc_final_kernel(par_ptr, root_ptr, N, maxh, ofl_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    r = i
    done = ~m
    for _hop in range(maxh):
        p = tl.load(par_ptr + r, mask=m, other=0)
        d = (p == r) | (~m)
        r = tl.where(d | done, r, p)
        done = done | d
    tl.atomic_max(ofl_ptr, tl.max(tl.where(~done & m, 1, 0).to(tl.int32),
                                  axis=0))
    tl.store(root_ptr + i, r, mask=m)



# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["N"])
def _split_kernel(par_ptr, N, BLOCK: tl.constexpr):
    """One path-splitting pass: par[i] <- par[par[i]].  Ancestor pointers
    stay valid ancestors so the forest is preserved, but tree depth is
    halved, which keeps the branchless walks in merge/find cheap.

    Connected components only: max_st recovers its spanning forest from
    the parent array and therefore requires exact parents -- splitting
    there would yield grandparent shortcut links, not graph edges.
    """
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    p = tl.load(par_ptr + i, mask=m, other=0)
    q = tl.load(par_ptr + p, mask=m, other=0)
    tl.store(par_ptr + i, q, mask=m)


def components(N, u, v, max_rounds=200):
    """Connected-component roots for graph (N nodes, edge arrays u,v)."""
    device = u.device
    u = u.contiguous().to(torch.int32)
    v = v.contiguous().to(torch.int32)
    E = u.numel()
    par = torch.arange(N, device=device, dtype=torch.int32)
    BLOCK = 1024
    grid = (triton.cdiv(E, BLOCK),) if E else (1,)

    prog = torch.zeros(3, device=device, dtype=torch.int32)  # merges,cross,ofl
    gridn = (triton.cdiv(N, BLOCK),)

    rounds = 0
    while True:
        _split_kernel[gridn](par, N, BLOCK=BLOCK, num_warps=4)
        hcap = 16
        while True:
            # fresh counters per launch: only a fully-converged launch has
            # meaningful crossing/merge totals (the termination predicate).
            prog.zero_()
            _cc_merge_kernel[grid](u, v, par, prog, prog[2:], E, hcap,
                                   BLOCK=BLOCK, num_warps=4)
            ctr = prog.cpu()          # single host sync per attempt
            if int(ctr[2]) == 0:
                break
            hcap *= 4
            if hcap > 8192:
                raise RuntimeError('chain overflow in cc merge find')
        nprog, ncross = int(ctr[0]), int(ctr[1])
        if ncross == 0:
            break
        rounds += 1
        if nprog == 0:
            raise RuntimeError(f'CC stalled with {ncross} crossings')
        if rounds >= max_rounds:
            raise RuntimeError('CC exceeded max rounds')

    root = torch.empty(N, device=device, dtype=torch.int64)
    hcap = 16
    ofl = torch.zeros(1, device=device, dtype=torch.int32)
    while True:
        ofl.zero_()
        _cc_final_kernel[gridn](par, root, N, hcap, ofl, BLOCK=BLOCK,
                                num_warps=4)
        if int(ofl[0]) == 0:
            break
        hcap *= 4
        if hcap > 8192:
            raise RuntimeError('chain overflow in cc finalize')
    return root


# ------------------------------------------------------------ max-ST path ---
# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["E", "maxh"])
def _ms_find(u_ptr, v_ptr, par_ptr, ru_ptr, rv_ptr, E, maxh, ofl_ptr,
             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < E
    ru = tl.load(u_ptr + i, mask=m, other=0)
    rv = tl.load(v_ptr + i, mask=m, other=0)
    au0 = ~m
    av0 = ~m
    for _hop in range(maxh):
        pru = tl.load(par_ptr + ru, mask=m, other=0)
        prv = tl.load(par_ptr + rv, mask=m, other=0)
        du = (pru == ru) | (~m)
        dv = (prv == rv) | (~m)
        ru = tl.where(du | au0, ru, pru)
        rv = tl.where(dv | av0, rv, prv)
        au0 = au0 | du
        av0 = av0 | dv
    tl.atomic_max(ofl_ptr, tl.max(tl.where((~au0 | ~av0) & m, 1, 0).to(tl.int32),
                                  axis=0))
    tl.store(ru_ptr + i, ru, mask=m)
    tl.store(rv_ptr + i, rv, mask=m)


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["E"])
def _ms_best(ru_ptr, rv_ptr, w_ptr, best_ptr, cnt_ptr, E, BLOCK: tl.constexpr):
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
    pack = tl.where(cross, pack, -1)
    tl.atomic_max(best_ptr + ru, pack)
    tl.atomic_max(best_ptr + rv, pack)


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["N"])
def _ms_ballot(best_ptr, ru_ptr, rv_ptr, ballot_ptr, N, BLOCK: tl.constexpr):
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


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["N", "E", "wshift"])
def _ms_hook(par_ptr, best_ptr, ballot_ptr, ru_ptr, rv_ptr, prog_ptr,
             ew_ptr, wshift, N, E, BLOCK: tl.constexpr):
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

    tgt = tl.where(won, c, 0)
    cmpv = tl.where(won, c, -1)
    old = tl.atomic_cas(par_ptr + tgt, cmpv, p)
    tok = won & (old == c)
    bad = tl.sum((won & ~tok).to(tl.int32), axis=0)
    tl.atomic_add(prog_ptr, tl.sum(tok.to(tl.int32), axis=0))
    if bad != 0:
        tl.atomic_add(prog_ptr + 1, bad)

    wbits = (b >> 32).to(tl.int32)
    w = wbits.to(tl.float32, bitcast=True) + wshift
    tl.store(ew_ptr + c, w, mask=tok)


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["N"])
def _ms_edges(par_ptr, ew_ptr, src_ptr, dst_ptr, w_ptr, nout_ptr, N,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < N
    p = tl.load(par_ptr + r, mask=m, other=0)
    is_edge = m & (p != r)
    k = tl.atomic_add(nout_ptr, tl.sum(is_edge.to(tl.int32), axis=0))
    # exclusive scan within this block
    cum = tl.cumsum(is_edge.to(tl.int32), axis=0) - 1
    pos = k + cum
    pos = tl.where(is_edge, pos, 0)
    tl.store(src_ptr + pos, r, mask=is_edge)
    tl.store(dst_ptr + pos, p, mask=is_edge)
    tl.store(w_ptr + pos, tl.load(ew_ptr + r, mask=is_edge, other=0.0),
             mask=is_edge)


def max_st(N, u, v, w, max_rounds=200):
    """Maximum spanning forest via Boruvka; returns (src, dst, w) where the
    edge arrays carry the ACTUAL crossing-edge weights used for each link."""
    device = u.device
    u = u.contiguous().to(torch.int32)
    v = v.contiguous().to(torch.int32)
    w = w.contiguous().float()
    # The best-edge pack needs nonnegative weights: float32 bit patterns of
    # negative weights have the sign bit set, which flips the packed key's
    # signed order and stalls Boruvka.  Ranking on a shifted copy preserves
    # the weight order exactly (IEEE subtraction is monotone); _ms_hook
    # un-shifts the output weights, which stay exact when none was needed.
    E = u.numel()
    wshift = float(w.min()) if E else 0.0
    if wshift < 0.0:
        wsort = w - wshift
    else:
        wshift = 0.0
        wsort = w

    par = torch.arange(N, device=device, dtype=torch.int32)
    ew = torch.zeros(N, device=device, dtype=torch.float32)
    ru = torch.empty(E, device=device, dtype=torch.int32)
    rv = torch.empty(E, device=device, dtype=torch.int32)
    best = torch.empty(N, device=device, dtype=torch.int64)
    ballot = torch.empty(N, device=device, dtype=torch.int64)
    cnt = torch.zeros(1, device=device, dtype=torch.int32)
    prog = torch.zeros(2, device=device, dtype=torch.int32)
    ofl = torch.zeros(1, device=device, dtype=torch.int32)

    BLOCK = 512
    grid = (triton.cdiv(E, BLOCK),)
    gridn = (triton.cdiv(N, BLOCK),)

    hcap = 16
    rounds = 0
    while True:
        while True:
            ofl.zero_()
            _ms_find[grid](u, v, par, ru, rv, E, hcap, ofl, BLOCK=BLOCK,
                           num_warps=4)
            if int(ofl[0]) == 0:
                break
            hcap *= 4
            if hcap > 8192:
                raise RuntimeError('chain overflow in maxST find')
        cnt.zero_()
        best.fill_(-1)
        _ms_best[grid](ru, rv, wsort, best, cnt, E, BLOCK=BLOCK, num_warps=4)
        if int(cnt[0]) == 0:
            break
        ballot.fill_(-1)
        _ms_ballot[gridn](best, ru, rv, ballot, N, BLOCK=BLOCK, num_warps=4)
        prog.zero_()
        _ms_hook[gridn](par, best, ballot, ru, rv, prog, ew, wshift, N, E,
                        BLOCK=BLOCK, num_warps=4)
        merges, failed = int(prog[0]), int(prog[1])
        if failed:
            raise RuntimeError(f'{failed} ballot winners lost their CAS')
        if merges == 0:
            raise RuntimeError('maxST Boruvka stalled')
        rounds += 1
        if rounds >= max_rounds:
            raise RuntimeError('maxST exceeded max rounds')

    src = torch.empty(N, device=device, dtype=torch.int32)
    dst = torch.empty(N, device=device, dtype=torch.int32)
    we = torch.empty(N, device=device, dtype=torch.float32)
    nout = torch.zeros(1, device=device, dtype=torch.int32)
    _ms_edges[gridn](par, ew, src, dst, we, nout, N, BLOCK=BLOCK,
                     num_warps=4)
    k = int(nout[0])
    return src[:k].long(), dst[:k].long(), we[:k]
