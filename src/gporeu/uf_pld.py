"""
uf_pld.py -- Pore Limiting Diameter by winding-vector union-find.

Replaces the cugraph maximum-spanning-tree PLD (see PLD_new_mst_legacy.py)
with a topologically exact GPU algorithm:

  A grid point is free for a probe of radius t when its radius r >= t.  A
  transport channel that survives to infinity under periodic continuation is
  precisely a non-contractible cycle of the free-volume lattice -- in the
  lifted (covering-space) picture, a cycle whose lifted lattice-index
  displacement  g != 0  ("winding vector").  Filtrating bonds in descending
  bottleneck radius w = min(r_u, r_v), the FIRST winding cycle along axis d
  appears at exactly the highest bottleneck over all d-winding cycles, so

      PLD_d = 2 * w_critical(d)

  is the diameter of the largest sphere with a periodic pathway along d.
  The overall PLD (PoreBlazer semantics: percolation along at least one
  direction) is max_d PLD_d.

Bond constraints: every bond (u -> v) along axis a demands P(v) - P(u) = +1
in lattice-index units, including wrap bonds oriented (n_a - 1) -> 0 by
percolate.periodic_ij_pairs.  The axis and sign are recovered from the
linear-index difference, so the bond set itself is inherited unchanged from
GPoreU_percolate (ij_geo + ij_geo_periodic).

Correctness (see verification/ in the dev tree):
  * synthetic battery (test_uf.py): defect cycles, torus rings, wrap cuts;
  * independent pure-Python lifted-space BFS on real grids at 9 thresholds;
  * PoreBlazer-log agreement within 0.07 A (exact-threshold to
    exact-threshold) on the validated set {WEI (triclinic, D=2), IRMOF1}.
"""

import time

import torch


class PotentialUnionFind:
    """Disjoint sets carrying integer potentials P(x) (lifted lattice index).

    Invariant: pot[x] = P(x) - P(parent(x)); find() returns
    (root, P(x) - P(root)) and path-compresses.  An edge processed inside one
    set reveals the cycle winding g = (P(v) - P(u)) - W; g != 0 marks a
    non-contractible cycle.
    """

    def __init__(self, n, device='cuda'):
        self.n = n
        self.parent = torch.arange(n, device=device, dtype=torch.int64)
        self.sz = torch.ones(n, device=device, dtype=torch.int64)
        self.pot = torch.zeros(n, 3, device=device, dtype=torch.int32)
        self.device = device

    def snapshot(self):
        return {'n': self.n,
                'parent': self.parent.clone(),
                'pot': self.pot.clone(),
                'sz': self.sz.clone()}

    # ------------------------------------------------------------ find -----
    def find(self, x):
        """Root of x plus accumulated potential P(x) - P(root)."""
        r = x
        acc = torch.zeros(x.numel(), 3, device=self.device, dtype=torch.int32)
        guard = 0
        limit = 64 + 8 * (self.n.bit_length())
        while True:
            p = self.parent[r]
            done = p == r
            if bool(done.all()):
                break
            guard += 1
            if guard > limit:
                raise RuntimeError('union-find parent cycle detected')
            acc += torch.where(done[:, None], torch.zeros_like(self.pot[r]),
                               self.pot[r])
            r = torch.where(done, r, p)
        self.parent[x] = r
        self.pot[x] = acc
        return r, acc

    # ------------------------------------------------------- kruskal round --
    def process_edges(self, u, v, W):
        """Batched Kruskal on edges with constraint P(v) - P(u) = W.

        Returns (events, n_unions); events are the winding vectors of all
        non-contractible cycles revealed by same-set edges.  Hook-to-larger
        claim acceptance keeps the parent forest acyclic within every round.
        """
        events = []
        n_unions = 0
        pending = torch.ones(u.numel(), dtype=torch.bool, device=self.device)
        max_rounds = 64 + 4 * int(torch.log2(torch.tensor(
            float(max(u.numel(), 2)))).item())
        for _ in range(max_rounds):
            if not bool(pending.any()):
                break
            iu = u[pending]
            iv = v[pending]
            Wp = W[pending]
            ru, pu = self.find(iu)
            rv, pv = self.find(iv)
            same = ru == rv
            if bool(same.any()):
                # winding residual g = (P(v)-P(u)) - W (root potentials cancel)
                g = pv[same] - pu[same] - Wp[same]
                nz = g.any(dim=1)
                if bool(nz.any()):
                    events.append(g[nz])
                idx = torch.nonzero(pending).squeeze(1)
                pending[idx[same]] = False
            cross = ~same
            if not bool(cross.any()):
                continue
            ruc, rvc, puc, pvc = ru[cross], rv[cross], pu[cross], pv[cross]
            Wc = Wp[cross]
            # child = smaller-index root (hook-to-larger): accepted claims
            # form increasing chains only, which cannot cycle and guarantee
            # steady progress; path compression is done in find()
            swap = ruc > rvc
            src = torch.where(swap, rvc, ruc)   # child root
            dst = torch.where(swap, ruc, rvc)   # parent root
            # attaching ruc -> rvc:  P(ruc) - P(rvc) = pv - pu - W
            # attaching rvc -> ruc:  P(rvc) - P(ruc) = pu - pv + W
            val_cc = pvc - puc - Wc
            val_opp = puc - pvc + Wc
            val = torch.where(swap[:, None], val_opp, val_cc)
            # deterministic conflict resolution: one claim per child root
            mult = int(u.numel()) + 1
            key = src * mult + torch.arange(
                src.numel(), device=self.device, dtype=torch.int64)
            order = torch.sort(key).indices
            ss, sd, sv = src[order], dst[order], val[order]
            uniq = torch.ones(ss.numel(), dtype=torch.bool, device=self.device)
            uniq[1:] = ss[1:] != ss[:-1]
            head = order[uniq]
            src_w, dst_w, val_w = src[head], dst[head], val[head]
            keep = src_w < dst_w
            src_w, dst_w, val_w = src_w[keep], dst_w[keep], val_w[keep]
            if src_w.numel() == 0:
                continue
            self.parent[src_w] = dst_w
            self.pot[src_w] = val_w
            self.sz.scatter_add_(0, dst_w, self.sz[src_w])
            n_unions += int(src_w.numel())
        else:
            if bool(pending.any()):
                raise RuntimeError('Kruskal stalled: '
                                   f'{int(pending.sum())} edges unresolved')
        if events:
            return torch.cat(events, 0), n_unions
        return torch.zeros(0, 3, device=self.device, dtype=torch.int32), \
            n_unions


# ---------------------------------------------------------------- builders --
def bonds_from_pairs(ij_geo, ij_geo_periodic, nx, ny, nz, device='cuda'):
    """Winding constraints for GPoreU's periodic pair lists.

    Interior pairs (percolate.ij_pairs) are oriented i -> i + e_axis and
    wrap pairs (percolate.periodic_ij_pairs) (n_axis - 1) -> 0; in the
    lifted lattice both advance the axis index by +1, hence W = +e_axis for
    both classes, recovered from the (signed) linear-index difference.
    """
    e_int = ij_geo.to(device=device, dtype=torch.int64)
    e_per = ij_geo_periodic.to(device=device, dtype=torch.int64)
    both = torch.cat([e_int, e_per], dim=0) if e_per.numel() else e_int
    # NOT [:, 0] views: those carry stride 2, which torch hides but raw
    # pointer kernels (Triton) silently misread.  Kernels need contiguous
    # storage.
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


# ---------------------------------------------------------------- driver ----
def _replay_has_axis(snap, u, v, W, lo, hi, axis, device):
    """Process edges [lo, hi) from snapshot; True iff winding cycle on axis."""
    uf = PotentialUnionFind(snap['n'], device=device)
    uf.parent.copy_(snap['parent'])
    uf.pot.copy_(snap['pot'])
    uf.sz.copy_(snap['sz'])
    if hi <= lo:
        return False, None
    events, _ = uf.process_edges(u[lo:hi], v[lo:hi], W[lo:hi])
    if events.numel() == 0:
        return False, None
    hit = events[events[:, axis] != 0]
    if hit.numel() == 0:
        return False, None
    return True, hit[0]


def pld_winding(radii, ij_geo, ij_geo_periodic, nx, ny, nz,
                block=None, verbose=False):
    """Per-axis PLD (diameter units) + winding diagnostics + LCD.

    radii          : flat float tensor [nx*ny*nz] of grid-point radii
    ij_geo(_periodic): bond lists from GPoreU_percolate (int32 [E,2])
    """
    t0 = time.perf_counter()
    device = radii.device
    radii = radii.double()
    u, v, W = bonds_from_pairs(ij_geo, ij_geo_periodic, nx, ny, nz,
                               device=device)
    E = u.numel()
    w = torch.minimum(radii[u], radii[v])
    order = torch.sort(w, descending=True).indices
    u, v, W, w = u[order], v[order], W[order], w[order]

    if block is None:
        block = 1 << max(15, min(18, int(E).bit_length() - 5))

    uf = PotentialUnionFind(nx * ny * nz, device=device)
    first = {d: None for d in (0, 1, 2)}     # axis -> (block_start, g)
    snaps = {}                               # axis -> (b0, snapshot)
    events_log = []

    b0 = 0
    while b0 < E:
        b1 = min(b0 + block, E)
        need_snap = any(first[d] is None for d in (0, 1, 2))
        snap = uf.snapshot() if need_snap else None
        events, _ = uf.process_edges(u[b0:b1], v[b0:b1], W[b0:b1])
        for d in (0, 1, 2):
            if first[d] is None and events.numel():
                hit = events[events[:, d] != 0]
                if hit.numel():
                    first[d] = (b0, hit[0])
                    snaps[d] = (b0, snap)
        if events.numel():
            for g in events[:8]:
                events_log.append((float(w[b1 - 1]), [int(x) for x in g]))
        if all(first[d] is not None for d in (0, 1, 2)):
            break
        b0 = b1

    result = {'n_edges': int(E), 'block': block}
    for d in (0, 1, 2):
        if first[d] is None:
            result[('pld_x', 'pld_y', 'pld_z')[d]] = 0.0
            result[('winding_x', 'winding_y', 'winding_z')[d]] = None
            continue
        bb0, snap = snaps[d]
        bb1 = min(bb0 + block, E)
        # stage 1: linear small-step scan from block start
        step = 2048
        lo = bb0
        found_g = None
        while lo < bb1:
            hi = min(lo + step, bb1)
            hit, g = _replay_has_axis(snap, u, v, W, bb0, hi, d, device)
            if hit:
                found_g = g
                bb1 = hi
                break
            lo = hi
        # stage 2: binary search smallest prefix with the event
        a, c = lo, bb1          # event exists at prefix c, not at a
        while c - a > 1:
            mid = (a + c) // 2
            hit, _ = _replay_has_axis(snap, u, v, W, bb0, mid, d, device)
            if hit:
                c = mid
            else:
                a = mid
        result[('pld_x', 'pld_y', 'pld_z')[d]] = 2.0 * float(w[c - 1])
        result[('winding_x', 'winding_y', 'winding_z')[d]] = \
            [int(x) for x in found_g]
        if verbose:
            print(f'  axis {d}: critical edge idx {c - 1} '
                  f'radius {float(w[c - 1]):.4f}')

    result['events'] = events_log
    result['time_s'] = time.perf_counter() - t0
    return result
