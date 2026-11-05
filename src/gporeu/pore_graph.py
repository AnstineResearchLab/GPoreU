"""
Pore-graph construction and void assignment.

Pore centers are detected as H-maxima of the grid radius field: a
maximum spanning forest of the grid graph (edge weight min(r_i, r_j))
supports a reconstruction pass that propagates water levels, and local
maxima that keep prominence ``h`` after reconstruction are significant
centers.  Adjacent centers are merged (union-find over the
maxima-induced subgraph, retaining the widest vertex of each cluster).

Each center then holds a safe core radius bounded by its clearance to
every other center minus a wall-thickness buffer; voxels claimed by
exactly one safe core are seeded, and the remaining void voxels are
assigned by widest-path flooding (Triton sweeps along the six axis
directions).  Pores that merge across periodic walls are detected and
consolidated, and final centers are relabeled.
"""

import math

import torch
import triton
import triton.language as tl

from .utils import prof
from .utils.union_find import components, max_st

def GPoreU_apply_pbc(pore_labels, merged_maxima_ids, grid_radaii,
                     wall_thickness, grid_spacing, cell):
        """Merge pore labels/centers identified across periodic boundaries,
            following the PoroNet grouping convention.

        For every axis the wall slab of ``wall_thickness`` (measured with
        the PERPENDICULAR cell length like PoroNet, so tilted cells get
        the right number of planes) is MIRRORED against the opposite
        wall, so plane i pairs with its wrap-neighbour plane n-1-i.  A
        pair of regions co-occurring across such a wall pair is accepted
        when more than half of either region's wall pixels find the same
        partner on the other side AND the maxima positions look like one
        region viewed through the boundary: both maxima inside the facing
        walls, or exactly one inside and that region is the smaller of
        the two (a cap split off a big region is absorbed by it).
        Accepted pairs are unioned into connected components (PBC
        groups).  One center survives per component - the widest center,
        smallest label on radius ties - voxels are relabeled 1..K, and
        the PoroNet counting convention holds: maxima before PBC,
        connected components of those regions after PBC.
        """

        import numpy as np
        prof.mark('  pbc:enter')

        nx, ny, nz = pore_labels.shape
        device = pore_labels.device
        dims = np.array((nx, ny, nz), dtype=np.int64)

        # Wall thickness in planes per axis, from the face-to-face
        # (perpendicular) cell lengths, exactly as PoroNet computes them.
        cell_np = cell.detach().cpu().numpy().astype(np.float64)
        eA, eB, eC = cell_np
        vol = abs(float(np.dot(eA, np.cross(eB, eC))))
        lperp = np.array([vol / np.linalg.norm(np.cross(eB, eC)),
                          vol / np.linalg.norm(np.cross(eC, eA)),
                          vol / np.linalg.norm(np.cross(eA, eB))])
        wt = float(wall_thickness.item() if torch.is_tensor(wall_thickness)
                   else wall_thickness)
        wall_cells = np.clip(np.ceil(wt / lperp * dims).astype(np.int64),
                             1, dims - 1)
        prof.mark('  pbc:prepared')

        lab_np = pore_labels.detach().cpu().numpy().reshape(-1)
        rad_np = grid_radaii.reshape(-1).detach().cpu().numpy()
        merged_ids_np = merged_maxima_ids.detach().cpu().numpy().astype(np.int64)

        center_labels = lab_np[merged_ids_np]
        center_radii = rad_np[merged_ids_np]
        center_pos = np.stack((merged_ids_np // (ny * nz),
                               (merged_ids_np % (ny * nz)) // nz,
                               merged_ids_np % nz), axis=1)

        order = np.argsort(center_labels, kind='stable')
        center_labels = center_labels[order]
        center_radii = center_radii[order]
        center_pos = center_pos[order]
        merged_maxima_np = merged_ids_np[order]
        prof.mark('  pbc:cpu-copied')

        accepted = []
        for a in range(3):
            n = int(dims[a])
            w = int(wall_cells[a])

            left = np.take(lab_np.reshape(nx, ny, nz),
                           np.arange(0, w), axis=a).reshape(-1)
            right = np.flip(np.take(lab_np.reshape(nx, ny, nz),
                                    np.arange(n - w, n), axis=a),
                            axis=a).reshape(-1)

            lmax = int(max(left.max(initial=0), right.max(initial=0))) + 1
            keys = left.astype(np.int64) * lmax + right.astype(np.int64)
            ukeys, pair_counts = np.unique(keys, return_counts=True)
            pairs = np.stack((ukeys // lmax, ukeys % lmax), axis=1)
            valid = (pairs[:, 0] > 0) & (pairs[:, 1] > 0) \
                  & (pairs[:, 0] != pairs[:, 1])
            pairs, pair_counts = pairs[valid], pair_counts[valid]
            if pairs.shape[0] == 0:
                continue

            # Two-sided overlap fractions; keep co-occurrences where
            # either direction exceeds one half (PoroNet pick_these).
            lvals, ltot = np.unique(left, return_counts=True)
            rvals, rtot = np.unique(right, return_counts=True)
            lfrac = (pair_counts / ltot[np.searchsorted(lvals, pairs[:, 0])])
            rfrac = (pair_counts / rtot[np.searchsorted(rvals, pairs[:, 1])])
            keep = (lfrac > 0.5) | (rfrac > 0.5)
            pairs = pairs[keep]
            if pairs.shape[0] == 0:
                continue

            # Maxima-position acceptance: both inside the facing walls,
            # or one-sided with the one-sided region the smaller cap.
            li = np.searchsorted(center_labels, pairs[:, 0])
            ri = np.searchsorted(center_labels, pairs[:, 1])
            fL = center_pos[li, a] < w
            fR = center_pos[ri, a] > n - w
            both = fL & fR
            xor = fL ^ fR
            one_rad = np.where(fL, center_radii[li], center_radii[ri])
            oth_rad = np.where(fL, center_radii[ri], center_radii[li])
            absorb = xor & (one_rad < oth_rad)
            keep = both | absorb
            pairs = pairs[keep]
            if pairs.shape[0] > 0:
                accepted.append(pairs)

        prof.mark('  pbc:walls')

        if accepted:
            edges = np.unique(np.concatenate(accepted, axis=0), axis=0)
        else:
            edges = np.zeros((0, 2), dtype=np.int64)

        # Connected components over region labels (PBC groups).
        maxlab = int(center_labels.max(initial=0))
        uf = np.arange(maxlab + 1)

        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        for p0, p1 in edges:
            r0, r1 = find(int(p0)), find(int(p1))
            if r0 != r1:
                uf[r0] = r1

        groups = np.array([find(int(l)) for l in center_labels], dtype=np.int64)

        # Group survivor: maximum radius, smallest label on ties.
        leader = center_labels.copy()
        gs = np.lexsort((center_labels, groups))
        i = 0
        while i < gs.size:
            j = i + 1
            while j < gs.size and groups[gs[j]] == groups[gs[i]]:
                j += 1
            seg = gs[i:j]
            winner = seg[np.lexsort((center_labels[seg],
                                     -center_radii[seg]))[0]]
            leader[seg] = center_labels[winner]
            i = j
        merged_labels = leader
        prof.mark('  pbc:merge-loop')

        surviving_labels = np.unique(merged_labels)
        keep_centers = np.isin(center_labels, surviving_labels)
        merged_maxima_np = merged_maxima_np[keep_centers]

        final_labels = np.searchsorted(surviving_labels, merged_labels) + 1
        periodic_label_map_np = np.zeros(maxlab + 1, dtype=np.int64)
        periodic_label_map_np[center_labels] = final_labels
        pore_labels_np = periodic_label_map_np[lab_np]
        prof.mark('  pbc:renumbered')

        pore_labels = torch.from_numpy(
            np.ascontiguousarray(pore_labels_np.reshape(nx, ny, nz))).to(device)
        merged_maxima_ids = torch.from_numpy(
            np.ascontiguousarray(merged_maxima_np)).to(device)
        periodic_label_map = torch.from_numpy(
            np.ascontiguousarray(periodic_label_map_np)).to(device)
        periodic_pairs = torch.from_numpy(np.ascontiguousarray(edges)).to(device)
        prof.mark('  pbc:returned')
        return pore_labels, merged_maxima_ids, periodic_label_map, periodic_pairs

# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["NGRID", "num_pores"])
def GPoreU_SafeRegion_Kernel(grid, merged_maxima_ids, safe_radaii, geo_grid_occ, pore_labels, safe_mask, NGRID, num_pores, BLOCKSIZE: tl.constexpr):

        """Kernel: assign voxels claimed by exactly one safe center
        sphere.
        """
        # Identify the voxels in this block.
        pid = tl.program_id(0)
        offsets = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
        valid = offsets < NGRID


# Load voxel coordinates
        x = tl.load(grid + offsets * 3, mask=valid, other=0).to(tl.float32)
        y = tl.load(grid + offsets * 3 + 1, mask=valid, other=0).to(tl.float32)
        z = tl.load(grid + offsets * 3 + 2, mask=valid, other=0).to(tl.float32)


# Check whether voxels are void
        occupied = tl.load(geo_grid_occ + offsets, mask=valid, other=1)
        void = valid & (occupied == 0)


# Start with no claims or labels
        claim_counts = tl.full((BLOCKSIZE,), 0, tl.int32)
        labels = tl.full((BLOCKSIZE,), 0, tl.int32)


# Check each pore's safe sphere
        for pore in range(num_pores):

                center_id = tl.load(merged_maxima_ids + pore)

                center_x = tl.load(grid + center_id * 3).to(tl.float32)
                center_y = tl.load(grid + center_id * 3 + 1).to(tl.float32)
                center_z = tl.load(grid + center_id * 3 + 2).to(tl.float32)

                radius = tl.load(safe_radaii + pore).to(tl.float32)

                dx = x - center_x
                dy = y - center_y
                dz = z - center_z

                distance_squared = dx * dx + dy * dy + dz * dz
                inside = void & (radius > 0) & (distance_squared < radius * radius)

                claim_counts = claim_counts + inside.to(tl.int32)
                labels = tl.where(inside, pore + 1, labels)


# Assign only voxels claimed by exactly one pore
        safe = void & (claim_counts == 1)
        labels = tl.where(safe, labels, 0)

        tl.store(pore_labels + offsets, labels, mask=valid)
        tl.store(safe_mask + offsets, safe, mask=valid)

# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["nx", "ny", "nz", "NGRID"])
def GPoreU_Flood_Sweep_Kernel(
        cap_src, lbl_src, cap_dst, lbl_dst, rad_ptr, comp_ptr, flag_ptr,
        nx, ny, nz, NGRID, BLOCKSIZE: tl.constexpr):
    """One synchronous competitive-flooding sweep over the whole grid.

    Six directed neighborhoods are voted in order (x+, x-, y+, y-, z+,
    z-); proposals are accepted only for competitive voxels with a
    labeled neighbor; higher capacity wins, ties prefer a smaller
    (zero-favoring) label; out-of-bounds neighbor capacity is -inf.
    Reads only from the *_src arrays (double buffered) so every vote
    sees the previous sweep.  Sets *flag_ptr if anything changed.
    """
    pid = tl.program_id(0)
    v = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    m = v < NGRID

    rad_v = tl.load(rad_ptr + v, mask=m, other=0.0)
    comp_v = tl.load(comp_ptr + v, mask=m, other=0) != 0

    bestc = tl.load(cap_src + v, mask=m, other=-float('inf'))
    bestl = tl.load(lbl_src + v, mask=m, other=0)
    oldc = bestc
    oldl = bestl

    iz = v % nz
    iy = (v // nz) % ny
    ix = v // (nz * ny)

    off_yz = ny * nz
    for dir in range(6):
        if dir == 0:
            off = off_yz          # proposal comes from ix+1
            inb = (ix + 1 < nx)
        elif dir == 1:
            off = -off_yz         # proposal comes from ix-1
            inb = ix > 0
        elif dir == 2:
            off = nz              # proposal comes from iy+1
            inb = (iy + 1 < ny)
        elif dir == 3:
            off = -nz             # proposal comes from iy-1
            inb = iy > 0
        elif dir == 4:
            off = 1               # proposal comes from iz+1
            inb = (iz + 1 < nz)
        else:
            off = -1              # proposal comes from iz-1
            inb = iz > 0
        inb = inb & m

        cap_n = tl.load(cap_src + v + off, mask=inb, other=-float('inf'))
        lbl_n = tl.load(lbl_src + v + off, mask=inb, other=0)

        prop = tl.minimum(cap_n, rad_v)
        valid = comp_v & (lbl_n > 0)

        higher = prop > bestc
        equal = prop == bestc
        smaller = (bestl == 0) | (lbl_n < bestl)
        improve = valid & (higher | (equal & smaller))

        bestc = tl.where(improve, prop, bestc)
        bestl = tl.where(improve, lbl_n, bestl)

    tl.store(cap_dst + v, bestc, mask=m)
    tl.store(lbl_dst + v, bestl, mask=m)

    changed = m & ((bestc != oldc) | (bestl != oldl))
    tl.atomic_max(flag_ptr, tl.max(changed.to(tl.int32), axis=0))


def GPoreU_safe_watershed(_input, merged_maxima_ids, wall_thickness, grid_spacing, safe_buffer=6.0):

        """Partition the void voxels into pore basins.

        Safe-region assignment, iterative competitive flooding (Triton
        sweeps along the six axis directions), then merger of periodic
        pores.  Updates and returns the input dict (``pore_labels``,
        ``num_pores``, center bookkeeping, iteration counts).
        """
        grid = _input['grid'].contiguous()
        grid_radaii = _input['grid_radaii'].reshape(-1).contiguous()
        geo_grid_occ = _input['geo_grid_occ'].reshape(-1).bool().contiguous()
        merged_maxima_ids = merged_maxima_ids.reshape(-1).long().contiguous()

        device = grid_radaii.device
        NGRID = grid_radaii.numel()
        prof.mark('ws:apply-pbc')
        num_pores = merged_maxima_ids.numel()

        if num_pores == 0:
                # A truly non-porous structure legitimately has no H-maxima:
                # report an empty pore network instead of failing the whole
                # file, so the other requested methods still deliver.
                nxv, nyv, nzv = (int(_input['nx']), int(_input['ny']),
                                 int(_input['nz']))
                nvox = nxv * nyv * nzv
                zeros_l = torch.zeros(nvox, device=device, dtype=torch.long)
                _input['pore_labels'] = zeros_l
                _input['merged_maxima_ids'] = merged_maxima_ids
                _input['markers'] = merged_maxima_ids
                _input['num_pores'] = 0
                _input['clusters_radaii'] = grid_radaii[merged_maxima_ids]
                _input['clusters_cart_positions'] = grid[merged_maxima_ids]
                _input['periodic_label_map'] = zeros_l.clone()
                _input['periodic_pairs'] = torch.empty(0, 2, device=device,
                                                       dtype=torch.long)
                _input['pore_labels_before_pbc'] = zeros_l.clone()
                _input['centers_before_pbc'] = merged_maxima_ids.clone()
                _input['center_distances'] = torch.empty(0, 0, device=device)
                _input['safe_radaii'] = torch.empty(0, device=device)
                _input['safe_mask'] = torch.zeros(nvox, device=device,
                                                  dtype=torch.bool)
                _input['flood_iterations'] = 0
                return _input

        if safe_buffer < 0:
                raise ValueError('Safe buffer cannot be negative.')


# Calculate distances between centers
        clusters_cart_positions = grid[merged_maxima_ids].float()
        clusters_radaii = grid_radaii[merged_maxima_ids].float()

        center_distances = torch.cdist(clusters_cart_positions, clusters_cart_positions)
        separation_distances = center_distances.clone()
        separation_distances.fill_diagonal_(float('inf'))


# Calculate safe radii
        clearance = separation_distances - clusters_radaii[None, :]
        clearance = clearance.amin(dim=1)

        safe_radaii = torch.minimum(clusters_radaii, clearance)
        safe_radaii = safe_radaii - safe_buffer
        safe_radaii = safe_radaii.clamp_min(0).contiguous()


# Prepare outputs
        pore_labels = torch.zeros(NGRID, device=device, dtype=torch.long)
        safe_mask = torch.zeros(NGRID, device=device, dtype=torch.bool)


# Assign safe regions
        BLOCKSIZE = 128
        safe_grid = (triton.cdiv(NGRID, BLOCKSIZE),)

        GPoreU_SafeRegion_Kernel[safe_grid](grid, merged_maxima_ids, safe_radaii, geo_grid_occ, pore_labels, safe_mask, NGRID, num_pores, BLOCKSIZE=BLOCKSIZE)


# Retain center markers even when their safe radius is zero
        pore_labels[merged_maxima_ids] = torch.arange(1, num_pores + 1, device=device)
        safe_mask[merged_maxima_ids] = True


# Remaining void voxels go to flooding
        competitive_mask = ~geo_grid_occ & ~safe_mask

# Reshape grids
        nx = int(_input['nx'])
        ny = int(_input['ny'])
        nz = int(_input['nz'])

        pore_labels = pore_labels.reshape(nx, ny, nz)
        safe_mask = safe_mask.reshape(nx, ny, nz)
        competitive_mask = competitive_mask.reshape(nx, ny, nz)
        grid_radaii = grid_radaii.reshape(nx, ny, nz)


# Start path capacities at the safe voxels
        capacity = torch.full((nx, ny, nz), -float('inf'), device=device, dtype=torch.float32)
        capacity[safe_mask] = grid_radaii[safe_mask].float()

        # Triton competitive flooding: bit-same semantics as the reference
        # torch loop (six directed votes in order x+, x-, y+, y-, z+, z-;
        # higher capacity, then smaller label; non-periodic neighbors), but
        # one kernel per sweep, double-buffered, device-side change flag.
        cap_a = capacity.contiguous()
        cap_b = torch.empty_like(cap_a)
        lbl_a = pore_labels.to(torch.int32).contiguous()
        lbl_b = torch.empty_like(lbl_a)
        rad_f = grid_radaii.reshape(nx, ny, nz).float().contiguous()
        comp8 = competitive_mask.reshape(nx, ny, nz).to(torch.int8).contiguous()
        flag = torch.zeros(1, device=device, dtype=torch.int32)

        NGRIDF = nx * ny * nz
        FLOOD_BLOCK = 256
        sweep = (triton.cdiv(NGRIDF, FLOOD_BLOCK),)

        flood_iterations = 0

        # Repeat until capacities and labels stop changing
        while True:

                flag.zero_()
                GPoreU_Flood_Sweep_Kernel[sweep](
                        cap_a, lbl_a, cap_b, lbl_b, rad_f, comp8, flag,
                        nx, ny, nz, NGRIDF, BLOCKSIZE=FLOOD_BLOCK)

                cap_a, cap_b = cap_b, cap_a
                lbl_a, lbl_b = lbl_b, lbl_a

                flood_iterations += 1

                # Stop only when neither capacities nor labels change
                if int(flag[0]) == 0:
                        break

        prof.mark('ws:flood')
        pore_labels = lbl_a.to(torch.long)
        capacity = cap_a


# Report results
        void_mask = ~geo_grid_occ.reshape(nx, ny, nz)
        unassigned = void_mask & (pore_labels == 0)

        prof.vprint('Competitive flooding iterations:', flood_iterations, flush=True)
        prof.vprint('Unassigned void points:', int(unassigned.sum().item()), flush=True)

# Save results before periodic merging
        pore_labels_before_pbc = pore_labels.clone()
        centers_before_pbc = merged_maxima_ids.clone()


# Merge pores across periodic walls
        pore_labels, merged_maxima_ids, periodic_label_map, periodic_pairs = GPoreU_apply_pbc(pore_labels, merged_maxima_ids, grid_radaii, wall_thickness, grid_spacing, _input['cell'])

        prof.mark('ws:apply-pbc')
        num_pores = merged_maxima_ids.numel()


# Pack results
        _input['pore_labels'] = pore_labels.reshape(-1)
        _input['merged_maxima_ids'] = merged_maxima_ids
        _input['markers'] = merged_maxima_ids
        _input['num_pores'] = num_pores
        _input['clusters_radaii'] = grid_radaii.reshape(-1)[merged_maxima_ids]
        _input['clusters_cart_positions'] = grid[merged_maxima_ids]
        _input['periodic_label_map'] = periodic_label_map
        _input['periodic_pairs'] = periodic_pairs
        _input['pore_labels_before_pbc'] = pore_labels_before_pbc.reshape(-1)
        _input['centers_before_pbc'] = centers_before_pbc
        _input['center_distances'] = center_distances
        _input['safe_radaii'] = safe_radaii
        _input['safe_mask'] = safe_mask.reshape(-1)
        _input['competitive_mask'] = competitive_mask.reshape(-1)
        _input['flood_iterations'] = flood_iterations

        return _input

# ------------------------------------------------------------ pore graph ----


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["n_ij"])
def _local_maxima_kernel(
    grid_radaii_ptr,
    ij_geo_ptr,
    not_max_ptr,
    n_ij,
    BLOCKSIZE: tl.constexpr,
):
    """Mark every grid point that has a neighbour with a larger radius.

    A point is a local maximum iff it is marked by no edge, so the
    surviving (unmarked) points are exactly the local maxima of the
    radius field over the grid graph.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offs < n_ij

    i = tl.load(ij_geo_ptr + offs * 2, mask=mask)
    j = tl.load(ij_geo_ptr + offs * 2 + 1, mask=mask)

    ri = tl.load(grid_radaii_ptr + i, mask=mask)
    rj = tl.load(grid_radaii_ptr + j, mask=mask)

    tl.atomic_or(not_max_ptr + i, (ri < rj).to(tl.int32), mask=mask)
    tl.atomic_or(not_max_ptr + j, (rj < ri).to(tl.int32), mask=mask)


def maximum_spanning_tree(ij_geo, grid_radaii):
    """Maximum spanning forest with edge weight min(r_i, r_j).

    Computed by the Triton Boruvka engine (:func:`gporeu.utils.union_find.max_st`).
    Any maximum spanning tree yields identical widest-path bottlenecks, so
    the downstream H-max reconstruction is tree-independent.
    """
    n_nodes = grid_radaii.shape[0]
    i = ij_geo[:, 0].to(torch.int32)
    j = ij_geo[:, 1].to(torch.int32)
    edge_radii = torch.minimum(grid_radaii[i.long()],
                               grid_radaii[j.long()]).to(torch.float32)
    return max_st(n_nodes, i, j, edge_radii)


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["n_edges"])
def _hmax_kernel(
    src_ptr,
    dst_ptr,
    edge_radius_ptr,
    reconstructed_ptr,
    n_edges,
    BLOCKSIZE: tl.constexpr,
):
    """One H-max reconstruction sweep over the spanning-tree edges.

    Water level min(level_neighbour, edge_radius) flows across each edge;
    the running maximum at every vertex is its reconstructed level.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offs < n_edges

    i = tl.load(src_ptr + offs, mask=mask, other=0)
    j = tl.load(dst_ptr + offs, mask=mask, other=0)
    edge_radius = tl.load(edge_radius_ptr + offs, mask=mask, other=0.0)

    value_i = tl.load(reconstructed_ptr + i, mask=mask, other=0.0)
    value_j = tl.load(reconstructed_ptr + j, mask=mask, other=0.0)

    tl.atomic_max(reconstructed_ptr + j, tl.minimum(value_i, edge_radius),
                  mask=mask)
    tl.atomic_max(reconstructed_ptr + i, tl.minimum(value_j, edge_radius),
                  mask=mask)


def _merge_adjacent_centers(hmax_ids, maxima_pairs, grid_radaii):
    """Merge pore centers joined by a grid edge, keeping the widest one.

    The maxima-induced subgraph is compacted to its K distinct vertices so
    the union-find only pays K-sized arrays; per cluster the first vertex
    (in stable order) attaining the cluster maximum radius is retained.
    Unpaired maxima pass through unchanged.
    """
    if maxima_pairs.shape[0] == 0:
        return hmax_ids

    vertices = torch.unique(maxima_pairs.flatten())
    K = int(vertices.shape[0])
    dense_map = torch.zeros(grid_radaii.shape[0], dtype=torch.int32,
                            device=vertices.device)
    dense_map[vertices] = torch.arange(K, dtype=torch.int32,
                                       device=vertices.device)
    prof.mark('poregraph:mc-compact')

    pu = dense_map[maxima_pairs[:, 0]]
    pv = dense_map[maxima_pairs[:, 1]]
    roots_p = components(K, pu, pv)
    cluster_ids = roots_p[torch.arange(K, device=vertices.device)]
    radaii = grid_radaii[vertices]
    unique_clusters = torch.unique(cluster_ids, return_counts=True)

    # Vectorized per-cluster argmax: sort by cluster, mark the positions
    # that attain the cluster maximum (via segment scatter_reduce), and
    # take the first such position in each segment.
    order = torch.argsort(cluster_ids, stable=True)
    v_s = vertices[order]
    r_s = radaii[order]
    grp = torch.searchsorted(unique_clusters[0], cluster_ids[order])
    nk = int(unique_clusters[0].shape[0])
    max_r = torch.full((nk,), float('-inf'), device=vertices.device,
                       dtype=r_s.dtype)
    max_r.scatter_reduce_(0, grp, r_s, reduce='amax', include_self=True)

    BIG = r_s.numel() + 1
    pos = ((r_s == max_r[grp]).to(v_s.dtype)
           * torch.arange(1, v_s.numel() + 1, device=v_s.device,
                          dtype=v_s.dtype)) - 1
    pos = torch.where(r_s == max_r[grp], pos, torch.full_like(pos, BIG))
    first_pos = torch.full((nk,), BIG, device=v_s.device, dtype=v_s.dtype)
    first_pos.scatter_reduce_(0, grp, pos, reduce='amin', include_self=True)
    max_vertices = torch.gather(v_s, 0, first_pos.to(torch.long))
    prof.mark('poregraph:mc-segment')

    not_paired_maxima = hmax_ids[
        ~torch.isin(hmax_ids, maxima_pairs.flatten())]
    return torch.cat((max_vertices, not_paired_maxima))


def GPoreU_poreGraph(_input, safe_buffer=6.0):
    """Build the pore network and store all results in a copy of ``_input``.

    Requires the stage outputs of ``all_grids_pbc`` and
    ``GPoreU_percolate`` (``grid``, ``grid_radaii``, ``ij_geo``,
    ``geo_grid_occ``, grid dimensions).  Adds pore-center, H-max, spanning
    -tree and watershed (``pore_labels``, ``num_pores``) fields.
    """
    grid = _input['grid']
    grid_radaii = _input['grid_radaii']
    ij_geo = _input['ij_geo']

    wall_thickness = _input['wall_thickness']
    grid_spacing = _input['grid_spacing']
    NGRID = grid.shape[0]
    n_ij = ij_geo.shape[0]

    # ---- local maxima of the radius field ------------------------------
    not_max = torch.zeros(NGRID, device='cuda', dtype=torch.int32)
    BLOCKSIZE = 128
    _local_maxima_kernel[(triton.cdiv(n_ij, BLOCKSIZE),)](
        grid_radaii, ij_geo, not_max, n_ij, BLOCKSIZE)

    # Maxima above the solid-detection floor only.
    maxima_check = (not_max == 0) & (grid_radaii > 1.0)
    maxima_ids = torch.argwhere(maxima_check).flatten()
    maxima_radii = grid_radaii[maxima_ids]

    sort_idx = torch.argsort(maxima_radii, descending=True)
    maxima_ids = maxima_ids[sort_idx]
    maxima_radii = maxima_radii[sort_idx]
    prof.mark('poregraph:maxima-detect')

    # ---- H-max reconstruction over the maximum spanning forest ---------
    mst_src, mst_dst, mst_edge_radii = maximum_spanning_tree(
        ij_geo, grid_radaii)
    prof.mark('poregraph:maxST')
    n_mst_edges = mst_src.shape[0]

    h = 0.5
    reconstructed = grid_radaii.to(torch.float32).clone() - h

    hmax_grid = (triton.cdiv(n_mst_edges, BLOCKSIZE),)
    iterations = 0
    while True:
        reconstructed_old = reconstructed.clone()
        _hmax_kernel[hmax_grid](
            mst_src, mst_dst, mst_edge_radii, reconstructed, n_mst_edges,
            BLOCKSIZE=BLOCKSIZE)
        iterations += 1
        if not torch.any(reconstructed != reconstructed_old).item():
            break
    prof.mark('poregraph:hmax-loop')

    maxima_reconstructed = reconstructed[maxima_ids]
    maxima_height = maxima_radii - maxima_reconstructed

    # Retain maxima whose height survived the reconstruction.
    significant_check = maxima_height >= (h - 1e-6)
    hmax_ids = maxima_ids[significant_check]
    hmax_radii = maxima_radii[significant_check]
    hmax_height = maxima_height[significant_check]

    nx, ny, nz = _input['nx'], _input['ny'], _input['nz']
    hmax_x = hmax_ids // (ny * nz)
    hmax_y = (hmax_ids % (ny * nz)) // nz
    hmax_z = hmax_ids % nz
    hmax_positions = torch.stack([hmax_x, hmax_y, hmax_z], dim=1)

    prof.mark('  poregraph:preprint')
    prof.vprint("H-max seed positions:")
    prof.vprint(hmax_positions)
    prof.mark('  poregraph:print1')
    prof.vprint("H-max physical positions:")
    prof.vprint(grid[hmax_ids])
    prof.mark('  poregraph:print2')

    hmax_check = torch.zeros(NGRID, device='cuda', dtype=torch.bool)
    hmax_check[hmax_ids] = True
    prof.mark('  poregraph:hmask')

    # ---- merge adjacent centers -----------------------------------------
    i, j = ij_geo[:, 0], ij_geo[:, 1]
    paired = hmax_check[i] & hmax_check[j]
    prof.mark('  poregraph:paired')
    maxima_pairs = ij_geo[paired]
    prof.mark('poregraph:mc-pairing')
    merged_maxima_ids = _merge_adjacent_centers(
        hmax_ids, maxima_pairs, grid_radaii)

    prof.vprint("H-max iterations:", iterations)
    prof.vprint("Number of local maxima before H-max:", maxima_ids.shape[0])
    prof.vprint("Number of maxima after H-max:", hmax_ids.shape[0])
    prof.vprint("H-max IDs:", hmax_ids)
    prof.vprint("H-max radii:", hmax_radii)
    prof.vprint("H-max heights:", hmax_height)

    # ---- store results ----------------------------------------------------
    _input['maxima_ids'] = merged_maxima_ids
    _input['maxima_radii'] = grid_radaii[merged_maxima_ids]
    _input['maxima_check'] = maxima_check
    _input['markers'] = merged_maxima_ids
    _input['mst_src'] = mst_src
    _input['mst_dst'] = mst_dst
    _input['mst_edge_radii'] = mst_edge_radii
    _input['reconstructed'] = reconstructed
    _input['hmax_ids'] = hmax_ids
    _input['hmax_radii'] = hmax_radii
    _input['hmax_height'] = hmax_height
    _input['hmax_check'] = hmax_check
    _input['hmax_positions'] = hmax_positions

    # ---- assign void voxels to centers (safe regions + flooding) ---------
    prof.mark('poregraph:merge-centers')
    _input = GPoreU_safe_watershed(_input, merged_maxima_ids,
                                   wall_thickness, grid_spacing,
                                   safe_buffer=safe_buffer)
    prof.mark('poregraph:watershed')
    return _input
