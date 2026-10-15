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

def GPoreU_apply_pbc(pore_labels, merged_maxima_ids, grid_radaii, wall_thickness, grid_spacing):
        """Merge pore labels/centers identified across periodic boundaries.

        Wall contact counts drive a sequential merge on the CPU using pure
        integer and float32 comparisons (exact for the value ranges
        involved: threshold tests against 0.5 do not flip between fp32 and
        fp64 for wall areas below 4e6 cells).  This avoids ~3 kernel
        launches and synchronizations per candidate pair.  Results are
        returned as device tensors.
        """

        import numpy as np
        prof.mark('  pbc:enter')

# General dimensions
        nx, ny, nz = pore_labels.shape
        device = pore_labels.device

        wall_cells = math.ceil(wall_thickness / grid_spacing)

        if wall_cells < 1 or wall_cells > min(nx, ny, nz):
                raise ValueError('Wall thickness must select between 1 and the smallest grid dimension.')

        prof.mark('  pbc:prepared')
# Pull the (small) center bookkeeping to CPU
        lab_np = pore_labels.detach().cpu().numpy()
        rad_np = grid_radaii.reshape(-1).detach().cpu().numpy()

# Get center labels, radii, and positions
        center_labels = lab_np.reshape(-1)[merged_maxima_ids.detach().cpu().numpy().astype(np.int64)]
        center_radii = rad_np[merged_maxima_ids.detach().cpu().numpy().astype(np.int64)]

        merged_ids_np = merged_maxima_ids.detach().cpu().numpy().astype(np.int64)
        center_x = merged_ids_np // (ny * nz)
        center_y = (merged_ids_np % (ny * nz)) // nz
        center_z = merged_ids_np % nz

        center_positions = np.stack((center_x, center_y, center_z), axis=1)


# Sort labels and keep their corresponding data together
        center_order = np.argsort(center_labels, kind='stable')

        center_labels = center_labels[center_order]
        center_radii = center_radii[center_order]
        center_positions = center_positions[center_order]
        merged_maxima_np = merged_ids_np[center_order]

        all_merge_pairs = []


        prof.mark('  pbc:cpu-copied')
# Check all three dimensions
        for dimension in range(3):

                if dimension == 0:
                        left_wall = lab_np[:wall_cells, :, :]
                        right_wall = lab_np[-wall_cells:, :, :]
                        dimension_size = nx

                elif dimension == 1:
                        left_wall = lab_np[:, :wall_cells, :]
                        right_wall = lab_np[:, -wall_cells:, :]
                        dimension_size = ny

                else:
                        left_wall = lab_np[:, :, :wall_cells]
                        right_wall = lab_np[:, :, -wall_cells:]
                        dimension_size = nz


# Count labels across the extracted layers
                left_labels, left_counts = np.unique(left_wall, return_counts=True)
                right_labels, right_counts = np.unique(right_wall, return_counts=True)


# Count pairwise label co-occurrences across the pair.  Key-encoded 1D
# uniques: BIG exceeds any pore label, so numeric key order equals
# lexicographic order of the (left, right) label pairs exactly.
                l_flat = left_wall.reshape(-1)
                r_flat = right_wall.reshape(-1)
                lmax = int(max(l_flat.max(initial=0), r_flat.max(initial=0))) + 1
                keys = l_flat.astype(np.int64) * lmax + r_flat.astype(np.int64)
                ukeys, pair_counts = np.unique(keys, return_counts=True)
                unique_pairs = np.stack((ukeys // lmax, ukeys % lmax), axis=1)

                valid = (unique_pairs[:, 0] > 0) & (unique_pairs[:, 1] > 0) & (unique_pairs[:, 0] != unique_pairs[:, 1])

                unique_pairs = unique_pairs[valid]
                pair_counts = pair_counts[valid]


# Get total wall counts for each label in each pair
                left_index = np.searchsorted(left_labels, unique_pairs[:, 0])
                right_index = np.searchsorted(right_labels, unique_pairs[:, 1])

                left_total = left_counts[left_index].astype(np.float64)
                right_total = right_counts[right_index].astype(np.float64)

                left_fraction = pair_counts.astype(np.float64) / left_total
                right_fraction = pair_counts.astype(np.float64) / right_total

                overlap_check = (left_fraction > 0.5) | (right_fraction > 0.5)
                passed_pairs = unique_pairs[overlap_check]


# Get corresponding center radii and positions
                left_index = np.searchsorted(center_labels, passed_pairs[:, 0])
                right_index = np.searchsorted(center_labels, passed_pairs[:, 1])

                left_radius = center_radii[left_index]
                right_radius = center_radii[right_index]

                left_close = center_positions[left_index, dimension] < wall_cells
                right_close = center_positions[right_index, dimension] >= dimension_size - wall_cells


# Merge the smaller pore only if its center is close
                merge_left = left_close & (left_radius < right_radius)
                merge_right = right_close & (right_radius < left_radius)

                left_into_right = passed_pairs[merge_left]
                right_into_left = np.stack((passed_pairs[merge_right, 1], passed_pairs[merge_right, 0]), axis=1)

                dimension_merge_pairs = np.concatenate((left_into_right, right_into_left), axis=0)
                all_merge_pairs.append(dimension_merge_pairs)


        prof.mark('  pbc:walls')
# Combine directed pairs from all dimensions
        if all_merge_pairs:
                periodic_pairs_np = np.concatenate(all_merge_pairs, axis=0)
        else:
                periodic_pairs_np = np.zeros((0, 2), dtype=np.int64)
        if periodic_pairs_np.shape[0]:
                b2 = int(periodic_pairs_np.max(initial=0)) + 1
                k2 = (periodic_pairs_np[:, 0].astype(np.int64) * b2
                      + periodic_pairs_np[:, 1].astype(np.int64))
                uk2 = np.unique(k2)
                periodic_pairs_np = np.stack((uk2 // b2, uk2 % b2), axis=1)


# Track the surviving original label for each pore
        merged_labels = center_labels.copy()

        for row in range(periodic_pairs_np.shape[0]):

                p0 = periodic_pairs_np[row, 0]
                p1 = periodic_pairs_np[row, 1]

                smaller_index = np.searchsorted(center_labels, p0)
                larger_index = np.searchsorted(center_labels, p1)

                smaller_label = merged_labels[smaller_index]
                larger_label = merged_labels[larger_index]


# Compare surviving centers when earlier merges have joined groups
                smaller_index = np.searchsorted(center_labels, smaller_label)
                larger_index = np.searchsorted(center_labels, larger_label)

                keep_smaller = center_radii[smaller_index] > center_radii[larger_index]
                equal_radius = center_radii[smaller_index] == center_radii[larger_index]
                keep_smaller = bool(keep_smaller) or (bool(equal_radius) and smaller_label < larger_label)

                keep_label = smaller_label if keep_smaller else larger_label
                remove_label = larger_label if keep_smaller else smaller_label

                merged_labels[merged_labels == remove_label] = keep_label


        prof.mark('  pbc:merge-loop')
# Keep one center per merged group
        surviving_labels = np.unique(merged_labels)
        keep_centers = np.isin(center_labels, surviving_labels)

        merged_maxima_np = merged_maxima_np[keep_centers]


# Number the final pores from 1 onward
        final_labels = np.searchsorted(surviving_labels, merged_labels)
        final_labels = final_labels + 1


# Map original labels to final labels, with background remaining zero
        periodic_label_map_np = np.zeros(int(center_labels.max()) + 1, dtype=np.int64)
        periodic_label_map_np[center_labels] = final_labels


# Update voxel labels
        pore_labels_np = periodic_label_map_np[lab_np]

        prof.mark('  pbc:renumbered')
        pore_labels = torch.from_numpy(np.ascontiguousarray(pore_labels_np)).to(device)
        merged_maxima_ids = torch.from_numpy(np.ascontiguousarray(merged_maxima_np)).to(device)
        periodic_label_map = torch.from_numpy(np.ascontiguousarray(periodic_label_map_np)).to(device)
        periodic_pairs = torch.from_numpy(np.ascontiguousarray(periodic_pairs_np)).to(device)

        prof.mark('  pbc:returned')
        return pore_labels, merged_maxima_ids, periodic_label_map, periodic_pairs

@triton.jit
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

@triton.jit
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
                raise ValueError('No pore centers were supplied.')

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
        pore_labels, merged_maxima_ids, periodic_label_map, periodic_pairs = GPoreU_apply_pbc(pore_labels, merged_maxima_ids, grid_radaii, wall_thickness, grid_spacing)

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


@triton.jit
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


@triton.jit
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
