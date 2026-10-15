import math
import torch
import triton
import triton.language as tl
from . import prof

def GPoreU_apply_pbc(pore_labels, merged_maxima_ids, grid_radaii, wall_thickness, grid_spacing):
        '''CPU/NumPy port of the periodic-center merge.  Pure integer +
    float32 index comparisons (every comparison is exact for the value
    ranges involved: fractions >0.5 cannot change outcome between fp32
    and fp64 for wall areas < 4e6 cells), so it is bit-identical to the
        original GPU version while avoiding ~3 launches/sync per pair in
        the sequential merge loop.  Returns device tensors as before. '''

        import numpy as np
        prof.mark('  pbc:enter')

#general dimensions
        nx, ny, nz = pore_labels.shape
        device = pore_labels.device

        wall_cells = math.ceil(wall_thickness / grid_spacing)

        if wall_cells < 1 or wall_cells > min(nx, ny, nz):
                raise ValueError('Wall thickness must select between 1 and the smallest grid dimension.')

        prof.mark('  pbc:prepared')
#pull the (small) center bookkeeping to CPU
        lab_np = pore_labels.detach().cpu().numpy()
        rad_np = grid_radaii.reshape(-1).detach().cpu().numpy()

#get center labels, radii, and positions
        center_labels = lab_np.reshape(-1)[merged_maxima_ids.detach().cpu().numpy().astype(np.int64)]
        center_radii = rad_np[merged_maxima_ids.detach().cpu().numpy().astype(np.int64)]

        merged_ids_np = merged_maxima_ids.detach().cpu().numpy().astype(np.int64)
        center_x = merged_ids_np // (ny * nz)
        center_y = (merged_ids_np % (ny * nz)) // nz
        center_z = merged_ids_np % nz

        center_positions = np.stack((center_x, center_y, center_z), axis=1)


#sort labels and keep their corresponding data together
        center_order = np.argsort(center_labels, kind='stable')

        center_labels = center_labels[center_order]
        center_radii = center_radii[center_order]
        center_positions = center_positions[center_order]
        merged_maxima_np = merged_ids_np[center_order]

        all_merge_pairs = []


        prof.mark('  pbc:cpu-copied')
#check all three dimensions
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


#count labels across the extracted layers
                left_labels, left_counts = np.unique(left_wall, return_counts=True)
                right_labels, right_counts = np.unique(right_wall, return_counts=True)


#count corresponding label pairs (key-encoded 1D unique: np.unique on
#row-views is the old hotspot; BIG exceeds any pore label so ordering by
#key equals lexicographic ordering of (left, right) pairs exactly)
                l_flat = left_wall.reshape(-1)
                r_flat = right_wall.reshape(-1)
                lmax = int(max(l_flat.max(initial=0), r_flat.max(initial=0))) + 1
                keys = l_flat.astype(np.int64) * lmax + r_flat.astype(np.int64)
                ukeys, pair_counts = np.unique(keys, return_counts=True)
                unique_pairs = np.stack((ukeys // lmax, ukeys % lmax), axis=1)

                valid = (unique_pairs[:, 0] > 0) & (unique_pairs[:, 1] > 0) & (unique_pairs[:, 0] != unique_pairs[:, 1])

                unique_pairs = unique_pairs[valid]
                pair_counts = pair_counts[valid]


#get total wall counts for each label in each pair
                left_index = np.searchsorted(left_labels, unique_pairs[:, 0])
                right_index = np.searchsorted(right_labels, unique_pairs[:, 1])

                left_total = left_counts[left_index].astype(np.float64)
                right_total = right_counts[right_index].astype(np.float64)

                left_fraction = pair_counts.astype(np.float64) / left_total
                right_fraction = pair_counts.astype(np.float64) / right_total

                overlap_check = (left_fraction > 0.5) | (right_fraction > 0.5)
                passed_pairs = unique_pairs[overlap_check]


#get corresponding center radii and positions
                left_index = np.searchsorted(center_labels, passed_pairs[:, 0])
                right_index = np.searchsorted(center_labels, passed_pairs[:, 1])

                left_radius = center_radii[left_index]
                right_radius = center_radii[right_index]

                left_close = center_positions[left_index, dimension] < wall_cells
                right_close = center_positions[right_index, dimension] >= dimension_size - wall_cells


#merge the smaller pore only if its center is close
                merge_left = left_close & (left_radius < right_radius)
                merge_right = right_close & (right_radius < left_radius)

                left_into_right = passed_pairs[merge_left]
                right_into_left = np.stack((passed_pairs[merge_right, 1], passed_pairs[merge_right, 0]), axis=1)

                dimension_merge_pairs = np.concatenate((left_into_right, right_into_left), axis=0)
                all_merge_pairs.append(dimension_merge_pairs)


        prof.mark('  pbc:walls')
#combine directed pairs from all dimensions
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


#track the surviving original label for each pore
        merged_labels = center_labels.copy()

        for row in range(periodic_pairs_np.shape[0]):

                p0 = periodic_pairs_np[row, 0]
                p1 = periodic_pairs_np[row, 1]

                smaller_index = np.searchsorted(center_labels, p0)
                larger_index = np.searchsorted(center_labels, p1)

                smaller_label = merged_labels[smaller_index]
                larger_label = merged_labels[larger_index]


#compare surviving centers when earlier merges have joined groups
                smaller_index = np.searchsorted(center_labels, smaller_label)
                larger_index = np.searchsorted(center_labels, larger_label)

                keep_smaller = center_radii[smaller_index] > center_radii[larger_index]
                equal_radius = center_radii[smaller_index] == center_radii[larger_index]
                keep_smaller = bool(keep_smaller) or (bool(equal_radius) and smaller_label < larger_label)

                keep_label = smaller_label if keep_smaller else larger_label
                remove_label = larger_label if keep_smaller else smaller_label

                merged_labels[merged_labels == remove_label] = keep_label


        prof.mark('  pbc:merge-loop')
#keep one center per merged group
        surviving_labels = np.unique(merged_labels)
        keep_centers = np.isin(center_labels, surviving_labels)

        merged_maxima_np = merged_maxima_np[keep_centers]


#number the final pores from 1 onward
        final_labels = np.searchsorted(surviving_labels, merged_labels)
        final_labels = final_labels + 1


#map original labels to final labels, with background remaining zero
        periodic_label_map_np = np.zeros(int(center_labels.max()) + 1, dtype=np.int64)
        periodic_label_map_np[center_labels] = final_labels


#update voxel labels
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

#identify voxels in this block
        pid = tl.program_id(0)
        offsets = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
        valid = offsets < NGRID


#load voxel coordinates
        x = tl.load(grid + offsets * 3, mask=valid, other=0).to(tl.float32)
        y = tl.load(grid + offsets * 3 + 1, mask=valid, other=0).to(tl.float32)
        z = tl.load(grid + offsets * 3 + 2, mask=valid, other=0).to(tl.float32)


#check whether voxels are void
        occupied = tl.load(geo_grid_occ + offsets, mask=valid, other=1)
        void = valid & (occupied == 0)


#start with no claims or labels
        claim_counts = tl.full((BLOCKSIZE,), 0, tl.int32)
        labels = tl.full((BLOCKSIZE,), 0, tl.int32)


#check each pore's safe sphere
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


#assign only voxels claimed by exactly one pore
        safe = void & (claim_counts == 1)
        labels = tl.where(safe, labels, 0)

        tl.store(pore_labels + offsets, labels, mask=valid)
        tl.store(safe_mask + offsets, safe, mask=valid)

@triton.jit
def GPoreU_Flood_Sweep_Kernel(
        cap_src, lbl_src, cap_dst, lbl_dst, rad_ptr, comp_ptr, flag_ptr,
        nx, ny, nz, NGRID, BLOCKSIZE: tl.constexpr):
    '''One synchronous competitive-flooding sweep over the whole grid.

    Exact replica of the reference torch loop: six directed neighborhoods
    in order (x+, x-, y+, y-, z+, z-); proposals only accepted by
    competitive voxels with a labeled neighbor; higher capacity wins,
    ties prefer a smaller (zero-favoring) label; out-of-bounds neighbor
    capacity is -inf.  Reads only from the *_src arrays (double buffered)
    so every vote sees the previous sweep.  Sets *flag_ptr if anything
    changed.
    '''
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

#get inputs
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


#calculate distances between centers
        clusters_cart_positions = grid[merged_maxima_ids].float()
        clusters_radaii = grid_radaii[merged_maxima_ids].float()

        center_distances = torch.cdist(clusters_cart_positions, clusters_cart_positions)
        separation_distances = center_distances.clone()
        separation_distances.fill_diagonal_(float('inf'))


#calculate safe radii
        clearance = separation_distances - clusters_radaii[None, :]
        clearance = clearance.amin(dim=1)

        safe_radaii = torch.minimum(clusters_radaii, clearance)
        safe_radaii = safe_radaii - safe_buffer
        safe_radaii = safe_radaii.clamp_min(0).contiguous()


#prepare outputs
        pore_labels = torch.zeros(NGRID, device=device, dtype=torch.long)
        safe_mask = torch.zeros(NGRID, device=device, dtype=torch.bool)


#assign safe regions
        BLOCKSIZE = 128
        safe_grid = (triton.cdiv(NGRID, BLOCKSIZE),)

        GPoreU_SafeRegion_Kernel[safe_grid](grid, merged_maxima_ids, safe_radaii, geo_grid_occ, pore_labels, safe_mask, NGRID, num_pores, BLOCKSIZE=BLOCKSIZE)


#retain center markers even when their safe radius is zero
        pore_labels[merged_maxima_ids] = torch.arange(1, num_pores + 1, device=device)
        safe_mask[merged_maxima_ids] = True


#remaining void voxels go to flooding
        competitive_mask = ~geo_grid_occ & ~safe_mask

#reshape grids
        nx = int(_input['nx'])
        ny = int(_input['ny'])
        nz = int(_input['nz'])

        pore_labels = pore_labels.reshape(nx, ny, nz)
        safe_mask = safe_mask.reshape(nx, ny, nz)
        competitive_mask = competitive_mask.reshape(nx, ny, nz)
        grid_radaii = grid_radaii.reshape(nx, ny, nz)


#start path capacities at the safe voxels
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

        #repeat until capacities and labels stop changing
        while True:

                flag.zero_()
                GPoreU_Flood_Sweep_Kernel[sweep](
                        cap_a, lbl_a, cap_b, lbl_b, rad_f, comp8, flag,
                        nx, ny, nz, NGRIDF, BLOCKSIZE=FLOOD_BLOCK)

                cap_a, cap_b = cap_b, cap_a
                lbl_a, lbl_b = lbl_b, lbl_a

                flood_iterations += 1

                #stop only when neither capacities nor labels change
                if int(flag[0]) == 0:
                        break

        prof.mark('ws:flood')
        pore_labels = lbl_a.to(torch.long)
        capacity = cap_a


#report results
        void_mask = ~geo_grid_occ.reshape(nx, ny, nz)
        unassigned = void_mask & (pore_labels == 0)

        prof.vprint('Competitive flooding iterations:', flood_iterations, flush=True)
        prof.vprint('Unassigned void points:', int(unassigned.sum().item()), flush=True)

#save results before periodic merging
        pore_labels_before_pbc = pore_labels.clone()
        centers_before_pbc = merged_maxima_ids.clone()


#merge pores across periodic walls
        pore_labels, merged_maxima_ids, periodic_label_map, periodic_pairs = GPoreU_apply_pbc(pore_labels, merged_maxima_ids, grid_radaii, wall_thickness, grid_spacing)

        prof.mark('ws:apply-pbc')
        num_pores = merged_maxima_ids.numel()


#pack results
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
