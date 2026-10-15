"""
Pore-network construction.

Detects pore centers as H-maxima of the grid radius field: a maximum
spanning forest of the grid graph (edge weight min(r_i, r_j)) supports a
reconstruction pass that propagates water levels; local maxima that retain
prominence ``h`` after reconstruction are significant pore centers.
Adjacent centers are then merged (union-find over the maxima-induced
subgraph, keeping the widest vertex of each cluster) and the remaining
void voxels are assigned to centers by safe-region watershed flooding
(:mod:`gporeu.watershed`).
"""

import torch
import triton
import triton.language as tl

from . import prof
from .union_find import components, max_st
from .watershed import GPoreU_safe_watershed


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

    Computed by the Triton Boruvka engine (:func:`gporeu.union_find.max_st`).
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
