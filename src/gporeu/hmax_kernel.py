import torch
import triton
import triton.language as tl
from .uf_graph import components, max_st
from . import prof
from .safe_kernel import GPoreU_safe_watershed


#my code and ideas
@triton.jit
def GPoreU_PoreGraph_Kernel(
                       NGRID: tl.constexpr,
                       cell_ptr,
                       grid_ptr,
                       grid_radaii_ptr,
                       ij_geo_ptr,
                       n_ij: tl.constexpr,
                       #ij_geo_periodic_ptr,
                       #n_ij_geo_periodic: tl.constexpr,
                       geo_grid_occ_ptr,
                       not_max_ptr,
                       BLOCKSIZE: tl.constexpr
                     ):


        pid = tl.program_id(0)


#make a decision of what the program handles
        ij_offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
        ij_mask = ij_offs < n_ij # n_ij= number of pairs in ij tensor(rows in ij)

        i = tl.load(ij_geo_ptr + ij_offs * 2, mask=ij_mask)
        j = tl.load(ij_geo_ptr + ij_offs * 2 + 1, mask=ij_mask)

        ri = tl.load(grid_radaii_ptr + i, mask=ij_mask)
        rj = tl.load(grid_radaii_ptr + j, mask=ij_mask)


#check which point has a larger radius
        i_smaller = (ri < rj)
        j_smaller = (rj < ri)

        tl.atomic_or(not_max_ptr + i, i_smaller.to(tl.int32), mask=ij_mask)
        tl.atomic_or(not_max_ptr + j, j_smaller.to(tl.int32), mask=ij_mask)


#exact same code for periodic wrapping
#periodic ij pairs handled by this program
#        ij_periodic_offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
 #       ij_periodic_mask = ij_periodic_offs < n_ij_geo_periodic


#load periodic i and j pairs
  #      ip = tl.load(ij_geo_periodic_ptr + ij_periodic_offs * 2, mask=ij_periodic_mask)
   #     jp = tl.load(ij_geo_periodic_ptr + ij_periodic_offs * 2 + 1, mask=ij_periodic_mask)


#load grid radius for periodic i and j
    #    rip = tl.load(grid_radaii_ptr + ip, mask=ij_periodic_mask)
     #   rjp = tl.load(grid_radaii_ptr + jp, mask=ij_periodic_mask)


#check which point has a larger radius
      #  ip_smaller = (rip < rjp)
       # jp_smaller = (rjp < rip)


#mark points which have a larger neighbor
        #tl.atomic_or(not_max_ptr + ip, ip_smaller.to(tl.int32), mask=ij_periodic_mask)
        #tl.atomic_or(not_max_ptr + jp, jp_smaller.to(tl.int32), mask=ij_periodic_mask)



def min_spanning_tree(ij_geo, grid_radaii):

    # Maximum spanning forest of the grid graph with edge weight
    # min(r_i, r_j) via Triton Boruvka (uf_graph.max_st).  Any maximum
    # spanning tree yields identical widest-path bottlenecks, so the
    # downstream H-max reconstruction is tree-independent.
    n_nodes = grid_radaii.shape[0]
    i = ij_geo[:, 0].to(torch.int32)
    j = ij_geo[:, 1].to(torch.int32)
    edge_radii = torch.minimum(grid_radaii[i.long()],
                               grid_radaii[j.long()]).to(torch.float32)
    src_e, dst_e, w_e = max_st(n_nodes, i, j, edge_radii)
    return src_e, dst_e, w_e


@triton.jit
def GPoreU_HMax_Kernel(
                       src_ptr,
                       dst_ptr,
                       edge_radius_ptr,
                       reconstructed_ptr,
                       n_edges,
                       BLOCKSIZE: tl.constexpr
                     ):


        pid = tl.program_id(0)


#load MST edges handled by this program
        offs = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
        mask = offs < n_edges

        i = tl.load(src_ptr + offs, mask=mask, other=0)
        j = tl.load(dst_ptr + offs, mask=mask, other=0)


#radius of the connection between i and j
        R_edge = tl.load(edge_radius_ptr + offs, mask=mask, other=0.0)


#current reconstructed value at both ends
        value_i = tl.load(reconstructed_ptr + i, mask=mask, other=0.0)
        value_j = tl.load(reconstructed_ptr + j, mask=mask, other=0.0)


#value that can propagate across the connection
        candidate_j = tl.minimum(value_i, R_edge)
        candidate_i = tl.minimum(value_j, R_edge)


#keep the largest reconstructed value reaching each point
        tl.atomic_max(reconstructed_ptr + j, candidate_j, mask=mask)
        tl.atomic_max(reconstructed_ptr + i, candidate_i, mask=mask)



def GPoreU_poreGraph(_input, safe_buffer=6.0):


#prepare inputs
        grid = _input['grid']
        cell = _input['cell']
        grid_radaii = _input['grid_radaii']
        ij_geo = _input['ij_geo']
        #ij_geo_periodic = _input['ij_geo_periodic']
        geo_grid_occ = _input['geo_grid_occ']

        wall_thickness = _input['wall_thickness']
        grid_spacing = _input['grid_spacing']
        NGRID = grid.shape[0]
        n_ij = ij_geo.shape[0]
       # n_ij_geo_periodic = ij_geo_periodic.shape[0]


#find local maxima
        not_max = torch.zeros(NGRID, device='cuda', dtype=torch.int32)

        BLOCKSIZE = 128

        n_pairs = n_ij
        triton_grid = (triton.cdiv(n_pairs, BLOCKSIZE),)

        GPoreU_PoreGraph_Kernel[triton_grid](
                NGRID,
                cell,
                grid,
                grid_radaii,
                ij_geo,
                n_ij,
               # ij_geo_periodic,
               # n_ij_geo_periodic,
                geo_grid_occ,
                not_max,
                BLOCKSIZE
        )


#keep maxima above absolute threshold
        maxima_check = (not_max == 0) & (grid_radaii > 1.0)

        maxima_ids = torch.argwhere(maxima_check).flatten()
        maxima_radii = grid_radaii[maxima_ids]


#sort maxima from largest to smallest
        sort_idx = torch.argsort(maxima_radii, descending=True)

        maxima_ids = maxima_ids[sort_idx]
        maxima_radii = maxima_radii[sort_idx]


#combine normal and periodic connections
       # ij_all = torch.cat([ij_geo, ij_geo_periodic], dim=0)


#make minimum spanning tree using negative edge radius
        prof.mark('poregraph:maxima-detect')
        MST = min_spanning_tree(ij_geo, grid_radaii)
        prof.mark('poregraph:maxST')


#unpack the Triton max-ST edge lists (already torch tensors)
        mst_src, mst_dst, mst_edge_radii = MST

        n_mst_edges = mst_src.shape[0]


#print current results before H-max
        prof.vprint("Number of local maxima before H-max:", maxima_ids.shape[0])
        prof.vprint("Number of MST edges:", n_mst_edges)


#H-max height
        h = 0.5


#start reconstruction h below original radius
        reconstructed = grid_radaii.to(torch.float32).clone() - h


#run H-max reconstruction on MST until nothing changes
        hmax_grid = (triton.cdiv(n_mst_edges, BLOCKSIZE),)

        iterations = 0

        while True:

                reconstructed_old = reconstructed.clone()

                GPoreU_HMax_Kernel[hmax_grid](
                        mst_src,
                        mst_dst,
                        mst_edge_radii,
                        reconstructed,
                        n_mst_edges,
                        BLOCKSIZE=BLOCKSIZE
                )

                iterations += 1

                changed = torch.any(reconstructed != reconstructed_old).item()

                if not changed:
                        break


        prof.mark('poregraph:hmax-loop')
#height remaining at each original local maximum
        prof.mark('  mc:gather0'); maxima_reconstructed = reconstructed[maxima_ids]
        prof.mark('  mc:gather1')
        maxima_height = maxima_radii - maxima_reconstructed


#keep maxima with prominence at least h
        significant_check = maxima_height >= (h - 1e-6)

        hmax_ids = maxima_ids[significant_check]
        hmax_radii = maxima_radii[significant_check]
        hmax_height = maxima_height[significant_check]

        nx = _input['nx']
        ny = _input['ny']
        nz = _input['nz']
        hmax_x = hmax_ids // (ny * nz)
        hmax_y = (hmax_ids % (ny * nz)) // nz
        hmax_z = hmax_ids % nz

        hmax_positions = torch.stack([hmax_x, hmax_y, hmax_z], dim=1)

        prof.mark('  mc:preprint')
        prof.vprint("H-max seed positions:")
        prof.vprint(hmax_positions)
        prof.mark('  mc:print1')

        hmax_cart_positions = grid[hmax_ids]

        prof.vprint("H-max physical positions:")
        prof.vprint(hmax_cart_positions)
        prof.mark('  mc:print2')
#make full grid mask for significant maxima
        hmax_check = torch.zeros(NGRID, device='cuda', dtype=torch.bool)
        hmax_check[hmax_ids] = True
        prof.mark('  mc:hmask')

#merging neighboring pore centers
        i = ij_geo[:,0]
        j = ij_geo[:,1]

        paired = hmax_check[i] & hmax_check[j]
        prof.mark('  mc:paired')

        maxima_pairs = ij_geo[paired]
        prof.mark('poregraph:mc-pairing')

        if maxima_pairs.shape[0] == 0:
                merged_maxima_ids = hmax_ids
        else:
                ''' ---- CLUSTERS VIA Triton UNION-FIND (paired maxima) ----
                Compact the maxima-induced subgraph to K nodes so the tiny
                union-find pays only K-sized arrays, and use vectorized
                segment reductions instead of a Python loop over clusters
                (same winner rule: per cluster, first max radius wins). '''
                vertices = torch.unique(maxima_pairs.flatten())
                K = int(vertices.shape[0])
                dense_map = torch.zeros(grid_radaii.shape[0],
                                        dtype=torch.int32,
                                        device=vertices.device)
                dense_map[vertices] = torch.arange(K, dtype=torch.int32,
                                                   device=vertices.device)
                prof.mark('poregraph:mc-compact')
                pu = dense_map[maxima_pairs[:, 0]]
                pv = dense_map[maxima_pairs[:, 1]]
                roots_p = components(K, pu, pv)
                cluster_ids = roots_p[torch.arange(K, device=vertices.device)]

                radaii = grid_radaii[vertices]
                unique_clusters = torch.unique(cluster_ids,
                                               return_counts=True)

        #group vertices and radii by cluster (vectorized per-cluster
        #argmax: first occurrence of the cluster maximum radius wins)
                order = torch.argsort(cluster_ids, stable=True)
                v_s = vertices[order]
                r_s = radaii[order]
                grp = torch.searchsorted(unique_clusters[0],
                                         cluster_ids[order])
                nk = int(unique_clusters[0].shape[0])
                max_r = torch.full((nk,), float('-inf'),
                                   device=vertices.device,
                                   dtype=r_s.dtype)
                max_r.scatter_reduce_(0, grp, r_s, reduce='amax',
                                      include_self=True)
                BIG = r_s.numel() + 1
                pos = ((r_s == max_r[grp]).to(v_s.dtype)
                       * torch.arange(1, v_s.numel() + 1,
                                      device=v_s.device,
                                      dtype=v_s.dtype)) - 1
                pos = torch.where(r_s == max_r[grp], pos,
                                  torch.full_like(pos, BIG))
                first_pos = torch.full((nk,), BIG, device=v_s.device,
                                       dtype=v_s.dtype)
                first_pos.scatter_reduce_(0, grp, pos, reduce='amin',
                                          include_self=True)
                max_vertices = torch.gather(v_s, 0,
                                            first_pos.to(torch.long))
                prof.mark('poregraph:mc-segment')
                not_paired_maxima = hmax_ids[~torch.isin(hmax_ids, maxima_pairs.flatten())]
                #not_paired_maxima_radaii=grid_radaii[not_paired_maxima]
                merged_maxima_ids = torch.cat((max_vertices, not_paired_maxima))
                #merged_maxima_radaii=grid_radaii[merged_maxima_ids]
#name used below for the retained pore centers
        merged_maxima_ids = merged_maxima_ids

#print H-max results
        prof.vprint("H-max iterations:", iterations)
        prof.vprint("Number of local maxima before H-max:", maxima_ids.shape[0])
        prof.vprint("Number of maxima after H-max:", hmax_ids.shape[0])
        prof.vprint("H-max IDs:", hmax_ids)
        prof.vprint("H-max radii:", hmax_radii)
        prof.vprint("H-max heights:", hmax_height)


#add pore graph results to input
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

#assign void cells to the retained pore centers
        prof.mark('poregraph:merge-centers')
        _input = GPoreU_safe_watershed(_input, merged_maxima_ids, wall_thickness, grid_spacing, safe_buffer=safe_buffer)
        prof.mark('poregraph:watershed')
        return _input



#def GPoreU_safe_watershed(_input, merged_maxima_ids, safe_buffer=6.0, chunk_size=8192):


#prepare inputs
 #       grid = _input['grid']
  #      grid_radaii = _input['grid_radaii'].reshape(-1)
   #     geo_grid_occ = _input['geo_grid_occ'].reshape(-1).bool()
    #    ij_geo = _input['ij_geo'].long()
     #   nx = int(_input['nx'])
      #  ny = int(_input['ny'])
       # nz = int(_input['nz'])
        #device = grid_radaii.device
        #NGRID = grid_radaii.numel()
        #merged_maxima_ids = merged_maxima_ids.reshape(-1).long()
        #num_pores = merged_maxima_ids.numel()
        #if num_pores == 0:
         #       raise ValueError('No pore centers were supplied.')
       # if safe_buffer < 0:
        #        raise ValueError('Safe buffer cannot be negative.')
     #   if geo_grid_occ[merged_maxima_ids].any().item():
      #          raise ValueError('A pore center lies inside the solid.')
      #  if grid.shape != (NGRID, 3):
      #          raise ValueError('grid must contain Cartesian x,y,z for each vertex.')


#get actual center coordinates, vertex IDs are not distances
       # clusters_cart_positions = grid[merged_maxima_ids].float()
       # clusters_radaii = grid_radaii[merged_maxima_ids].float()
        #center_distances = torch.cdist(clusters_cart_positions, clusters_cart_positions)
       # separation_distances = center_distances.clone()
        #separation_distances.fill_diagonal_(float('inf'))


#safe core stays inside its own sphere and outside every other center sphere
       # clearance = (separation_distances - clusters_radaii[None, :]).amin(dim=1)
       # safe_radaii = (torch.minimum(clusters_radaii, clearance) - safe_buffer).clamp_min(0)
       # void_mask = ~geo_grid_occ
       # pore_labels = torch.zeros(NGRID, device=device, dtype=torch.long)
       # safe_mask = torch.zeros(NGRID, device=device, dtype=torch.bool)
       # print('Starting safe region assignment', flush=True)


#check grid points in chunks to limit the distance matrix memory
       # for start in range(0, NGRID, chunk_size):
        #        stop = min(start + chunk_size, NGRID)
         #       distances = torch.cdist(grid[start:stop].float(), clusters_cart_positions)
          #      claims = distances < safe_radaii[None, :]
           #     safe = (claims.sum(dim=1) == 1) & void_mask[start:stop]
           #     labels = claims.long().argmax(dim=1) + 1
            #    pore_labels[start:stop] = torch.where(safe, labels, 0)
             #   safe_mask[start:stop] = safe


#always retain the original center markers even when their safe radius is zero
      #  pore_labels[merged_maxima_ids] = torch.arange(1, num_pores + 1, device=device)
      #  safe_mask[merged_maxima_ids] = True
      #  safe_labels = pore_labels.clone()
      #  competitive_mask = void_mask & ~safe_mask
      #  print('Safe grid points:', int(safe_mask.sum().item()), flush=True)
       # print('Remaining void points:', int(competitive_mask.sum().item()), flush=True)


#use ordinary voxel neighbors here, periodic pore merging follows assignment
     #   i, j = ij_geo[:, 0], ij_geo[:, 1]
     #   voxel_positions = torch.stack((
      #          torch.arange(NGRID, device=device) // (ny * nz),
       #         (torch.arange(NGRID, device=device) % (ny * nz)) // nz,
        #        torch.arange(NGRID, device=device) % nz
       # ), dim=1)
       # local_edges = (voxel_positions[i] - voxel_positions[j]).abs().amax(dim=1) <= 1
       # valid = local_edges & void_mask[i] & void_mask[j] & (i != j)
       # i, j = i[valid], j[valid]
       # src = torch.cat((i, j))
       # dst = torch.cat((j, i))
       # edge_radaii = torch.minimum(grid_radaii[src], grid_radaii[dst]).float()


#sort directed edges by source so only changed vertices need propagation
       # order = torch.argsort(src)
       # src, dst, edge_radaii = src[order], dst[order], edge_radaii[order]
       # degree = torch.bincount(src, minlength=NGRID)
       # offsets = torch.cat((torch.zeros(1, device=device, dtype=torch.long), degree.cumsum(0)))
       # capacity = torch.full((NGRID,), -float('inf'), device=device)
       # capacity[safe_mask] = grid_radaii[safe_mask].float()
       # frontier = torch.where(safe_mask)[0]
       # flood_iterations = 0
 #       print('Starting competitive flooding', flush=True)


#prefer the path with the largest minimum radius, break equal proposals by label
#safe cells remain fixed; only remaining void cells receive competitive updates
#        while frontier.numel() > 0:
 #               edge_counts = degree[frontier]
  #              total = int(edge_counts.sum().item())
   #             if total == 0:
    #                    break
     #           starts = torch.repeat_interleave(offsets[frontier], edge_counts)
      #          group_starts = edge_counts.cumsum(0) - edge_counts
        #        relative = torch.arange(total, device=device) - torch.repeat_interleave(group_starts, edge_counts)
       #         edges = starts + relative
         #       sources = src[edges]
          #      targets = dst[edges]
           #     movable = ~safe_mask[targets]
            #    sources, targets, edges = sources[movable], targets[movable], edges[movable]
             #   if targets.numel() == 0:
              #          break
              #  proposals = torch.minimum(capacity[sources], edge_radaii[edges])
              #  proposal_labels = pore_labels[sources]
               # touched, inverse = torch.unique(targets, return_inverse=True)
              #  best = torch.full((touched.numel(),), -float('inf'), device=device)
              #  best.scatter_reduce_(0, inverse, proposals, reduce='amax', include_self=True)
              #  winners = proposals == best[inverse]
              #  best_labels = torch.full((touched.numel(),), num_pores + 1, device=device, dtype=torch.long)
               # best_labels.scatter_reduce_(0, inverse[winners], proposal_labels[winners], reduce='amin', include_self=True)
               # improve = (best > capacity[touched]) | (
               #         (best == capacity[touched]) & (best_labels < pore_labels[touched])
             #   )
              #  frontier = touched[improve]
               # capacity[frontier] = best[improve]
               # pore_labels[frontier] = best_labels[improve]
               # flood_iterations += 1

    #    print('Competitive flooding iterations:', flood_iterations, flush=True)
     #   print('Unassigned void points:', int((void_mask & (pore_labels == 0)).sum().item()), flush=True)


#merge pore labels across periodic walls using the existing PoroNet checks
      #  pore_labels_before_pbc = pore_labels.clone()
       # centers_before_pbc = merged_maxima_ids.clone()
      #  print('Starting periodic merging', flush=True)
      #  pore_labels, merged_maxima_ids, periodic_label_map, periodic_pairs = GPoreU_apply_pbc(
       #         pore_labels.reshape(nx, ny, nz), merged_maxima_ids, grid_radaii, _input['cell']
      #  )
      #  num_pores = merged_maxima_ids.numel()
      #  markers = torch.zeros((nx, ny, nz), device=device, dtype=torch.int32)
      #  markers.reshape(-1)[geo_grid_occ] = -1
      #  markers.reshape(-1)[merged_maxima_ids] = torch.arange(1, num_pores + 1, device=device, dtype=torch.int32)
      #  distance_grid = grid_radaii.reshape(nx, ny, nz).clone()
      #  distance_grid.reshape(-1)[geo_grid_occ] = 0


#pack results, safe labels and center distances refer to centers before periodic merging
     #   _out = dict(_input)
      #  _out.update(
       #         distance_grid=distance_grid,
        #        markers=merged_maxima_ids,
         #       pore_markers=markers,
          #      pore_labels=pore_labels.reshape(-1),
           #     num_pores=num_pores,
            #    merged_maxima_ids=merged_maxima_ids,
             #   clusters_radaii=grid_radaii[merged_maxima_ids],
              #  clusters_cart_positions=grid[merged_maxima_ids],
              #  clusters_grid_positions=voxel_positions[merged_maxima_ids],
               # centers_before_pbc=centers_before_pbc,
               # center_distances=center_distances,
                #safe_radaii=safe_radaii,
              #  safe_mask=safe_mask,
               # safe_labels=safe_labels,
             #   competitive_mask=competitive_mask,
             #   pore_labels_before_pbc=pore_labels_before_pbc,
              #  flood_iterations=flood_iterations,
              #  periodic_label_map=periodic_label_map,
              #  periodic_pairs=periodic_pairs
      #  )
       # print('Final pore count:', num_pores, flush=True)
      #  return _out

