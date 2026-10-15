"""
Connectivity and accessibility.

Builds periodic nearest-neighbour bond lists (interior and wrapping face
pairs), clusters accessible lattice points (Triton union-find), decides
per-direction percolation from labels shared between opposite faces, and
computes the helium-accessible and geometric pore volumes.
"""

import torch

from .union_find import components

def GPoreU_percolate(_input):

    """Connectivity stage.

        Builds the interior and periodic neighbour pair lists, clusters the
        accessible points (Triton union-find), decides per-direction
        percolation for the He, N2 and geometric masks, and computes the
        helium-accessible and geometric pore volumes.  Returns a dict of the
        pair lists, cluster labels, percolation flags and volumes.
    """


    # Unpack the pipeline dict.
    nx           =           _input['nx']
    ny           =           _input['ny']
    nz           =           _input['nz']
    geo_grid_occ = _input['geo_grid_occ'].view(nx, ny, nz)
    n_grid_occ   =   _input['n_grid_occ'].view(nx, ny, nz)
    he_grid_occ  =  _input['he_grid_occ'].view(nx, ny, nz)
    he_lj_energy = _input['he_lj_energy']
    volume       =       _input['volume']

    # Require CUDA tensors.
    assert geo_grid_occ.is_cuda
    assert   n_grid_occ.is_cuda
    assert  he_grid_occ.is_cuda
    assert  he_lj_energy.is_cuda

    # Nearest-neighbour bond lists (interior + periodic).
    ij_geo = ij_pairs(geo_grid_occ)
    ij_he  =  ij_pairs(he_grid_occ)
    ij_n   =   ij_pairs(n_grid_occ)
    ij_geo_periodic = periodic_ij_pairs(geo_grid_occ)
    # Connected components over the accessible lattice.
    geo_clusters, geo_perc_x, geo_perc_y, geo_perc_z =  get_clusters(ij_geo, nx, ny, nz)
    he_clusters,  he_perc_x,  he_perc_y,  he_perc_z  =  get_clusters(ij_he,  nx, ny, nz)
    n_clusters,   n_perc_x,   n_perc_y,    n_perc_z  =  get_clusters(ij_n,   nx, ny, nz)

    # Helium-accessible pore volume.
    if he_perc_x or he_perc_y or he_perc_z:
           access_he_pore_volume= he_pore_volume(he_lj_energy, he_clusters, volume)
    else:
           access_he_pore_volume=0.0
    # Geometric pore volume.
    if geo_perc_x or geo_perc_y or geo_perc_z:
          access_geo_pore_volume=geo_volume(geo_clusters, volume)
    else:
         access_geo_pore_volume=0.0
    # Collect results.
    _out = dict(ij_geo                =                ij_geo,
                ij_geo_periodic       =       ij_geo_periodic,
                geo_access            =          geo_clusters,
                he_access             =           he_clusters,
                n_access              =            n_clusters,
                geo_perc_x            =            geo_perc_x,
                geo_perc_y            =            geo_perc_y,
                geo_perc_z            =            geo_perc_z,
                he_perc_x             =             he_perc_x,
                he_perc_y             =             he_perc_y,
                he_perc_z             =             he_perc_z,
                n_perc_x              =              n_perc_x,
                n_perc_y              =              n_perc_y,
                n_perc_z              =              n_perc_z,
                access_he_pore_volume = access_he_pore_volume,
                access_geo_pore_volume=access_geo_pore_volume)
    return _out

def ij_pairs(O: torch.Tensor):

    """Adjacent (linear-index) pairs along +x, +y, +z of mask O.
    """


    # Strides for the linear (x-major) index.
    x_stride, y_stride, z_stride = O.stride()

    # Neighbours in +x.
    O_rolled = O.roll(-1,0)
    O_combined = ((O==False) & (O_rolled==False))
    _ids = torch.argwhere(O_combined[:-1,:,:])
    _ix = _ids[:,0] * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride
    _jx = (_ids[:,0]+1) * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride

    # Neighbours in +y.
    O_rolled = O.roll(-1,1)
    O_combined = ((O==False) & (O_rolled==False))
    _ids = torch.argwhere(O_combined[:,:-1,:])
    _iy = _ids[:,0] * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride
    _jy = (_ids[:,0]) * x_stride + (_ids[:,1]+1) * y_stride + _ids[:,2] * z_stride

    # Neighbours in +z.
    O_rolled = O.roll(-1,2)
    O_combined = ((O==False) & (O_rolled==False))
    _ids = torch.argwhere(O_combined[:,:,:-1])
    _iz = _ids[:,0] * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride
    _jz = _ids[:,0] * x_stride + _ids[:,1] * y_stride + (_ids[:,2]+1) * z_stride

    # Stack into (i, j) pair arrays.
    _i = torch.concatenate([_ix,_iy,_iz])
    _j = torch.concatenate([_jx,_jy,_jz])
    pairs = torch.concatenate([_i.view(-1,1),_j.view(-1,1)], -1).to(torch.int32)

    return pairs

def periodic_ij_pairs(O: torch.Tensor):

    """Wrap pairs joining opposite faces of mask O along each axis.
    """


    # Strides for the linear (x-major) index.
    x_stride, y_stride, z_stride = O.stride()

    # Wrap pairs across the x faces.
    x_check = ((O[-1,:,:]==False) & (O[0,:,:]==False))
    _ids = torch.argwhere(x_check)
    _ix = (O.shape[0]-1) * x_stride + _ids[:,0] * y_stride + _ids[:,1] * z_stride
    _jx =  _ids[:,0] * y_stride + _ids[:,1] * z_stride

    # Wrap pairs across the y faces.
    y_check = ((O[:,-1,:]==False) & (O[:,0,:]==False))
    _ids = torch.argwhere(y_check)

    _iy = _ids[:,0] * x_stride + (O.shape[1]-1) * y_stride + _ids[:,1] * z_stride
    _jy = _ids[:,0] * x_stride + _ids[:,1] * z_stride


    # Wrap pairs across the z faces.
    z_check = ((O[:,:,-1]==False) & (O[:,:,0]==False))
    _ids = torch.argwhere(z_check)

    _iz = _ids[:,0] * x_stride + _ids[:,1] * y_stride + (O.shape[2]-1) * z_stride
    _jz = _ids[:,0] * x_stride + _ids[:,1] * y_stride


    # Stack the periodic pair arrays.
    _i = torch.concatenate([_ix,_iy,_iz])
    _j = torch.concatenate([_jx,_jy,_jz])

    pairs = torch.concatenate([_i.view(-1,1),_j.view(-1,1)], -1).to(torch.int32)

    return pairs

def get_clusters(ij_pairs: torch.Tensor,
                 nx: int,
                 ny: int,
                 nz: int):

    """Cluster labels for all lattice points via Triton union-find;
        points in no edge carry the -1 sentinel (see face-percolation test).
    """



    # Label clusters (Triton union-find).
    # Connected-component roots; cluster ids are only ever compared for
    # equality below.  Nodes that appear in NO edge (occupied, or isolated
    # accessible points) carry the -1 sentinel: the face-percolation test
    # counts unique labels shared between opposite faces and declares
    # percolation only when that set has MORE THAN ONE member, which
    # relies on -1 (present on both faces) occupying the set.
    N = nx * ny * nz
    roots = components(N, ij_pairs[:, 0], ij_pairs[:, 1])
    in_edges = torch.zeros(N, dtype=torch.bool, device=ij_pairs.device)
    in_edges[
        torch.cat([ij_pairs[:, 0], ij_pairs[:, 1]]).to(torch.long)
    ] = True
    clusters = torch.where(
        in_edges, roots, torch.full_like(roots, -1)
    ).view(nx, ny, nz)

    # Percolation test: labels shared between opposite faces.
    # X axis
    lower_face = (clusters[0,:,:]).view(-1)
    upper_face = (clusters[-1,:,:]).view(-1)
    percolated_x_clusters = torch.unique(lower_face[torch.isin(lower_face, upper_face)])
    if len(percolated_x_clusters) > 1:
        perc_in_x = True
    else:
        perc_in_x = False

    # Y axis
    lower_face = (clusters[:,0,:]).reshape(-1)
    upper_face = (clusters[:,-1,:]).reshape(-1)
    percolated_y_clusters = torch.unique(lower_face[torch.isin(lower_face, upper_face)])
    if len(percolated_y_clusters) > 1:
        perc_in_y = True
    else:
        perc_in_y = False

    # Z axis
    lower_face = (clusters[:,:,0]).view(-1)
    upper_face = (clusters[:,:,-1]).view(-1)
    percolated_z_clusters = torch.unique(lower_face[torch.isin(lower_face, upper_face)])
    if len(percolated_z_clusters) > 1:
        perc_in_z = True
    else:
        perc_in_z = False


    # Accessible / inaccessible masks.
    percolated_clusters = torch.unique(
                                        torch.concatenate([percolated_x_clusters,
                                                           percolated_y_clusters,
                                                           percolated_z_clusters])
                                      )
    # drop -1 (i.e., occupied spaces)
    mask = (percolated_clusters != -1)
    percolated_clusters = percolated_clusters[mask]

    clusters = clusters.view(-1)
    mask = torch.isin(clusters, percolated_clusters)

    # define 1 to be "is percolated"
    clusters[mask] = 1

    #define 0 to be "is not percolated"
    clusters[~mask] = 0

    # Boolean masks (~4x less memory than int64 labels).
    clusters = clusters.to(torch.bool)

  
    return clusters, perc_in_x, perc_in_y, perc_in_z

def he_pore_volume(
    he_lj_energy: torch.Tensor,
    he_access: torch.Tensor,
    volume
):

    """Helium-accessible pore volume from the soft Boltzmann
        accessibility field.
    """

    NGRID = he_access.numel()
    
    access_he_pore_volume = (
        volume
        * torch.exp(-4.0 * he_lj_energy[he_access] / 298.0).sum()
        / NGRID
    )

    return access_he_pore_volume
def geo_volume(geo_access, volume):

    """Geometric pore volume from the count of accessible voxels.
    """

    NGRID=geo_access.numel()
    nz_geo_access=geo_access[geo_access>0]
    nz_geo_access_gridpoints=nz_geo_access.numel()
    geo_access_volume=volume*nz_geo_access_gridpoints/NGRID
    return geo_access_volume
