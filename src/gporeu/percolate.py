import torch

from .uf_graph import components

def GPoreU_percolate(_input):

    ''' ------------- UNPACK DICT TO TENSORS ------------------ '''
    nx           =           _input['nx']
    ny           =           _input['ny']
    nz           =           _input['nz']
    geo_grid_occ = _input['geo_grid_occ'].view(nx, ny, nz)
    n_grid_occ   =   _input['n_grid_occ'].view(nx, ny, nz)
    he_grid_occ  =  _input['he_grid_occ'].view(nx, ny, nz)
    he_lj_energy = _input['he_lj_energy']
    volume       =       _input['volume']

    ''' ------------ ASSERTS TO VERIFY ALL CUDA --------------- '''
    assert geo_grid_occ.is_cuda
    assert   n_grid_occ.is_cuda
    assert  he_grid_occ.is_cuda
    assert  he_lj_energy.is_cuda

    ''' --------- CALCULATE ALL IJ NEIGHBORS GRIDS ------------ '''
    ij_geo = ij_pairs(geo_grid_occ)
    ij_he  =  ij_pairs(he_grid_occ)
    ij_n   =   ij_pairs(n_grid_occ)
    ij_geo_periodic = periodic_ij_pairs(geo_grid_occ)
    ''' ------- CALCULATE ALL CLUSTERS FROM IJ NEIGHS --------- '''
    geo_clusters, geo_perc_x, geo_perc_y, geo_perc_z =  get_clusters(ij_geo, nx, ny, nz)
    he_clusters,  he_perc_x,  he_perc_y,  he_perc_z  =  get_clusters(ij_he,  nx, ny, nz)
    n_clusters,   n_perc_x,   n_perc_y,    n_perc_z  =  get_clusters(ij_n,   nx, ny, nz)

    ''' ----------  HELIUM PORE VOLUME ------------------------ '''
    if he_perc_x or he_perc_y or he_perc_z:
           access_he_pore_volume= he_pore_volume(he_lj_energy, he_clusters, volume)
    else:
           access_he_pore_volume=0.0
    ''' ----------  GEOMETRIC  PORE VOLUME ------------------------ '''
    if geo_perc_x or geo_perc_y or geo_perc_z:
          access_geo_pore_volume=geo_volume(geo_clusters, volume)
    else:
         access_geo_pore_volume=0.0
    ''' ------------- PACK OUTPUT TO DICT --------------- '''
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

    ''' ----------- STRIDES FOR LINEAR INDEX CALC ------------- '''
    x_stride, y_stride, z_stride = O.stride()

    ''' ----------- LOOK TO THE RIGHT ------------- '''
    O_rolled = O.roll(-1,0)
    O_combined = ((O==False) & (O_rolled==False))
    _ids = torch.argwhere(O_combined[:-1,:,:])
    _ix = _ids[:,0] * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride
    _jx = (_ids[:,0]+1) * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride

    ''' ----------- LOOK IN FRONT OF -------------- '''
    O_rolled = O.roll(-1,1)
    O_combined = ((O==False) & (O_rolled==False))
    _ids = torch.argwhere(O_combined[:,:-1,:])
    _iy = _ids[:,0] * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride
    _jy = (_ids[:,0]) * x_stride + (_ids[:,1]+1) * y_stride + _ids[:,2] * z_stride

    ''' -------------- LOOK ABOVE ----------------- '''
    O_rolled = O.roll(-1,2)
    O_combined = ((O==False) & (O_rolled==False))
    _ids = torch.argwhere(O_combined[:,:,:-1])
    _iz = _ids[:,0] * x_stride + _ids[:,1] * y_stride + _ids[:,2] * z_stride
    _jz = _ids[:,0] * x_stride + _ids[:,1] * y_stride + (_ids[:,2]+1) * z_stride

    ''' ---------- FORMAT ALL IJ PAIRS ------------ '''
    _i = torch.concatenate([_ix,_iy,_iz])
    _j = torch.concatenate([_jx,_jy,_jz])
    pairs = torch.concatenate([_i.view(-1,1),_j.view(-1,1)], -1).to(torch.int32)

    return pairs

def periodic_ij_pairs(O: torch.Tensor):

    ''' ----------- STRIDES FOR LINEAR INDEX CALC ------------- '''
    x_stride, y_stride, z_stride = O.stride()

    ''' ----------- X-EDGE CHECK ------------- '''
    x_check = ((O[-1,:,:]==False) & (O[0,:,:]==False))
    _ids = torch.argwhere(x_check)
    _ix = (O.shape[0]-1) * x_stride + _ids[:,0] * y_stride + _ids[:,1] * z_stride
    _jx =  _ids[:,0] * y_stride + _ids[:,1] * z_stride

    ''' ----------- Y-EDGE CHECK ----------------------- '''
    y_check = ((O[:,-1,:]==False) & (O[:,0,:]==False))
    _ids = torch.argwhere(y_check)

    _iy = _ids[:,0] * x_stride + (O.shape[1]-1) * y_stride + _ids[:,1] * z_stride
    _jy = _ids[:,0] * x_stride + _ids[:,1] * z_stride


    ''' ----------- Z-EDGE CHECK ----------------------- '''
    z_check = ((O[:,:,-1]==False) & (O[:,:,0]==False))
    _ids = torch.argwhere(z_check)

    _iz = _ids[:,0] * x_stride + _ids[:,1] * y_stride + (O.shape[2]-1) * z_stride
    _jz = _ids[:,0] * x_stride + _ids[:,1] * y_stride


    ''' ---------- FORMAT PERIODIC IJ PAIRS ------------------- '''
    _i = torch.concatenate([_ix,_iy,_iz])
    _j = torch.concatenate([_jx,_jy,_jz])

    pairs = torch.concatenate([_i.view(-1,1),_j.view(-1,1)], -1).to(torch.int32)

    return pairs

def get_clusters(ij_pairs: torch.Tensor,
                 nx: int,
                 ny: int,
                 nz: int):


    ''' ---------- LABEL CLUSTERS WITH Triton UNION-FIND ------- '''
    # connected-component roots; cluster ids are only ever compared for
    # equality below.  Nodes that appear in NO edge (occupied, or isolated
    # accessible points) must carry the -1 sentinel: the face-percolation
    # test counts unique labels shared between opposite faces and declares
    # percolation only when that set has MORE THAN ONE member, which relies
    # on -1 (both faces) occupying the set.  This reproduces the old cugraph
    # convention bit-for-bit.
    N = nx * ny * nz
    roots = components(N, ij_pairs[:, 0], ij_pairs[:, 1])
    in_edges = torch.zeros(N, dtype=torch.bool, device=ij_pairs.device)
    in_edges[
        torch.cat([ij_pairs[:, 0], ij_pairs[:, 1]]).to(torch.long)
    ] = True
    clusters = torch.where(
        in_edges, roots, torch.full_like(roots, -1)
    ).view(nx, ny, nz)

    ''' ----- CHECK PERCOLATION WITH TORCH UNIQUE ------ '''
    ''' X '''
    lower_face = (clusters[0,:,:]).view(-1)
    upper_face = (clusters[-1,:,:]).view(-1)
    percolated_x_clusters = torch.unique(lower_face[torch.isin(lower_face, upper_face)])
    if len(percolated_x_clusters) > 1:
        perc_in_x = True
    else:
        perc_in_x = False

    ''' Y '''
    lower_face = (clusters[:,0,:]).reshape(-1)
    upper_face = (clusters[:,-1,:]).reshape(-1)
    percolated_y_clusters = torch.unique(lower_face[torch.isin(lower_face, upper_face)])
    if len(percolated_y_clusters) > 1:
        perc_in_y = True
    else:
        perc_in_y = False

    ''' Z '''
    lower_face = (clusters[:,:,0]).view(-1)
    upper_face = (clusters[:,:,-1]).view(-1)
    percolated_z_clusters = torch.unique(lower_face[torch.isin(lower_face, upper_face)])
    if len(percolated_z_clusters) > 1:
        perc_in_z = True
    else:
        perc_in_z = False


    ''' --- DEFINE ACCESSIBLE / INACCESIBLE GRID ---- '''
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

    # convert to boolean, for ~4x better memory
    clusters = clusters.to(torch.bool)

  
    return clusters, perc_in_x, perc_in_y, perc_in_z

def he_pore_volume(
    he_lj_energy: torch.Tensor,
    he_access: torch.Tensor,
    volume
):
    NGRID = he_access.numel()
    
    access_he_pore_volume = (
        volume
        * torch.exp(-4.0 * he_lj_energy[he_access] / 298.0).sum()
        / NGRID
    )

    return access_he_pore_volume
def geo_volume(geo_access, volume):
    NGRID=geo_access.numel()
    nz_geo_access=geo_access[geo_access>0]
    nz_geo_access_gridpoints=nz_geo_access.numel()
    geo_access_volume=volume*nz_geo_access_gridpoints/NGRID
    return geo_access_volume
