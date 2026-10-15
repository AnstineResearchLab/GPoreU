import torch
import triton
import triton.language as tl

@triton.jit
def GPoreU_Lattice_Kernel(
                            coord_ptr,
                            grid_ptr,
                            cell_ptr,
                            grid_radaii_ptr,
                            geo_grid_occ_ptr,
                            he_grid_occ_ptr,
                            n_grid_occ_ptr,
                            he_lj_energy_ptr,
                            n_lj_energy_ptr,
                            NGRID: tl.constexpr,
                            NATOMS: tl.constexpr,
                            sigmas_ptr,
                            he_sigma_ptr,
                            n_sigma_ptr,
                            epsilons_ptr,
                            he_epsilon_ptr,
                            n_epsilon_ptr
                         ):

    '''------ 1 PID PER GRID POINT ------'''
    pid = tl.program_id( 0 )
    if pid >= NGRID:
        return
    
    '''--- LOAD He AND N PROBE PARAMS ---'''
    he_sigma   = tl.load( he_sigma_ptr   )
    he_epsilon = tl.load( he_epsilon_ptr )
    n_sigma    = tl.load( n_sigma_ptr    )
    n_epsilon  = tl.load( n_epsilon_ptr  )

    '''--- LOAD GRID POINT COORDINATES --'''
    a_off = pid * 3  # memory offset
    a0 = tl.load( grid_ptr + a_off + 0 ) # x
    a1 = tl.load( grid_ptr + a_off + 1 ) # y
    a2 = tl.load( grid_ptr + a_off + 2 ) # z


    '''--- LINEAR SCALAR CELL LOADING ---'''
    c00 = tl.load( cell_ptr + 0 )  
    c01 = tl.load( cell_ptr + 1 )  
    c02 = tl.load( cell_ptr + 2 )
    c10 = tl.load( cell_ptr + 3 )  
    c11 = tl.load( cell_ptr + 4 )  
    c12 = tl.load( cell_ptr + 5 )
    c20 = tl.load( cell_ptr + 6 )  
    c21 = tl.load( cell_ptr + 7 )  
    c22 = tl.load( cell_ptr + 8 )

    inv00 = tl.load( cell_ptr + 9  )  
    inv01 = tl.load( cell_ptr + 10 ) 
    inv02 = tl.load( cell_ptr + 11 )
    inv10 = tl.load( cell_ptr + 12 ) 
    inv11 = tl.load( cell_ptr + 13 ) 
    inv12 = tl.load( cell_ptr + 14 )
    inv20 = tl.load( cell_ptr + 15 ) 
    inv21 = tl.load( cell_ptr + 16 ) 
    inv22 = tl.load( cell_ptr + 17 )
    '''--- LINEAR SCALAR CELL LOADING ---'''
 
    LARGE      = 1e10
    min_d      = LARGE
    sigma_min  = LARGE
    prefactor  = 1.122 * 1.122

    geo_check     = False
    he_check      = False
    n_check       = False
    he_lj_energy  = 0.0
    n_lj_energy   = 0.0

    offs_b = tl.arange( 0, 128 ) # WORK IN BLOCKS OF 128
    
    '''--- LOOP OVER ALL ATOMS IN THE CELL ---'''
    for start in range(0, NATOMS, 128):
        
        if geo_check == False:

            idx  = start + offs_b # MEMORY OFFSET FOR ATOM LOADING
            mask = idx < NATOMS   # MASK FOR OUT OF BOUNDS
    
            '''--- LOAD A BLOCK OF ATOM COORDINATES ----'''
            b0 = tl.load( coord_ptr + idx * 3 + 0, mask=mask, other=0.0 )
            b1 = tl.load( coord_ptr + idx * 3 + 1, mask=mask, other=0.0 )
            b2 = tl.load( coord_ptr + idx * 3 + 2, mask=mask, other=0.0 )
   
            '''--- LOAD SIGMAS AND EPSILONS OF ATOMS ---'''
            sigmas   = tl.load( sigmas_ptr  +  idx, mask=mask, other=0.0 )
            epsilons = tl.load( epsilons_ptr + idx, mask=mask, other=0.0 )
    
            '''----- BLOCK OF DISPLACEMENT VECTORS -----'''
            dx = a0 - b0
            dy = a1 - b1
            dz = a2 - b2
    
            '''---- BLOCK OF FRACTIONAL COORDINATES ----'''
            ''' --- row-lattice convention: v = f @ C,  f = d @ inv(C)  ---- '''
            fx = inv00 * dx + inv10 * dy + inv20 * dz
            fy = inv01 * dx + inv11 * dy + inv21 * dz
            fz = inv02 * dx + inv12 * dy + inv22 * dz
    
    
            '''------ TRITON FRIENDLY "ROUNDING" -------'''
            fx = fx - tl.floor( fx + 0.5 )
            fy = fy - tl.floor( fy + 0.5 )
            fz = fz - tl.floor( fz + 0.5 )

            '''-------- PERIODIC DISPLACEMENTS ---------'''
            dx_p = c00 * fx + c10 * fy + c20 * fz
            dy_p = c01 * fx + c11 * fy + c21 * fz
            dz_p = c02 * fx + c12 * fy + c22 * fz
    
            '''------- WRAPPED SQUARED DISTANCE --------'''
            d2 = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p
    
            '''------ MASK DISTANCE OUT OF BOUNDS ------'''
            d2 = tl.where( mask, d2, LARGE )
    
            '''----- ZERO LJ PARAMS OUT OF BOUNDS ------'''
            sigma_masked   = tl.where( mask,   sigmas, 0.0 )
            epsilon_masked = tl.where( mask, epsilons, 0.0 )
    
            '''----- ZERO LJ PARAMS OUT OF BOUNDS ------'''
            s2    = sigma_masked * sigma_masked

            '''----- GEOMETRIC ACCESSIBILITY CHECK -----'''
            d_check = ( d2 < 0.25 * s2 ) # note this is squared verision of d < sig
            any_val = tl.max( d_check )
            if any_val == 1: # geo inaccessible grid point
                geo_check = True 
            else:
                '''----- DO HELIUM ACCESSIBILITY PART -----'''
                if he_check == False:
                    s_he    = 0.5 * ( sigma_masked + he_sigma )
                    s2_he   = s_he * s_he
                    s2_he   = tl.where( mask, s2_he, 0.0 )
                    d_check = ( d2 < 0.25 * s2_he )
                    any_val = tl.max( d_check )

                    '''----- LJ IF ACCESSIBLE -----'''
                    if any_val == 1: # He inaccessible grid point
                        he_check = True

                    else: # do LJ part for He
                        he_eps         = tl.sqrt( epsilon_masked*he_epsilon )
                        s2_he_d2       = s2_he / d2
                        s2_he_d2_6     = s2_he_d2 * s2_he_d2 * s2_he_d2
                        s2_he_d2_12    = s2_he_d2_6 * s2_he_d2_6
                        he_lj_energy  += tl.sum( he_eps * (s2_he_d2_12-s2_he_d2_6) )


                '''----- DO NITROGEN ACCESSIBILITY PART -----'''
                if n_check == False:
                    s_n     = 0.5 * ( sigma_masked + n_sigma )
                    s2_n    = s_n * s_n
                    s2_n    = tl.where( mask, s2_n, 0.0 )
                    d_check = ( d2 < s2_n )
                    any_val = tl.max( d_check )

                    '''----- LJ IF ACCESSIBLE -----'''
                    if any_val == 1: # N inaccessible grid point
                        n_check = True

                    else: # do LJ part for N
                        n_eps         = tl.sqrt( epsilon_masked*n_epsilon )
                        s2_n_d2       = s2_n / d2
                        s2_n_d2_6     = s2_n_d2 * s2_n_d2 * s2_n_d2
                        s2_n_d2_12    = s2_n_d2_6 * s2_n_d2_6
                        n_lj_energy  += tl.sum( n_eps * (s2_n_d2_12-s2_n_d2_6) )

                '''----- GET MIN D2 TO THE FRAMEWORK ------'''
                d = tl.sqrt(d2) - 0.5 * sigma_masked
                block_min = tl.min( d )
                if block_min < min_d: # update if block is less
                    min_d = block_min

    '''----- STORE RESULTS TO OUTPUT MEMORY ------'''
    if geo_check == False:
        tl.store( geo_grid_occ_ptr + pid, False ) # accessible
        tl.store( grid_radaii_ptr  + pid, min_d ) # store distance to nearest atom center
        
        if he_check == True: # inaccessible to He
            tl.store( he_lj_energy_ptr + pid, LARGE ) # some LARGE value to make Boltzmann weight ~0
            tl.store( he_grid_occ_ptr  + pid, True)   # inaccessible
        else:
            tl.store( he_lj_energy_ptr + pid, he_lj_energy ) # regular energies for Bolttzmann weight
            tl.store( he_grid_occ_ptr  + pid, False ) # accessible

        if n_check == True: # inaccessible to N
            tl.store( n_lj_energy_ptr + pid, LARGE ) # some LARGE value to make Boltzmann weight ~0
            tl.store( n_grid_occ_ptr  + pid, True) # inaccesible

        else:
            tl.store( n_lj_energy_ptr + pid, n_lj_energy ) # regular energies for Bolttzmann weight
            tl.store( n_grid_occ_ptr  + pid, False) # accessible

    else:
        tl.store( geo_grid_occ_ptr + pid, True  ) # inaccessible
        tl.store( he_grid_occ_ptr  + pid, True  ) # inaccessible
        tl.store( n_grid_occ_ptr   + pid, True  ) # inaccessible
        tl.store( he_lj_energy_ptr + pid, LARGE ) # some LARGE value to make Boltzmann weight ~0
        tl.store( n_lj_energy_ptr  + pid, LARGE ) # some LARGE value to make Boltzmann weight ~0
        tl.store( grid_radaii_ptr  + pid, 0.0  ) # some small arbitrary min just for book keeping

def all_grids_pbc(_input):

    ''' ------------- UNPACK DICT TO TENSORS ------------------ '''
    coord      =      _input['coord']
    grid       =       _input['grid']
    cell       =       _input['cell']
    sigmas     =     _input['sigmas']
    he_sigma   =   _input['he_sigma']
    n_sigma    =    _input['n_sigma']
    n_epsilon  =  _input['n_epsilon']
    epsilons   =   _input['epsilons']
    he_epsilon = _input['he_epsilon']

    ''' ------------ ASSERTS TO VERIFY ALL CUDA --------------- '''
    assert      coord.is_cuda
    assert       grid.is_cuda
    assert       cell.is_cuda
    assert     sigmas.is_cuda
    assert   he_sigma.is_cuda
    assert    n_sigma.is_cuda
    assert  n_epsilon.is_cuda
    assert   epsilons.is_cuda
    assert he_epsilon.is_cuda

    ''' -------- DEFAULTS AND CONVENIENT CELL FORMAT ---------- '''
    NATOMS = coord.shape[0]
    NGRID  =  grid.shape[0]
    cell_inv = torch.linalg.inv(cell)
    cell_flat = torch.cat([cell.reshape(-1), 
                           cell_inv.reshape(-1)]).contiguous()
    
    ''' ------------- INITIALIZE OUTPUT TENSORS --------------- '''
    grid_radaii   =  torch.empty(NGRID, device='cuda', dtype=torch.float32)
    geo_grid_occ  =     torch.empty(NGRID, device='cuda', dtype=torch.bool)
    he_grid_occ   =     torch.empty(NGRID, device='cuda', dtype=torch.bool)
    n_grid_occ    =     torch.empty(NGRID, device='cuda', dtype=torch.bool)
    he_lj_energy  =  torch.empty(NGRID, device='cuda', dtype=torch.float32)
    n_lj_energy   =  torch.empty(NGRID, device='cuda', dtype=torch.float32)

    ''' ------------- CALL FUSED KERNEL --------------- '''
    GPoreU_Lattice_Kernel[(NGRID,)](
                                     coord,
                                     grid,
                                     cell_flat,
                                     grid_radaii,
                                     geo_grid_occ,
                                     he_grid_occ,
                                     n_grid_occ,
                                     he_lj_energy,
                                     n_lj_energy,
                                     NGRID,
                                     NATOMS,
                                     sigmas,
                                     he_sigma,
                                     n_sigma,
                                     epsilons,
                                     he_epsilon,
                                     n_epsilon, 
                                     num_warps=4
                                   )

    ''' ------- NICE PLACE FOR PORE VOLUME CALC --------- '''
    vol = _input['volume']
    he_pore_volume  = vol * ( -4 * he_lj_energy / 298.0 ).exp().sum() / NGRID
    n_pore_volume   = vol * ( -4 * n_lj_energy  / 298.0 ).exp().sum() / NGRID
    geo_pore_volume = vol * (~geo_grid_occ).sum() / NGRID
    

    ''' ------------- PACK OUTPUT TO DICT --------------- '''
    _out = dict(grid_radaii      =      grid_radaii,
                geo_grid_occ     =     geo_grid_occ,
                he_grid_occ      =      he_grid_occ,
                n_grid_occ       =       n_grid_occ,
                he_lj_energy     =     he_lj_energy,
                n_lj_energy      =      n_lj_energy,
                he_pore_volume   =   he_pore_volume,
                geo_pore_volume  =  geo_pore_volume)
    
    return _out
