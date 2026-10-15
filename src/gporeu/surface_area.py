import torch
import triton
import triton.language as tl
from . import prof


@triton.jit
def GPoreU_SA_Kernel2(
    coord_ptr,
    sigmas_ptr,
    rand1_ptr,
    rand2_ptr,
    cell_ptr,
    n_sigma_ptr,
    accessible_count_ptr,
    surface_radius_ptr,
    NATOMS: tl.constexpr,
    SAMP: tl.constexpr,
    BLOCK_ATOMS: tl.constexpr,
):
    '''Blocked fraction of 512 Fibonacci-free sphere samples around each
    atom (same random draws as the original GPoreU_SA_Kernel), with the
    dead n-grid lookups removed and a spill-free [SAMP, BLOCK_ATOMS] tile.
    2D program grid: (atom, sample chunk); per-atom counts accumulate with
    integer-valued atomic adds, so accessible_count is bit-identical to the
    single-program-per-atom original.  Blocking rule unchanged: a surface
    point at 1.122*sigma_mix from the host atom is blocked iff its wrapped
    distance to ANY other framework atom j is below 1.122*0.5*(sigma_j +
    sigma_N); the host atom itself never blocks (idx != pid).'''
    pid = tl.program_id(0)
    chunk = tl.program_id(1)

    a_off = pid * 3
    x = tl.load(coord_ptr + a_off + 0)
    y = tl.load(coord_ptr + a_off + 1)
    z = tl.load(coord_ptr + a_off + 2)

    sigmas_mix = tl.load(sigmas_ptr + pid)
    n_sigma = tl.load(n_sigma_ptr)
    sigma_mix = 0.5 * (sigmas_mix + n_sigma)
    surface_radius = 1.122 * sigma_mix

    '''--- SAMPLES OF THIS CHUNK (of 512 per atom) ---------'''
    point_offs = chunk * SAMP + tl.arange(0, SAMP)[:, None]
    rand_offset = pid * 512
    u = tl.load(rand1_ptr + rand_offset + point_offs)
    v = tl.load(rand2_ptr + rand_offset + point_offs)

    phi = 2.0 * 3.141592653589793 * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = tl.sqrt(tl.maximum(0.0, 1.0 - cos_theta * cos_theta))

    a0 = sin_theta * tl.cos(phi)
    a1 = sin_theta * tl.sin(phi)
    a2 = cos_theta

    surface_point_x = a0 * 1.122 * sigma_mix + x
    surface_point_y = a1 * 1.122 * sigma_mix + y
    surface_point_z = a2 * 1.122 * sigma_mix + z

    '''--- CELL + INVERSE (SCALARS, LOADED ONCE) -----------'''
    c00 = tl.load(cell_ptr + 0)
    c01 = tl.load(cell_ptr + 1)
    c02 = tl.load(cell_ptr + 2)
    c10 = tl.load(cell_ptr + 3)
    c11 = tl.load(cell_ptr + 4)
    c12 = tl.load(cell_ptr + 5)
    c20 = tl.load(cell_ptr + 6)
    c21 = tl.load(cell_ptr + 7)
    c22 = tl.load(cell_ptr + 8)

    inv00 = tl.load(cell_ptr + 9)
    inv01 = tl.load(cell_ptr + 10)
    inv02 = tl.load(cell_ptr + 11)
    inv10 = tl.load(cell_ptr + 12)
    inv11 = tl.load(cell_ptr + 13)
    inv12 = tl.load(cell_ptr + 14)
    inv20 = tl.load(cell_ptr + 15)
    inv21 = tl.load(cell_ptr + 16)
    inv22 = tl.load(cell_ptr + 17)

    LARGE = 1e10
    prefactor = 1.122 * 1.122

    blocked = tl.zeros((SAMP, 1), dtype=tl.int32)
    offs_b = tl.arange(0, BLOCK_ATOMS)[None, :]

    for start in range(0, NATOMS, BLOCK_ATOMS):
        '''--- SKIP REMAINING BLOCKS ONCE ALL SAMPLES BLOCKED ---'''
        if tl.sum((blocked == 0).to(tl.int32), axis=0) > 0:
            idx = start + offs_b
            mask = idx < NATOMS

            b0 = tl.load(coord_ptr + idx * 3 + 0, mask=mask, other=0.0)
            b1 = tl.load(coord_ptr + idx * 3 + 1, mask=mask, other=0.0)
            b2 = tl.load(coord_ptr + idx * 3 + 2, mask=mask, other=0.0)
            sigmas = tl.load(sigmas_ptr + idx, mask=mask, other=0.0)

            dx = surface_point_x - b0
            dy = surface_point_y - b1
            dz = surface_point_z - b2

            fx = inv00 * dx + inv10 * dy + inv20 * dz
            fy = inv01 * dx + inv11 * dy + inv21 * dz
            fz = inv02 * dx + inv12 * dy + inv22 * dz

            fx = fx - tl.floor(fx + 0.5)
            fy = fy - tl.floor(fy + 0.5)
            fz = fz - tl.floor(fz + 0.5)

            dx_p = c00 * fx + c10 * fy + c20 * fz
            dy_p = c01 * fx + c11 * fy + c21 * fz
            dz_p = c02 * fx + c12 * fy + c22 * fz

            d2 = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p
            d2 = tl.where(mask, d2, LARGE)

            sigma_masked = tl.where(mask, sigmas, 0.0)
            s_n = 0.5 * (sigma_masked + n_sigma)
            s2_n = s_n * s_n
            s2_n = tl.where(mask, s2_n, 0.0)
            d2_n = prefactor * s2_n

            comparison_mask = mask & (idx != pid)
            d_check = comparison_mask & (d2 < d2_n)
            blocked = blocked | tl.max(d_check.to(tl.int32), axis=1,
                                       keep_dims=True)

    accessible = blocked == 0
    partial = tl.sum(accessible.to(tl.int32), axis=0)
    partial = tl.sum(partial, axis=0)
    tl.atomic_add(accessible_count_ptr + pid, partial.to(tl.float32))
    tl.store(surface_radius_ptr + pid, surface_radius)

@triton.jit
def GPoreU_SA_Kernel(
    coord_ptr,
    sigmas_ptr,
    rand1_ptr,
    rand2_ptr,
    cell_ptr,
    n_sigma_ptr,
    n_grid_occ_ptr,
    accessible_count_ptr,
    surface_radius_ptr,
    NATOMS: tl.constexpr,
    BLOCK_POINTS: tl.constexpr,
    BLOCK_ATOMS: tl.constexpr,
    nx,
    ny,
    nz,
):
    # Load program as one program id handles one atom  point and a random point
    '''------ 1 PID PER ATOM ------------------------------'''
    pid = tl.program_id( 0 )
   
    '''--- LOAD AN ATOM FOR SURFACE AREA CHECK AROUND IT --'''
    a_off = pid * 3  # memory offset
    x  = tl.load( coord_ptr + a_off + 0 ) # x
    y  = tl.load( coord_ptr + a_off + 1 ) # y
    z  = tl.load( coord_ptr + a_off + 2 ) # z

    '''--- LOAD THE TWO RANDOM NUMBERS --------------------'''
     # Shape: [BLOCK_POINTS, 1]
    point_offs = tl.arange(0, 512)[:, None]
    rand_offset = pid * 512
    u = tl.load(rand1_ptr + rand_offset + point_offs)
    v = tl.load(rand2_ptr + rand_offset + point_offs)

    # Convert to spherical coordinates
    phi = 2.0 *3.141592653589793  * u
    cos_theta = 2.0 * v - 1.0
    sin_theta = tl.sqrt(tl.maximum(0.0, 1.0 - cos_theta * cos_theta))


    # Unit vector on the sphere
    a0 = sin_theta * tl.cos(phi)
    a1 = sin_theta * tl.sin(phi)
    a2 = cos_theta
    # centering around the atom

    '''--- PARAMETERS FOR ATOM ---------------------------'''
    sigmas_mix  =tl.load(sigmas_ptr+pid)

    '''--- LOAD He AND N PROBE PARAMS --------------------'''
    n_sigma    = tl.load( n_sigma_ptr    )

    '''---- SURFACE AREA POINT ---------------------------'''
    sigma_mix=0.5*(sigmas_mix+n_sigma)

    surface_point_x= (a0 * 1.122 *sigma_mix + x)
    surface_point_y= (a1 * 1.122 *sigma_mix + y)
    surface_point_z= (a2 * 1.122 *sigma_mix + z)
    surface_radius = (1.122 *sigma_mix)
    '''--- LINEAR SCALAR CELL LOADING --------------------'''
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

    '''--- CREATE TEMP. DATA STRUCS ----------------------'''
    LARGE      = 1e10
    prefactor  = 1.122 * 1.122
    
    # One Boolean value for each surface point
   
    blocked = tl.zeros((512, 1), dtype=tl.int32)
    # Shape: [1, BLOCK_ATOMS]
    offs_b = tl.arange(0, BLOCK_ATOMS)[None, :]

    '''--- LOOP OVER ALL ATOMS IN THE CELL ---------------'''
    for start in range(0, NATOMS, BLOCK_ATOMS):

            idx  = start + offs_b # MEMORY OFFSET FOR ATOM LOADING
            mask = idx < NATOMS   # MASK FOR OUT OF BOUNDS

            '''--- LOAD A BLOCK OF ATOM COORDINATES ----'''
            b0 = tl.load( coord_ptr + idx * 3 + 0, mask=mask, other=0.0 )
            b1 = tl.load( coord_ptr + idx * 3 + 1, mask=mask, other=0.0 )
            b2 = tl.load( coord_ptr + idx * 3 + 2, mask=mask, other=0.0 )

            '''--- LOAD SIGMAS OF ATOMS ----------------'''
            sigmas   = tl.load( sigmas_ptr  +  idx, mask=mask, other=0.0 )

            '''----- BLOCK OF DISPLACEMENT VECTORS -----'''
            dx = surface_point_x - b0
            dy = surface_point_y - b1
            dz = surface_point_z - b2

            '''---- BLOCK OF FRACTIONAL COORDINATES ----'''
            fx = inv00 * dx + inv10 * dy + inv20 * dz
            fy = inv01 * dx + inv11 * dy + inv21 * dz
            fz = inv02 * dx + inv12 * dy + inv22 * dz
            '''- BLOCK OF FRACTIONAL COORDINATES RN ----'''
            gx = inv00 * surface_point_x + inv10 *surface_point_y + inv20 *surface_point_z
            gy = inv01 * surface_point_x + inv11 *surface_point_y + inv21 *surface_point_z
            gz = inv02 * surface_point_x + inv12 *surface_point_y + inv22 *surface_point_z


            '''------ TRITON FRIENDLY "ROUNDING" -------'''
            fx = fx - tl.floor( fx + 0.5 )
            fy = fy - tl.floor( fy + 0.5 )
            fz = fz - tl.floor( fz + 0.5 )
  
            gx = gx - tl.floor( gx )
            gy = gy - tl.floor( gy )
            gz = gz - tl.floor( gz )

            # Find containing cubelet
            ix = tl.floor(gx * nx).to(tl.int32)
            iy = tl.floor(gy * ny).to(tl.int32)
            iz = tl.floor(gz * nz).to(tl.int32)

            # Flatten 3D grid index
            grid_idx = ix * (ny * nz) + iy * nz + iz

            # One Boolean for each surface point
            grid_accessible = tl.load(n_grid_occ_ptr + grid_idx)

            # Inaccessible grid location means already blocked
       
            n_grid_occ = tl.load(n_grid_occ_ptr + grid_idx)
            n_blocked = n_grid_occ.to(tl.int32)
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

            '''----- ZERO LJ PARAMS OUT OF BOUNDS ------'''
            s2    = sigma_masked * sigma_masked

            '''----- DO NITROGEN ACCESSIBILITY PART ----'''
            s_n     = 0.5 * ( sigma_masked + n_sigma )
            s2_n    = s_n * s_n
            s2_n    = tl.where( mask, s2_n, 0.0 )
            d2_n    = prefactor *s2_n
            comparison_mask = mask & (idx != pid)
            d_check = comparison_mask & (d2 < d2_n)
            overlap = tl.max(d_check.to(tl.int32), axis=1, keep_dims=True)
            blocked = blocked | overlap

    '''---- COUNT ACCESSIBLE POINTS ----------------------'''
    accessible = blocked == 0
    accessible_count = tl.sum( accessible.to(tl.int32), axis=0)
    accessible_count = tl.sum(accessible_count,axis=0)
    '''----- STORE RESULTS TO OUTPUT MEMORY --------------'''
    
    tl.store( accessible_count_ptr + pid, accessible_count)
    tl.store( surface_radius_ptr + pid, surface_radius)

def GPoreU_SA(_input):

    '''------------- UNPACK DICT TO TENSORS --------------'''
    coord      =      _input['coord']
    cell       =       _input['cell']
    sigmas     =     _input['sigmas']
    n_sigma    =    _input['n_sigma']
    n_grid_occ = _input['n_grid_occ']
    nx         =         _input['nx']
    ny         =         _input['ny']
    nz         =         _input['nz']
    trials     =     _input['trials']
    '''------------ ASSERTS TO VERIFY ALL CUDA -----------'''
    assert      coord.is_cuda
    assert       cell.is_cuda
    assert     sigmas.is_cuda
    assert    n_sigma.is_cuda
    assert n_grid_occ.is_cuda
    '''------- DEFAULTS AND CONVENIENT CELL FORMAT -------'''
    NATOMS = coord.shape[0]
    cell_inv = torch.linalg.inv(cell)
    cell_flat = torch.cat([cell.reshape(-1),
                           cell_inv.reshape(-1)]).contiguous()

    '''------------- INITIALIZE OUTPUT TENSORS -----------'''
    accessible_count     = torch.zeros( NATOMS, device='cuda', dtype=torch.float32)
    surface_radius       = torch.empty( NATOMS, device="cuda", dtype=torch.float32)
    '''---------RANDOM POINT GENERATION ---------'''

    rand1 = torch.rand(_input['trials'], device='cuda')
    rand2 = torch.rand(_input['trials'], device='cuda')

    '''------------ CALL FUSED KERNEL (2D atom x sample-chunk) ----'''
    SAMP = 64
    grid = (NATOMS, 512 // SAMP)

    block_atoms = 128
    GPoreU_SA_Kernel2[grid](
                                coord,
                                sigmas,
                                rand1,
                                rand2,
                                cell_flat,
                                n_sigma,
                                accessible_count,
                                surface_radius,
                                NATOMS=NATOMS,
                                SAMP=SAMP,
                                BLOCK_ATOMS=block_atoms,
                                num_warps=4
                              )
    points_per_atom = 512

    sfrac = accessible_count.float() / points_per_atom

    surface_area_per_atom = ( 4.0 * torch.pi * surface_radius**2)

    full_surface_area = torch.sum(surface_area_per_atom)

    accessible_surface_area_per_atom = (surface_area_per_atom * sfrac)

    total_surface_area = torch.sum(accessible_surface_area_per_atom)

    prof.vprint("--------------- SURFACE AREA DEBUG ---------------")

    prof.vprint("NATOMS:", NATOMS)

    prof.vprint("rand1 numel:", rand1.numel())
    prof.vprint("rand2 numel:", rand2.numel())
    prof.vprint("expected random values:", NATOMS * points_per_atom)

    prof.vprint("accessible_count shape:", accessible_count.shape)
    prof.vprint("surface_radius shape:", surface_radius.shape)

    prof.vprint("accessible count min:", accessible_count.min().item())
    prof.vprint("accessible count max:", accessible_count.max().item())
    prof.vprint("accessible count mean:", accessible_count.mean().item())

    prof.vprint("surface fraction min:", sfrac.min().item())
    prof.vprint("surface fraction max:", sfrac.max().item())
    prof.vprint("surface fraction mean:", sfrac.mean().item())

    prof.vprint("surface radius min:", surface_radius.min().item())
    prof.vprint("surface radius max:", surface_radius.max().item())
    prof.vprint("surface radius mean:", surface_radius.mean().item())

    prof.vprint("full surface area:", full_surface_area.item())
    prof.vprint("accessible surface area:", total_surface_area.item())

    prof.vprint( "overall accessible-area ratio:",(total_surface_area / full_surface_area).item())

    prof.vprint("first 20 accessible counts:")
    prof.vprint(accessible_count[:20])

    prof.vprint("first 20 surface fractions:")
    prof.vprint(sfrac[:20])

    prof.vprint("first 20 surface radii:")
    prof.vprint(surface_radius[:20])

    prof.vprint("first 20 full atom areas:")
    prof.vprint(surface_area_per_atom[:20])

    prof.vprint("first 20 accessible atom areas:")
    prof.vprint(accessible_surface_area_per_atom[:20])

    prof.vprint("--------------------------------------------------")
    '''------------ PACK OUTPUT TO DICT ------------------'''
    _out = dict(total_surface_area= total_surface_area)
    return _out
