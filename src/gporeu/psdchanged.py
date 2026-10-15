import math
import torch
import triton
import triton.language as tl
from . import prof

@triton.jit
def GPoreU_PSD_Kernel_2D(
                       N_CAND,
                       N_RP,
                       rand_points_ptr,
                       cand_ptr,          # compacted indices of n-accessible
                                          # points with radius > 0
                       pore_out_ptr,
                       cell_ptr,
                       grid_ptr,
                       grid_radaii_ptr,
                       BLOCK_RP: tl.constexpr,
                       BLOCK_C: tl.constexpr,
                     ):
    '''Same pore radius per random point as GPoreU_PSD_Kernel:
    pore_d(rp) = max{ r_g : n-accessible g, min-image |rp - g| <= r_g },
    computed over a compact candidate list with a 2D program grid
    (random-point chunk x candidate chunk); programs combine through
    tl.atomic_max on the (nonnegative) per-point output.  The comparison
    keeps the reference sqrt-then-compare form so boundary points resolve
    bit-identically.'''
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    rp_offs = pid_r * BLOCK_RP + tl.arange(0, BLOCK_RP)[:, None]
    rmask = rp_offs < N_RP

    rp0 = tl.load(rand_points_ptr + rp_offs * 3 + 0, mask=rmask, other=0.0)
    rp1 = tl.load(rand_points_ptr + rp_offs * 3 + 1, mask=rmask, other=0.0)
    rp2 = tl.load(rand_points_ptr + rp_offs * 3 + 2, mask=rmask, other=0.0)

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

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    cmask = offs_c < N_CAND
    gidx = tl.load(cand_ptr + offs_c, mask=cmask, other=0)

    b0 = tl.load(grid_ptr + gidx * 3 + 0, mask=cmask, other=0.0)
    b1 = tl.load(grid_ptr + gidx * 3 + 1, mask=cmask, other=0.0)
    b2 = tl.load(grid_ptr + gidx * 3 + 2, mask=cmask, other=0.0)
    ref_d = tl.load(grid_radaii_ptr + gidx, mask=cmask, other=0.0)

    dx = rp0 - b0
    dy = rp1 - b1
    dz = rp2 - b2

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
    d = tl.sqrt(d2)

    d_check = (d <= ref_d) & cmask
    d_masked = tl.where(d_check, ref_d, 0.0)
    best = tl.max(d_masked, axis=1, keep_dims=True)

    tl.atomic_max(pore_out_ptr + rp_offs, best, mask=rmask)

@triton.jit
def GPoreU_PSD_Kernel(
                       NGRID: tl.constexpr,
                       N_RP:  tl.constexpr, # number of random points
                       rand_points_ptr,
                       pore_out_ptr,
                       cell_ptr,
                       grid_ptr,
                       grid_radaii_ptr,
                       n_grid_occ_ptr,
                     ):

    '''------ 1 PID PER CYCLE BATCH ------'''
    pid = tl.program_id(0)
    if pid>=N_RP:
        return

    '''--- LOAD A  RAND POINT ---'''
    rp0 = tl.load(rand_points_ptr + pid * 3 + 0)
    rp1 = tl.load(rand_points_ptr + pid * 3 + 1)
    rp2 = tl.load(rand_points_ptr + pid * 3 + 2)

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
    
    offs_b = tl.arange( 0, 128 ) # WORK IN BLOCKS OF 128

    '''--- LOOP OVER ALL GRID POINTS IN THE CELL ---'''
    
    pore_d  = 0.0
    ZERO = 0.0
    LARGE = 10000000000
    for start in range(0, NGRID, 128):
        
            idx  = start + offs_b # MEMORY OFFSET FOR GRID LOADING
            block_mask = idx < NGRID    # MASK FOR OUT OF BOUNDS
    
            '''---- LOAD A BLOCK OF GRID COORDINATES ------'''
            b0 = tl.load( grid_ptr + idx * 3 + 0, mask=block_mask, other=0.0 )
            b1 = tl.load( grid_ptr + idx * 3 + 1, mask=block_mask, other=0.0 )
            b2 = tl.load( grid_ptr + idx * 3 + 2, mask=block_mask, other=0.0 )
    
            '''--- GRID BLOCK REFERENCE MIN D TO FRAME ----'''
            ref_d      = tl.load( grid_radaii_ptr + idx, mask=block_mask, other=0.0 )
            ref_access = tl.load( n_grid_occ_ptr + idx,    mask=block_mask, other=1.0 )

            '''----- BLOCK OF DISPLACEMENT VECTORS -----'''
            dx = rp0 - b0 
            dy = rp1 - b1 
            dz = rp2 - b2 

            '''---- BLOCK OF FRACTIONAL COORDINATES ----'''
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
            d  = tl.sqrt( d2 )                             
            d  = tl.where(block_mask, d, LARGE)

            d_check = ( d <= ref_d )  & ( ref_access == False ) # shape 128
            d_masked = tl.where(d_check, ref_d, ZERO)
            block_max=tl.max(d_masked)
            pore_d=tl.maximum(pore_d,block_max)
            
    tl.store( pore_out_ptr + pid, pore_d )
           
def GPoreU_psd_ffv(_input):
    ''' ------------- UNPACK DICT TO TENSORS ------------------ '''
    grid         =         _input['grid']
    cell         =         _input['cell']
    n_grid_occ   =   _input['n_grid_occ']
    grid_radaii  =  _input['grid_radaii']
    bin_size     =     _input['bin_size']
    cell_inv     = torch.linalg.inv(cell)
    cell_flat    = torch.cat([cell.reshape(-1),
                   cell_inv.reshape(-1)]).contiguous()
    volume       =       _input['volume']
    maxpore      =      _input['max_pore']
    ''' ------------ ASSERTS TO VERIFY ALL CUDA --------------- '''
    assert  grid.is_cuda
    assert  cell.is_cuda
    assert  grid_radaii.is_cuda
    assert n_grid_occ.is_cuda

    ''' ---------- PRECOMPUTE RANDOM VOLUME POINTS ----------- '''
    rand_points = (torch.rand(_input['psd_cycles'], 3).to('cuda') @ \
                   cell.unsqueeze(0)).view(-1,3)

    ''' ---------- INIT OUTPUT AND CONSTANT PARAMS ----------- '''
    NGRID     =  grid.shape[0]
    N_RP      =  int(_input['psd_cycles'].item())
    pore_out  =  torch.zeros(N_RP, device='cuda', dtype=torch.float32)

    ''' ---- compact candidate list: N-accessible points with radius>0 --- '''
    occ = n_grid_occ.view(-1)
    rad = grid_radaii.view(-1)
    cand = torch.nonzero((~occ) & (rad > 0.0)).view(-1).to(torch.int32)
    N_CAND = int(cand.numel())

    ''' ------------- CALL FUSED KERNEL (2D tiled, atomics) -- '''
    if N_CAND:
        BRP = 8
        BC = 256
        grid2d = (triton.cdiv(N_RP, BRP), triton.cdiv(N_CAND, BC))
        GPoreU_PSD_Kernel_2D[grid2d](
                                          N_CAND,
                                          N_RP,
                                          rand_points,
                                          cand,
                                          pore_out,
                                          cell_flat,
                                          grid,
                                          rad.contiguous(),
                                          BLOCK_RP=BRP,
                                          BLOCK_C=BC,
                                          num_warps=2
                                         )
    ''' ------------- CALL FUSED KERNEL --------------- ''' 
    nzpore_out = pore_out[pore_out > 0]
    diameter = 2 * nzpore_out
    nbins = int(math.ceil(maxpore / bin_size))

    bins = torch.arange( 0,(nbins + 1) * bin_size, bin_size, device='cuda')

# Histogram on CPU (CUDA backend does not support tensor-valued bins)
    hist = torch.histogram( diameter.cpu(), bins=bins.cpu()).hist.to(diameter.device)

    hist_flip = torch.flip(hist, dims=[0])
    cumul_hist = torch.cumsum(hist_flip, dim=0)
    cumul_psd = torch.flip(cumul_hist, dims=[0])

    norm_cumul_psd = cumul_psd / cumul_psd[0]

    #psd = -( norm_cumul_psd[1:] - norm_cumul_psd[:-1]) / bin_size
    psd = -(norm_cumul_psd[2:] - norm_cumul_psd[:-2]) / (2 * bin_size)
    ''' ------------- PACK OUTPUT TO DICT --------------- '''
    ffv = (pore_out > 0.0).float().mean()
    prof.vprint(ffv)
    prof.vprint("nonzero pore_out:", torch.count_nonzero(pore_out).item())
    prof.vprint("maximum pore_out:", pore_out.max().item())
    probe_occupiable_volume= volume*ffv
    _out = dict(
    ffv=ffv,
    pore_out=pore_out,
    norm_cumul_psd=norm_cumul_psd,
    psd=psd,
    probe_occupiable_volume=probe_occupiable_volume)
    return _out
