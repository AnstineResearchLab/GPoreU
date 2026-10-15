"""
Pore size distribution (PSD) and fractional free volume (FFV).

A fixed number of random probe points is drawn uniformly in the unit cell.
For each probe point, the pore radius is the largest grid radius ``r_g``
over all free (N2-accessible) grid points ``g`` whose free sphere covers
the probe under the minimum-image convention:

    pore_d(rp) = max{ r_g : n-accessible g, min-image |rp - g| <= r_g }

The search runs as a single fused 2D-tiled Triton kernel over a compacted
candidate list; FFV is the fraction of probe points with pore radius > 0,
and the PSD follows from the finite-differentiated complementary
cumulative distribution of the nonzero pore diameters.
"""

import math

import torch
import triton
import triton.language as tl

from . import prof


@triton.jit
def GPoreU_PSD_Kernel(
    N_CAND,
    N_RP,
    rand_points_ptr,
    cand_ptr,
    pore_out_ptr,
    cell_ptr,
    grid_ptr,
    grid_radaii_ptr,
    BLOCK_RP: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """2D program grid: (random-point chunk) x (candidate chunk).

    ``cand_ptr`` holds the compacted indices of candidate grid points
    (free with radius > 0).  Programs combine through ``tl.atomic_max`` on
    the (nonnegative) per-point output.  The comparison keeps the
    sqrt-then-compare form so boundary points resolve bit-identically.
    """
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

    # Minimum-image displacement rp - g under the (possibly tilted) cell.
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


def GPoreU_psd_ffv(_input):
    """Compute PSD and FFV; returns a dict of ``ffv``, ``pore_out``,
    ``norm_cumul_psd``, ``psd`` and ``probe_occupiable_volume``.

    Requires ``grid``, ``cell``, ``n_grid_occ``, ``grid_radaii``,
    ``bin_size``, ``max_pore``, ``psd_cycles`` and ``volume`` from the
    grid stage.
    """
    grid = _input['grid']
    cell = _input['cell']
    n_grid_occ = _input['n_grid_occ']
    grid_radaii = _input['grid_radaii']
    bin_size = _input['bin_size']
    volume = _input['volume']
    maxpore = _input['max_pore']

    cell_inv = torch.linalg.inv(cell)
    cell_flat = torch.cat([cell.reshape(-1),
                           cell_inv.reshape(-1)]).contiguous()

    for t in (grid, cell, grid_radaii, n_grid_occ):
        assert t.is_cuda

    # Probe points drawn uniformly in the cell (unit fractionals @ cell).
    rand_points = (torch.rand(_input['psd_cycles'], 3).to('cuda')
                   @ cell.unsqueeze(0)).view(-1, 3)

    N_RP = int(_input['psd_cycles'].item())
    pore_out = torch.zeros(N_RP, device='cuda', dtype=torch.float32)

    # Compact candidate list: free grid points with positive radius.
    occ = n_grid_occ.view(-1)
    rad = grid_radaii.view(-1)
    cand = torch.nonzero((~occ) & (rad > 0.0)).view(-1).to(torch.int32)
    N_CAND = int(cand.numel())

    if N_CAND:
        BRP = 8
        BC = 256
        grid2d = (triton.cdiv(N_RP, BRP), triton.cdiv(N_CAND, BC))
        GPoreU_PSD_Kernel[grid2d](
            N_CAND, N_RP, rand_points, cand, pore_out, cell_flat, grid,
            rad.contiguous(), BLOCK_RP=BRP, BLOCK_C=BC, num_warps=2)

    prof.vprint("nonzero pore_out:", torch.count_nonzero(pore_out).item())
    prof.vprint("maximum pore_out:", pore_out.max().item())

    diameter = 2 * pore_out[pore_out > 0]
    nbins = int(math.ceil(maxpore / bin_size))
    bins = torch.arange(0, (nbins + 1) * bin_size, bin_size, device='cuda')

    # torch.histogram on CUDA does not support tensor-valued bins.
    hist = torch.histogram(diameter.cpu(), bins=bins.cpu()).hist.to(
        diameter.device)

    # Complementary cumulative distribution, normalized to 1 at bin 0.
    hist_flip = torch.flip(hist, dims=[0])
    cumul_hist = torch.cumsum(hist_flip, dim=0)
    cumul_psd = torch.flip(cumul_hist, dims=[0])
    norm_cumul_psd = cumul_psd / cumul_psd[0]

    # Central finite difference of the complementary CDF.
    psd = -(norm_cumul_psd[2:] - norm_cumul_psd[:-2]) / (2 * bin_size)

    ffv = (pore_out > 0.0).float().mean()
    return dict(
        ffv=ffv,
        pore_out=pore_out,
        norm_cumul_psd=norm_cumul_psd,
        psd=psd,
        probe_occupiable_volume=volume * ffv,
    )
