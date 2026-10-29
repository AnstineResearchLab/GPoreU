"""
Accessible surface area by Monte-Carlo SASA.

``trials`` (=512 per atom) uniform samples are drawn on the sphere of
radius ``1.122 * sigma_mix`` around each framework atom, where
``sigma_mix = 0.5 * (sigma_atom + sigma_N2)``; 1.122 is the probe-radius
factor turning the LJ contact distance into a SASA surface for the N2
probe.  A sample point is *blocked* when its minimum-image distance to
any other framework atom j falls below ``1.122 * 0.5 * (sigma_j +
sigma_N2)``.  The accessible area of an atom is its full spherical area
times the fraction of unblocked samples; the total is their sum over all
atoms (A^2 per unit cell).
"""

import torch
import triton
import triton.language as tl

from .utils import prof

SAMPLES_PER_ATOM = 512


# do_not_specialize: per-structure sizes must not fragment the Triton
# JIT cache (value-specialized ints/constexpr each compile a fresh binary).
@triton.jit(do_not_specialize=["NATOMS"])
def GPoreU_SA_Kernel(
    coord_ptr,
    sigmas_ptr,
    rand1_ptr,
    rand2_ptr,
    cell_ptr,
    n_sigma_ptr,
    accessible_count_ptr,
    surface_radius_ptr,
    NATOMS,
    SAMP: tl.constexpr,
    BLOCK_ATOMS: tl.constexpr,
):
    """2D program grid: (atom, sample chunk).

    Each program tests a block of SAMP samples of one atom against a tile
    of BLOCK_ATOMS framework atoms (spill-free [SAMP, BLOCK_ATOMS]
    tile).  Per-atom accessible counts accumulate through integer-valued
    atomic adds, so accessible_count is exact.  The host atom itself
    never blocks its own samples (idx != pid).
    """
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

    # Uniform sphere samples (u, v) -> spherical angles for this chunk.
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

    # Cell matrix and inverse (scalars, loaded once).
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
        # Skip remaining blocks once every sample is already blocked.
        if tl.sum((blocked == 0).to(tl.int32), axis=0) > 0:
            idx = start + offs_b
            mask = idx < NATOMS

            b0 = tl.load(coord_ptr + idx * 3 + 0, mask=mask, other=0.0)
            b1 = tl.load(coord_ptr + idx * 3 + 1, mask=mask, other=0.0)
            b2 = tl.load(coord_ptr + idx * 3 + 2, mask=mask, other=0.0)
            sigmas = tl.load(sigmas_ptr + idx, mask=mask, other=0.0)

            # Minimum-image displacement sample -> atom j.
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


def GPoreU_SA(_input):
    """Compute the accessible surface area (A^2 per cell); returns
    ``{'total_surface_area': tensor}``.

    Requires ``coord``, ``cell``, ``sigmas``, ``n_sigma`` and ``trials``
    from the prepared input.
    """
    coord = _input['coord']
    cell = _input['cell']
    sigmas = _input['sigmas']
    n_sigma = _input['n_sigma']
    trials = _input['trials']

    for t in (coord, cell, sigmas, n_sigma):
        assert t.is_cuda

    NATOMS = coord.shape[0]
    cell_inv = torch.linalg.inv(cell)
    cell_flat = torch.cat([cell.reshape(-1),
                           cell_inv.reshape(-1)]).contiguous()

    accessible_count = torch.zeros(NATOMS, device='cuda',
                                   dtype=torch.float32)
    surface_radius = torch.empty(NATOMS, device='cuda', dtype=torch.float32)

    rand1 = torch.rand(trials, device='cuda')
    rand2 = torch.rand(trials, device='cuda')

    SAMP = 64
    grid = (NATOMS, SAMPLES_PER_ATOM // SAMP)
    GPoreU_SA_Kernel[grid](
        coord, sigmas, rand1, rand2, cell_flat, n_sigma,
        accessible_count, surface_radius,
        NATOMS=NATOMS, SAMP=SAMP, BLOCK_ATOMS=128, num_warps=4)

    sfrac = accessible_count / SAMPLES_PER_ATOM
    surface_area_per_atom = 4.0 * torch.pi * surface_radius ** 2
    full_surface_area = torch.sum(surface_area_per_atom)
    accessible_surface_area_per_atom = surface_area_per_atom * sfrac
    total_surface_area = torch.sum(accessible_surface_area_per_atom)

    prof.vprint("-------------- SURFACE AREA ---------------")
    prof.vprint("NATOMS:", NATOMS)
    prof.vprint("accessible fraction: mean %.4f min %.4f max %.4f"
                % (sfrac.mean().item(), sfrac.min().item(),
                   sfrac.max().item()))
    prof.vprint("full surface area:       ", full_surface_area.item())
    prof.vprint("accessible surface area: ", total_surface_area.item())
    prof.vprint("overall accessible ratio:",
                (total_surface_area / full_surface_area).item())
    prof.vprint("--------------------------------------------")

    return dict(total_surface_area=total_surface_area)
