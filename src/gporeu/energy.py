import math

import torch
import triton
import triton.language as tl


@triton.jit
def energy_gradient_kernel(
    grid_ptr,
    coord_ptr,
    sigma_ptr,
    epsilon_ptr,
    cell_ptr,
    inv_cell_ptr,
    energy_ptr,
    gradient_ptr,
    PROBE_SIGMA: tl.constexpr,
    PROBE_EPSILON: tl.constexpr,
    RCUT: tl.constexpr,
    DEL: tl.constexpr,
    NATOMS: tl.constexpr,
    NGRID: tl.constexpr,
    KXIMG: tl.constexpr,
    KYIMG: tl.constexpr,
    KZIMG: tl.constexpr,
    BLOCK_GRID: tl.constexpr,
    BLOCK_ATOMS: tl.constexpr
):

    # Load grid points
    grid_off = tl.program_id(0) * BLOCK_GRID + tl.arange(0, BLOCK_GRID)
    grid_mask = grid_off < NGRID

    gx = tl.load(grid_ptr + grid_off * 3 + 0, mask=grid_mask, other=0.0)
    gy = tl.load(grid_ptr + grid_off * 3 + 1, mask=grid_mask, other=0.0)
    gz = tl.load(grid_ptr + grid_off * 3 + 2, mask=grid_mask, other=0.0)

    gx = gx[:, None]
    gy = gy[:, None]
    gz = gz[:, None]


    # Load cell matrix
    c00 = tl.load(cell_ptr + 0)
    c01 = tl.load(cell_ptr + 1)
    c02 = tl.load(cell_ptr + 2)

    c10 = tl.load(cell_ptr + 3)
    c11 = tl.load(cell_ptr + 4)
    c12 = tl.load(cell_ptr + 5)

    c20 = tl.load(cell_ptr + 6)
    c21 = tl.load(cell_ptr + 7)
    c22 = tl.load(cell_ptr + 8)


    # Load inverse cell matrix
    i00 = tl.load(inv_cell_ptr + 0)
    i01 = tl.load(inv_cell_ptr + 1)
    i02 = tl.load(inv_cell_ptr + 2)

    i10 = tl.load(inv_cell_ptr + 3)
    i11 = tl.load(inv_cell_ptr + 4)
    i12 = tl.load(inv_cell_ptr + 5)

    i20 = tl.load(inv_cell_ptr + 6)
    i21 = tl.load(inv_cell_ptr + 7)
    i22 = tl.load(inv_cell_ptr + 8)


    # Running totals for each grid point
    total_energy = tl.zeros((BLOCK_GRID,), dtype=tl.float32)

    grad_x = tl.zeros((BLOCK_GRID,), dtype=tl.float32)
    grad_y = tl.zeros((BLOCK_GRID,), dtype=tl.float32)
    grad_z = tl.zeros((BLOCK_GRID,), dtype=tl.float32)


    # Loop through atoms in blocks
    for atom_start in range(0, NATOMS, BLOCK_ATOMS):

        atom_off = atom_start + tl.arange(0, BLOCK_ATOMS)
        atom_mask = atom_off < NATOMS


        # Load atom coordinates
        ax = tl.load(coord_ptr + atom_off * 3 + 0, mask=atom_mask, other=0.0)
        ay = tl.load(coord_ptr + atom_off * 3 + 1, mask=atom_mask, other=0.0)
        az = tl.load(coord_ptr + atom_off * 3 + 2, mask=atom_mask, other=0.0)

        ax = ax[None, :]
        ay = ay[None, :]
        az = az[None, :]


        # Atom-grid displacement
        dx = ax - gx
        dy = ay - gy
        dz = az - gz


        # Cartesian to fractional
        fx = dx * i00 + dy * i10 + dz * i20
        fy = dx * i01 + dy * i11 + dz * i21
        fz = dx * i02 + dy * i12 + dz * i22


        # Minimum image wrapping
        fx = fx - tl.floor(fx + 0.5)
        fy = fy - tl.floor(fy + 0.5)
        fz = fz - tl.floor(fz + 0.5)


        # Fractional back to Cartesian
        bx = fx * c00 + fy * c10 + fz * c20
        by = fx * c01 + fy * c11 + fz * c21
        bz = fx * c02 + fy * c12 + fz * c22


        # Loop over every periodic image of each atom within the cutoff
        # sphere.  With KXIMG = KYIMG = KZIMG = 0 this is exactly the plain
        # minimum-image sum; the minimum-image displacement above is the
        # k = (0, 0, 0) member of the enumerated image set.
        for kx in range(-KXIMG, KXIMG + 1):
            for ky in range(-KYIMG, KYIMG + 1):
                for kz in range(-KZIMG, KZIMG + 1):

                    # Cartesian lattice translation of this image
                    shx = kx * c00 + ky * c10 + kz * c20
                    shy = kx * c01 + ky * c11 + kz * c21
                    shz = kx * c02 + ky * c12 + kz * c22

                    dx = bx + shx
                    dy = by + shy
                    dz = bz + shz

                    # Distance squared
                    rsq = dx * dx + dy * dy + dz * dz


                    # Broadcast atom mask
                    valid_atom = atom_mask[None, :]


                    # Invalid atoms get huge distance
                    rsq = tl.where(valid_atom, rsq, 1.0e20)


                    # Prevent r = 0
                    rsq = tl.where(
                        valid_atom & (rsq < 1.0e-2),
                        1.0e-2,
                        rsq
                    )


                    # Distance
                    r = tl.sqrt(rsq)


                    # Only pairs within cutoff region are relevant
                    active_pair = valid_atom & (r <= RCUT + DEL)


                    # Out-of-cutoff pairs get huge distance
                    rsq = tl.where(
                        active_pair,
                        rsq,
                        1.0e20
                    )

                    r = tl.sqrt(rsq)


                    # Load atom LJ parameters
                    atom_sigma = tl.load(
                        sigma_ptr + atom_off,
                        mask=atom_mask,
                        other=0.0
                    )

                    atom_epsilon = tl.load(
                        epsilon_ptr + atom_off,
                        mask=atom_mask,
                        other=0.0
                    )

                    atom_sigma = atom_sigma[None, :]
                    atom_epsilon = atom_epsilon[None, :]


                    # Lorentz-Berthelot mixing
                    sigma = 0.5 * (PROBE_SIGMA + atom_sigma)
                    epsilon = tl.sqrt(PROBE_EPSILON * atom_epsilon)


                    # Cutoff regions
                    normal_region = active_pair & (r < RCUT - DEL)

                    inner_cutoff_region = (
                        active_pair
                        & (r >= RCUT - DEL)
                        & (r <= RCUT)
                    )

                    outer_cutoff_region = (
                        active_pair
                        & (r > RCUT)
                        & (r <= RCUT + DEL)
                    )


                    # LJ terms
                    sigma2 = sigma * sigma

                    sr2 = sigma2 / rsq
                    sr6 = sr2 * sr2 * sr2
                    sr12 = sr6 * sr6


                    # LJ energy
                    lj_energy = 4.0 * epsilon * (sr12 - sr6)


                    # Energy exists up to RCUT
                    energy_region = normal_region | inner_cutoff_region

                    energy_contribution = tl.where(
                        energy_region,
                        lj_energy,
                        0.0
                    )


                    # LJ derivative factor
                    grad_factor = (
                        24.0
                        * epsilon
                        * (2.0 * sr12 - sr6)
                        / rsq
                    )


                    # Standard LJ force exists up to RCUT
                    gradient_region = normal_region | inner_cutoff_region

                    grad_factor = tl.where(
                        gradient_region,
                        grad_factor,
                        0.0
                    )


                    # LJ energy at RCUT
                    rcut2 = RCUT * RCUT

                    cutoff_sr2 = sigma2 / rcut2
                    cutoff_sr6 = cutoff_sr2 * cutoff_sr2 * cutoff_sr2
                    cutoff_sr12 = cutoff_sr6 * cutoff_sr6

                    uc = 4.0 * epsilon * (
                        cutoff_sr12 - cutoff_sr6
                    )


                    # Impulse correction
                    impulse = uc / (2.0 * DEL)

                    impulse_region = (
                        inner_cutoff_region
                        | outer_cutoff_region
                    )

                    impulse_factor = tl.where(
                        impulse_region,
                        impulse / r,
                        0.0
                    )


                    # Total force factor
                    total_grad_factor = grad_factor + impulse_factor


                    # x, y, z contributions
                    grad_x_contribution = total_grad_factor * dx
                    grad_y_contribution = total_grad_factor * dy
                    grad_z_contribution = total_grad_factor * dz


                    # Sum all atoms for each grid point
                    total_energy += tl.sum(
                        energy_contribution,
                        axis=1
                    )

                    grad_x += tl.sum(
                        grad_x_contribution,
                        axis=1
                    )

                    grad_y += tl.sum(
                        grad_y_contribution,
                        axis=1
                    )

                    grad_z += tl.sum(
                        grad_z_contribution,
                        axis=1
                    )


    # Norm of total gradient at each grid point
    gradient_norm = tl.sqrt(
        grad_x * grad_x
        + grad_y * grad_y
        + grad_z * grad_z
    )


    # Store outputs
    tl.store(
        energy_ptr + grid_off,
        total_energy,
        mask=grid_mask
    )

    tl.store(
        gradient_ptr + grid_off,
        gradient_norm,
        mask=grid_mask
    )



@triton.jit
def histogram_2d_kernel(
    energy_ptr,
    gradient_ptr,
    histogram_ptr,
    NGRID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):

    offsets = (
        tl.program_id(0) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    mask = offsets < NGRID


    # Load energy and gradient
    energy = tl.load(
        energy_ptr + offsets,
        mask=mask,
        other=0.0
    )

    gradient = tl.load(
        gradient_ptr + offsets,
        mask=mask,
        other=0.0
    )


    # Energy bins
    energy_bin = tl.floor(
        (energy + 24.0) / 2.0
    ).to(tl.int32)

    energy_bin = energy_bin + 1

    energy_bin = tl.where(
        energy < -24.0,
        0,
        energy_bin
    )

    energy_bin = tl.where(
        energy >= 0.0,
        13,
        energy_bin
    )


    # Gradient bins
    gradient_bin = tl.floor(
        gradient / 28.0
    ).to(tl.int32)

    gradient_bin = tl.where(
        gradient >= 112.0,
        4,
        gradient_bin
    )


    # Flatten 5 x 14 histogram
    histogram_index = (
        gradient_bin * 14
        + energy_bin
    )


    # Add grid point to histogram
    tl.atomic_add(
        histogram_ptr + histogram_index,
        1,
        mask=mask
    )



def image_counts(cell, rcut, delta=0.005):
    """Per-axis periodic-image ranges for an exact rcut spherical sum.

    Every lattice translation n with |n| <= rcut + delta satisfies
    |n_i| <= (rcut + delta) / w_i where w_i = V / |a_j x a_k| is the
    perpendicular width of the cell along axis i, so enumerating
    k_i in [-n_i, n_i] covers all images inside the cutoff sphere.
    Returns (0, 0, 0) when the cell is at least 2 * (rcut + delta) wide
    in every direction, reducing the kernel to the minimum-image sum.
    """
    c = cell.detach().cpu().double()
    vol = abs(torch.linalg.det(c)).item()
    ks = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        nrm = torch.linalg.cross(c[j], c[k]).norm().item()
        w = vol / nrm if nrm > 1e-12 else 1.0
        ks.append(max(0, int(math.floor((rcut + delta) / w))))
    return tuple(ks)


def calculate_energy_gradient(
    grid,
    coord,
    sigmas,
    epsilons,
    cell,
    probe_sigma,
    probe_epsilon,
    rcut=12.0,
    delta=0.005,
    images=True
):

    inv_cell = torch.linalg.inv(cell).contiguous()

    # Periodic images needed for an exact truncated lattice sum at this
    # cutoff; (0, 0, 0) for cells wider than 2 * (rcut + delta).
    if images:
        kx_img, ky_img, kz_img = image_counts(cell, rcut, delta)
    else:
        kx_img = ky_img = kz_img = 0

    ngrid = grid.shape[0]
    natoms = coord.shape[0]


    # Kernel outputs are initially in K and K/A
    energies = torch.empty(
        ngrid,
        device='cuda',
        dtype=torch.float32
    )

    gradients = torch.empty(
        ngrid,
        device='cuda',
        dtype=torch.float32
    )


    BLOCK_GRID = 128
    BLOCK_ATOMS = 32


    launch_grid = (
        triton.cdiv(ngrid, BLOCK_GRID),
    )


    energy_gradient_kernel[launch_grid](
        grid,
        coord,
        sigmas,
        epsilons,
        cell,
        inv_cell,
        energies,
        gradients,
        PROBE_SIGMA=probe_sigma,
        PROBE_EPSILON=probe_epsilon,
        RCUT=rcut,
        DEL=delta,
        NATOMS=natoms,
        NGRID=ngrid,
        KXIMG=kx_img,
        KYIMG=ky_img,
        KZIMG=kz_img,
        BLOCK_GRID=BLOCK_GRID,
        BLOCK_ATOMS=BLOCK_ATOMS
    )


    # Convert K to kJ/mol
    K_TO_KJMOL = 0.008314462618

    energies = energies * K_TO_KJMOL
    gradients = gradients * K_TO_KJMOL


    return energies, gradients



def build_2d_energy_histogram(
    energies,
    gradients
):

    ngrid = energies.numel()


    # 14 energy bins x 5 gradient bins
    histogram = torch.zeros(
        70,
        device='cuda',
        dtype=torch.int32
    )


    BLOCK_SIZE = 256


    launch_grid = (
        triton.cdiv(ngrid, BLOCK_SIZE),
    )


    histogram_2d_kernel[launch_grid](
        energies,
        gradients,
        histogram,
        NGRID=ngrid,
        BLOCK_SIZE=BLOCK_SIZE
    )


    # Rows = gradient bins
    # Columns = energy bins
    histogram = histogram.reshape(
        5,
        14
    )


    # Normalize by total number of grid points
    histogram_normalized = (
        histogram.to(torch.float32)
        / ngrid
    )


    return histogram, histogram_normalized



def GPoreU_energy(data, rcut=12.0, delta=0.005, images=True, histogram=True):

    # Use all grid points
    grid = data['grid']


    # Use existing framework data
    coord = data['coord']
    sigmas = data['sigmas']
    epsilons = data['epsilons']
    cell = data['cell']


    # Methane probe
    probe_sigma = float(data['probe_sigma'].item())
    probe_epsilon = float(data['probe_epsilon'].item())


    # Calculate energy and gradient for all grid points
    energies, gradients = calculate_energy_gradient(
        grid,
        coord,
        sigmas,
        epsilons,
        cell,
        probe_sigma,
        probe_epsilon,
        rcut,
        delta,
        images
    )


    out = {
        'energies': energies,
        'gradients': gradients
    }


    if histogram:
        # Build 2D histogram
        hist, hist_norm = build_2d_energy_histogram(
            energies,
            gradients
        )
        out['energy_histogram'] = hist
        out['energy_histogram_normalized'] = hist_norm


    return out
