import torch
from . import util
from .grid import all_grids_pbc
from .percolate import GPoreU_percolate
from .psdchanged import GPoreU_psd_ffv
from .surface_area import GPoreU_SA
from .PLD_new import calculate_pld
from .energy import GPoreU_energy
from .hmax_kernel import GPoreU_poreGraph
#from watershed_gpore import GPoreU_watershed


def run_workload(fname, grid_spacing):

    data = util.prepare_input(fname, grid_spacing)

    framework_ff = torch.cat(
    (
        data['coord'],
        data['sigmas'][:, None],
        data['epsilons'][:, None]
    ),
    dim=1
)

    torch.save(framework_ff.cpu(), 'framework_ff.pt')

    print("UFF sigma/epsilon pairs used by GPore-U:")

    import numpy as np

    np.savetxt(
        'framework_ff.txt',
        framework_ff.cpu().numpy()
    )

    ff_pairs = torch.stack(
        (
            data['sigmas'],
            data['epsilons']
        ),
        dim=1
    )

    unique_pairs = torch.unique(
        ff_pairs,
        dim=0
    )

    for pair in unique_pairs:
        print(
            "sigma:",
            pair[0].item(),
            "epsilon K:",
            pair[1].item()
        )


    data.update(all_grids_pbc(data))
    data.update(GPoreU_percolate(data))
    data.update(GPoreU_psd_ffv(data))
    data.update(GPoreU_SA(data))
  # data.update(GPoreU_watershed(data))
    data = GPoreU_poreGraph(data)
   # data = GPoreU_poreGraph(data)
    print("Final pore count:", data['num_pores'])
    print("Final center IDs:", data['merged_maxima_ids'])
    print("Final center radii:", data['clusters_radaii'])
    print("Pore labels shape:", data['pore_labels'].shape)
    print("Pore labels and grid counts:", data['pore_labels'].unique(return_counts=True))
    print("Accepted periodic pairs:", data['periodic_pairs'])
    print("Number of H-max pore centers:", data['hmax_ids'].shape[0])
    print("H-max pore center IDs:", data['hmax_ids'])
    print("H-max pore radii:", data['hmax_radii'])
    print("H-max heights:", data['hmax_height'])


    ''' ----------- CALCULATE PLD ---------------------------- '''

    data.update(calculate_pld(data))


    # Energy and energy gradient without impulse
    data.update(GPoreU_energy(data))


    # Save no-impulse energy and gradient data
    energy_gradient_data = torch.stack(
        (
            data['energies'],
            data['gradients']
        ),
        dim=1
    )

    torch.save(
        energy_gradient_data.cpu(),
        'energy_gradient_data_no_impulse.pt'
    )

    print(
        "No-impulse energy-gradient data saved to "
        "energy_gradient_data_no_impulse.pt"
    )


    # Energy histogram sanity check
    accessible_count = (
        ~data['geo_grid_occ']
    ).sum().item()

    histogram_count = (
        data['energy_histogram']
        .sum()
        .item()
    )

    histogram_normalized_sum = (
        data['energy_histogram_normalized']
        .sum()
        .item()
    )


    print("-------------------------------------------")
    print("Accessible grid points:", accessible_count)
    print("Histogram total count:", histogram_count)
    print("Normalized histogram sum:", histogram_normalized_sum)
    print("-------------------------------------------")
   # print("Number of pore center voxels:", data['pore_centers'].sum().item())
    #print("Number of markers:", data['num_pores'])
    #print("Unique marker IDs:", torch.unique(data['markers']))

    ''' ----------- PRINT OUTPUT ----------------------------- '''

    for k, v in data.items():
        print('key: ', k)
        print('value: ', v)


    quit()

    return


if __name__ == '__main__':
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()

    fname = Path(args.fname).resolve()
    if not fname.is_file():
        parser.error(f'Input file not found: {fname}')

    root = Path(__file__).resolve().parent
    run_id = os.environ.get('SLURM_JOB_ID')
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    output = root / 'results' / fname.stem / run_id
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)

    print('Starting GPoreU')
    print('Input:', fname)
    print('Results:', output)
    grid_spacing= 0.4
    run_workload(str(fname), grid_spacing)
