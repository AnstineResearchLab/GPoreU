"""GPoreU command-line interface.

Run pore characterization on one CIF file or a whole directory of them.

    gporeu structure.cif
    gporeu ./cifs/ -o ./results --save-grids
    gporeu ./cifs/ -m pld,porosity -c settings.json
    gporeu --list-methods

The workflow is driven by a method registry: each user-facing METHOD is
backed by one or more pipeline STEPS, every step declares the steps it
depends on, and an automatically computed topological order guarantees the
values a method needs are produced before it runs.  Requesting e.g.
``energy_histogram`` transparently pulls in ``energy_grid`` and ``grids``.

Outputs per structure ``NAME``:
    NAME.summary   formatted characterization summary (always)
    NAME.grid      grid property dump (npz or pt, only with --save-grids)
"""

import argparse
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path


# ---------------------------------------------------------------------------
# Pipeline steps: name -> (required steps, runner)
#
# A runner takes the shared `data` dict (plus resolved config), executes one
# GPoreU stage, and updates the dict with its outputs.  The canonical order
# below is also the topological order of the dependency graph.
# ---------------------------------------------------------------------------

STEP_ORDER = [
    'grids',
    'connectivity',
    'psd',
    'surface_area',
    'network',
    'pld',
    'energy_grid',
    'energy_hist',
]

STEP_REQUIRES = {
    'grids':        [],
    'connectivity': ['grids'],
    'psd':          ['grids'],
    'surface_area': [],                         # only needs prepared input
    'network':      ['connectivity'],          # ij_geo + geo_access
    'pld':          ['connectivity'],          # consumes ij_geo adjacency
    'energy_grid':  ['grids'],                 # evaluated on grid points
    'energy_hist':  ['energy_grid'],           # 2D histogram of E, |grad E|
}


def _step_grids(data, cfg):
    from .grid import all_grids_pbc
    data.update(all_grids_pbc(data))


def _step_connectivity(data, cfg):
    from .percolate import GPoreU_percolate
    data.update(GPoreU_percolate(data))


def _step_psd(data, cfg):
    from .psd_ffv import GPoreU_psd_ffv
    data.update(GPoreU_psd_ffv(data))


def _step_surface_area(data, cfg):
    from .surface_area import GPoreU_SA
    data.update(GPoreU_SA(data))


def _step_network(data, cfg):
    from .pore_graph import GPoreU_poreGraph
    out = GPoreU_poreGraph(data, safe_buffer=cfg['pore_network']['safe_buffer'])
    if out is not None and out is not data:
        data.update(out)


def _step_pld(data, cfg):
    from .pld import calculate_pld
    data.update(calculate_pld(data))


def _step_energy_grid(data, cfg):
    from .energy import calculate_energy_gradient
    e = cfg['energy']
    energies, gradients = calculate_energy_gradient(
        data['grid'], data['coord'], data['sigmas'], data['epsilons'],
        data['cell'], float(data['probe_sigma'].item()),
        float(data['probe_epsilon'].item()),
        rcut=e['rcut'], delta=e['delta'], images=e['images'])
    data['energies'] = energies
    data['gradients'] = gradients


def _step_energy_hist(data, cfg):
    from .energy import build_2d_energy_histogram
    hist, hist_norm = build_2d_energy_histogram(data['energies'],
                                                data['gradients'])
    data['energy_histogram'] = hist
    data['energy_histogram_normalized'] = hist_norm


STEP_RUNNERS = {
    'grids':        _step_grids,
    'connectivity': _step_connectivity,
    'psd':          _step_psd,
    'surface_area': _step_surface_area,
    'network':      _step_network,
    'pld':          _step_pld,
    'energy_grid':  _step_energy_grid,
    'energy_hist':  _step_energy_hist,
}

STEP_LABELS = {
    'grids':        'lattice grid construction',
    'connectivity': 'percolation + accessible volumes',
    'psd':          'pore size distribution / FFV',
    'surface_area': 'Monte-Carlo surface area',
    'network':      'pore network (H-max + watershed)',
    'pld':          'PLD / LCD (winding union-find)',
    'energy_grid':  'CH4 energy + gradient grid',
    'energy_hist':  '2D energy histogram (E, |grad E|)',
}

# ---------------------------------------------------------------------------
# User-facing methods: name -> steps required, aliases, help text
# ---------------------------------------------------------------------------

METHODS = {
    'porosity': dict(
        steps=['connectivity'],
        help='helium + geometric accessible pore volumes (percolating) '
             'and gravimetric uptakes'),
    'percolation': dict(
        steps=['connectivity'],
        help='dimensionality / percolation flags for He, N2, geometric probes'),
    'ffv': dict(
        steps=['psd'],
        help='fractional free volume + probe-occupiable volume (N2 probe)'),
    'pore_size_distribution': dict(
        steps=['psd'], aliases=['psd'],
        help='pore size distribution from random probe sampling'),
    'surface_area': dict(
        steps=['surface_area'], aliases=['sa'],
        help='accessible surface area (MC SASA on 1.122*sigma_mix sphere)'),
    'pore_network': dict(
        steps=['network'], aliases=['poregraph', 'network'],
        help='pore network graph: H-maxima, watershed flooding, merged '
             'pore count and centers'),
    'pld': dict(
        steps=['pld'],
        help='pore-size limiting diameter per axis + connectivity dimension'),
    'lcd': dict(
        steps=['pld'],
        help='largest cavity diameter (computed by the PLD engine)'),
    'energy_grid': dict(
        steps=['energy_grid'], aliases=['energy'],
        help='CH4 interaction-energy and gradient grids (kJ/mol)'),
    'energy_histogram': dict(
        steps=['energy_hist'], aliases=['ehist', '2d_energy_histogram'],
        help='2D energy/|gradient| histogram (pulls in the CH4 energy grid)'),
}

DEFAULT_METHODS = ['porosity', 'percolation', 'ffv', 'pore_size_distribution',
                   'surface_area', 'pore_network', 'pld', 'energy_histogram']

METHOD_ALIASES = {}
for _m, _d in METHODS.items():
    for _a in _d.get('aliases', []):
        METHOD_ALIASES[_a] = _m


def resolve_method(name):

    """Map a user method name or alias to its canonical method name.
    """

    key = name.strip().lower()
    if key in METHODS:
        return key
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    raise KeyError(name)


def plan_steps(method_names):
    """Transitive closure of the requested methods' steps, in canonical
    (topological) order.  Returns (steps, why) where `why` maps every step
    to the chain of methods/steps that pulled it in."""
    needed, why = set(), {}

    def require(step, reason):
        if step in needed:
            return
        needed.add(step)
        why[step] = reason
        for dep in STEP_REQUIRES[step]:
            require(dep, f'{STEP_LABELS.get(step, step)} -> '
                         f'{STEP_LABELS.get(dep, dep)}')

    for m in method_names:
        for s in METHODS[m]['steps']:
            require(s, f'method "{m}"')
    return [s for s in STEP_ORDER if s in needed], why


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    'methods': DEFAULT_METHODS,
    'grid_spacing': 0.4,
    'seed': 0,
    'probes': {},                    # optional overrides, see example JSON
    'psd_cycles': 10000,
    'psd_bin_size': 0.25,
    'psd_max_pore': 20.0,
    'pore_network': {'wall_thickness': 1.0, 'safe_buffer': 6.0},
    'energy': {'rcut': 12.0, 'delta': 0.005, 'images': True},
    'output': {'save_grids': False, 'grid_format': 'npz',
               'compress': False,
               'summary_suffix': '.summary', 'grid_suffix': '.grid'},
}


def deep_merge(base, override):

    """Recursively merge ``override`` onto ``base`` into a new dict.
    """

    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):

    """Load a JSON settings file merged onto DEFAULT_CONFIG; keys
        starting with '_' (documentation comments) are ignored.
    """

    with open(path) as fh:
        cfg = json.load(fh)
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    return deep_merge(DEFAULT_CONFIG, cfg)


PROBE_KEYS = {                     # config name -> data dict key
    'helium_sigma': 'he_sigma', 'helium_epsilon': 'he_epsilon',
    'nitrogen_sigma': 'n_sigma', 'nitrogen_epsilon': 'n_epsilon',
    'ch4_sigma': 'probe_sigma', 'ch4_epsilon': 'probe_epsilon',
}


# ---------------------------------------------------------------------------
# Per-structure run
# ---------------------------------------------------------------------------

def run_one(cif_path, cfg, verbose=False):

    """Run the auto-resolved pipeline on one CIF file.

        Returns (data, steps, why, timings): the populated pipeline dict,
        the executed step order, the per-step dependency explanation, and
        per-step wall times in seconds.
    """

    import torch
    from .utils import structure as util
    torch.manual_seed(cfg['seed'])

    t0 = time.perf_counter()
    data = util.prepare_input(str(cif_path), cfg['grid_spacing'])

    # apply configurable knobs
    data['psd_cycles'] = torch.tensor([int(cfg['psd_cycles'])],
                                      dtype=torch.long, device='cuda')
    data['bin_size'] = float(cfg['psd_bin_size'])
    data['max_pore'] = torch.tensor([float(cfg['psd_max_pore'])],
                                    device='cuda')
    data['wall_thickness'] = torch.tensor(
        [float(cfg['pore_network']['wall_thickness'])], device='cuda')
    for cname, dkey in PROBE_KEYS.items():
        if cname in cfg['probes']:
            data[dkey] = torch.tensor([float(cfg['probes'][cname])],
                                      device='cuda')
    timings = {'prepare': (time.perf_counter() - t0) * 1e3}

    steps, why = plan_steps(cfg['methods'])
    for s in steps:
        torch.cuda.synchronize()
        t = time.perf_counter()
        if verbose:
            STEP_RUNNERS[s](data, cfg)
        else:
            with redirect_stdout(io.StringIO()):
                STEP_RUNNERS[s](data, cfg)
        torch.cuda.synchronize()
        timings[s] = (time.perf_counter() - t) * 1e3

    return data, steps, why, timings


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def _cell_params(cell):
    import numpy as np
    c = cell.detach().cpu().numpy()
    a, b, cc = (np.linalg.norm(c[i]) for i in range(3))
    def ang(u, v):
        return math.degrees(math.acos(float(np.dot(u, v)
                                 / (np.linalg.norm(u) * np.linalg.norm(v)))))
    return a, b, cc, ang(c[1], c[2]), ang(c[0], c[2]), ang(c[0], c[1])


def _fmt_flag(t):
    return 'yes' if t else 'no '


def format_summary(data, cif_path, cfg, requested, steps, timings):

    """Format the human-readable characterization report for one
        structure (also the content of NAME.summary).
    """

    L = []
    W = 68
    vol = float(data['volume'].item())
    mass = float(data['mass'].item())          # Da per cell
    dens = float(data['density'].item())
    a, b, c, al, be, ga = _cell_params(data['cell'])
    grid_pts = int(data['grid'].shape[0])
    m3g = mass * 1.660539                      # Da -> g scale factor

    def grav(v_a3):
        return v_a3 / m3g                      # cm^3/g

    L.append('=' * W)
    L.append(f' GPoreU characterization summary - {cif_path.name}')
    L.append('=' * W)
    L.append(f'  atoms            : {int(data["coord"].shape[0])}')
    L.append(f'  cell             : a={a:.3f}  b={b:.3f}  c={c:.3f}  '
             f'alpha={al:.2f} beta={be:.2f} gamma={ga:.2f}')
    L.append(f'  cell volume      : {vol:.1f} A^3     '
             f'density: {dens:.3f} g/cm^3')
    L.append(f'  grid             : {int(data["nx"])} x {int(data["ny"])} x '
             f'{int(data["nz"])} @ {cfg["grid_spacing"]:.3f} A '
             f'({grid_pts} points)')
    L.append('-' * W)

    req = set(requested)

    if 'porosity' in req:
        vh = float(data['access_he_pore_volume'])
        vg = float(data['access_geo_pore_volume'])
        L.append(f'  pore volume (He, probe {float(data["he_sigma"]):.2f} A)'
                 f' : {vh:10.2f} A^3   {grav(vh):6.3f} cm^3/g')
        L.append(f'  pore volume (geometric)              '
                 f' : {vg:10.2f} A^3   {grav(vg):6.3f} cm^3/g')

    if 'percolation' in req:
        def flags(p):
            return (f'[{_fmt_flag(data[p + "_x"])},'
                    f'{_fmt_flag(data[p + "_y"])},'
                    f'{_fmt_flag(data[p + "_z"])}]  (x,y,z)')
        L.append(f'  percolates He        : {flags("he_perc")}')
        L.append(f'  percolates N2        : {flags("n_perc")}')
        L.append(f'  percolates geometric : {flags("geo_perc")}')

    if 'ffv' in req:
        ffv = float(data['ffv'].item()) if hasattr(data['ffv'], 'item') \
            else float(data['ffv'])
        pov = data['probe_occupiable_volume']
        pov = float(pov.item()) if hasattr(pov, 'item') else float(pov)
        L.append(f'  FFV (N2 probe)       : {ffv:10.4f}     occupiable '
                 f'{pov:.1f} A^3 ({grav(pov):.3f} cm^3/g)')

    if 'pore_size_distribution' in req:
        pmax = float(data['pore_out'].max().item())
        L.append(f'  PSD: max sampled pore diameter {pmax:.2f} A '
                 f'(bins {cfg["psd_bin_size"]} A to {cfg["psd_max_pore"]} A, '
                 f'{cfg["psd_cycles"]} cycles)')

    if 'surface_area' in req:
        sa = float(data['total_surface_area'].item())
        L.append(f'  surface area       : {sa:10.1f} A^2/cell   '
                 f'{sa / m3g * 1e4:8.2f} m^2/g')

    if 'pore_network' in req:
        npores = int(data['num_pores'])
        rmax = (float(data['maxima_radii'].max().item())
                if 'maxima_radii' in data and data['maxima_radii'].numel()
                else float('nan'))
        L.append(f'  pore network       : {npores} pore(s); largest pore '
                 f'radius ~{rmax:.2f} A')
        L.append(f'     floods {int(data["flood_iterations"])} sweeps; '
                 f'{int(data["periodic_pairs"].shape[0])} periodic '
                 f'center merges')

    if 'pld' in req or 'lcd' in req:
        if 'pld' in data:
            L.append(f'  PLD                : {float(data["pld"]):10.3f} A '
                     f'(dimension D{int(data["pld_dimension"])})')
            L.append(f'     per axis: X {float(data["pld_x"]):.3f}   '
                     f'Y {float(data["pld_y"]):.3f}   '
                     f'Z {float(data["pld_z"]):.3f} A')
        if 'lcd' in data:
            L.append(f'  LCD                : {float(data["lcd"]):10.3f} A')

    if 'energy_grid' in req and 'energies' in data:
        acc = ~data['geo_grid_occ']
        e = data['energies'][acc]
        if e.numel():
            L.append(f'  CH4 energy grid    : accessible min '
                     f'{float(e.min()):.2f} / mean {float(e.mean()):.2f} / '
                     f'max {float(e.max()):.2f} kJ/mol '
                     f'(rcut {cfg["energy"]["rcut"]} A)')

    if 'energy_histogram' in req and 'energy_histogram' in data:
        h = data['energy_histogram']
        hn = data['energy_histogram_normalized']
        L.append(f'  CH4 2D histogram   : {int(h.sum().item())} grid points '
                 f'-> {tuple(h.shape)} (E x |grad E|) bins; '
                 f'normalized sum {float(hn.sum().item()):.4f}')

    L.append('-' * W)
    order = ' -> '.join(steps)
    L.append(f'  pipeline (auto-resolved): {order}')
    tline = '  ' + '  '.join(f'{k}={v:.1f}ms' for k, v in timings.items()
                             if v >= 0.05)
    L.append(f' timings:{tline[1:]}')
    L.append('=' * W)
    return '\n'.join(L)


# ---------------------------------------------------------------------------
# Grid property dump
# ---------------------------------------------------------------------------

GRID_DUMP_KEYS = {               # file key -> data key
    'points': 'grid',
    'occ_geo': 'geo_grid_occ',
    'occ_helium': 'he_grid_occ',
    'occ_nitrogen': 'n_grid_occ',
    'free_radius': 'grid_radaii',
    'lj_helium_K': 'he_lj_energy',
    'lj_nitrogen_K': 'n_lj_energy',
    'access_geo': 'geo_access',
    'access_helium': 'he_access',
    'access_nitrogen': 'n_access',
    'pore_out': 'pore_out',
    'pore_labels': 'pore_labels',
    'safe_mask': 'safe_mask',
    'ch4_energy_kJ_mol': 'energies',
    'ch4_gradient_kJ_mol_A': 'gradients',
}


def save_grid(data, grid_path, cfg):

    """Write the grid property arrays of one structure to FILE
        (numpy npz format, or torch pt with grid_format='pt').
    """

    import torch
    fmt = cfg['output']['grid_format']
    payload = {k: data[v].detach().cpu()
               for k, v in GRID_DUMP_KEYS.items() if v in data}
    payload['grid_shape'] = torch.tensor(
        [int(data['nx']), int(data['ny']), int(data['nz'])])
    if fmt == 'pt':
        torch.save(dict(payload), grid_path)
        return
    import numpy as np
    savef = np.savez_compressed if cfg['output']['compress'] else np.savez
    arrays = {k: (v.numpy() if hasattr(v, 'numpy') else v)
              for k, v in payload.items()}
    # open handle: np.savez would otherwise append '.npz' to the path
    with open(grid_path, 'wb') as fh:
        savef(fh, **arrays)


# ---------------------------------------------------------------------------
# Input expansion / main
# ---------------------------------------------------------------------------

def expand_inputs(paths):

    """Expand CLI inputs (CIF files and/or directories) into a sorted
        list of existing CIF paths, skipping broken symlinks.
    """

    cif_files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found = sorted(p.glob('*.cif')) + sorted(p.glob('*.[Cc][Ii][Ff]'))
            uniq = []
            for f in found:
                # skip duplicates, broken symlinks, non-regular entries
                if f in uniq or not f.is_file():
                    continue
                uniq.append(f)
            if not uniq:
                print(f'warning: no .cif files found in {p}',
                      file=sys.stderr)
            cif_files.extend(uniq)
        elif p.is_file():
            cif_files.append(p)
        else:
            raise FileNotFoundError(f'input not found: {p}')
    # stable, de-duplicated order
    out, seen = [], set()
    for f in cif_files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            out.append(f)
    return out


def list_methods():

    """Print the method table with aliases and resolved step
        dependencies.
    """

    print('GPoreU methods (name = steps required):')
    for m in METHODS:
        steps, _ = plan_steps([m])
        alias = ', '.join(METHODS[m].get('aliases', []))
        print(f'  {m:<24}{"[alias: " + alias + "]" if alias else "":<26}'
              f'{METHODS[m]["help"]}')
        if len(steps) > len(METHODS[m]['steps']):
            extra = [s for s in steps if s not in METHODS[m]['steps']]
            print(f'  {"":<24}{"":<26}'
                  f'(auto-requests: {", ".join(extra)})')
    print('\ndefault set: ' + ', '.join(DEFAULT_METHODS))
    print('\nsteps and dependencies:')
    for s in STEP_ORDER:
        deps = ', '.join(STEP_REQUIRES[s]) or '-'
        print(f'  {s:<14} {STEP_LABELS[s]:<38} requires: {deps}')


def main(argv=None):

    """Command-line entry point; returns the process exit code
        (0 success, 1 any structure failed, 2 no CUDA GPU).
    """

    ap = argparse.ArgumentParser(
        prog='gporeu',
        description='GPoreU pore characterization for one CIF file or a '
                    'directory of CIF files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='examples:\n'
               '  gporeu IRMOF1.cif\n'
               '  gporeu ./cifs/ -o out/ --save-grids\n'
               '  gporeu ./cifs/ -m pld,porosity -c settings.json\n'
               '  gporeu --list-methods\n')
    ap.add_argument('inputs', nargs='*', help='CIF file(s) and/or director'
                    'ies containing CIF files')
    ap.add_argument('-o', '--outdir', default=None,
                    help='output directory (default: ./gporeu_results)')
    ap.add_argument('-c', '--config', default=None,
                    help='JSON settings file (see settings.example.json)')
    ap.add_argument('-m', '--methods', default=None,
                    help='comma-separated method list (overrides config)')
    ap.add_argument('-g', '--grid-spacing', type=float, default=None,
                    help='grid spacing in A (overrides config; default 0.4)')
    ap.add_argument('-s', '--seed', type=int, default=None,
                    help='RNG seed (overrides config; default 0)')
    ap.add_argument('--save-grids', action='store_true',
                    help='also write NAME.grid with grid property arrays')
    ap.add_argument('--grid-format', choices=['npz', 'pt'], default=None,
                    help='grid dump format (default npz)')
    ap.add_argument('--explain', action='store_true',
                    help='print the dependency-resolution plan and exit')
    ap.add_argument('--list-methods', action='store_true',
                    help='list selectable methods and their dependencies')
    ap.add_argument('--init-config', metavar='FILE', default=None,
                    help='write the documented example settings JSON to '
                         'FILE and exit')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='show stage diagnostics (prints, profiler off)')
    args = ap.parse_args(argv)

    if args.list_methods:
        list_methods()
        return 0
    if args.init_config:
        import shutil
        tpl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'settings.example.json')
        shutil.copyfile(tpl, args.init_config)
        print(f'example settings written to {args.init_config}')
        return 0
    if not args.inputs:
        ap.error('no input given (a CIF file or a directory of them)')

    cfg = load_config(args.config) if args.config else dict(DEFAULT_CONFIG)
    if args.methods:
        try:
            cfg['methods'] = [resolve_method(m) for m in
                              args.methods.split(',')]
        except KeyError as e:
            ap.error(f'unknown method: {e} (see --list-methods)')
    else:
        cfg['methods'] = [resolve_method(m) for m in cfg['methods']]
    if args.grid_spacing is not None:
        cfg['grid_spacing'] = args.grid_spacing
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.save_grids:
        cfg['output']['save_grids'] = True
    if args.grid_format:
        cfg['output']['grid_format'] = args.grid_format

    files = expand_inputs(args.inputs)
    if not files:
        ap.error('nothing to run on')

    if args.explain:
        steps, why = plan_steps(cfg['methods'])
        print('requested methods : ' + ', '.join(cfg['methods']))
        print('execution plan    : ' + ' -> '.join(steps))
        for s in steps:
            print(f'   {s:<14} <- {why[s]}')
        return 0

    import torch
    if not torch.cuda.is_available():
        print('error: GPoreU requires a CUDA GPU', file=sys.stderr)
        return 2

    outdir = Path(args.outdir) if args.outdir else Path('gporeu_results')
    outdir.mkdir(parents=True, exist_ok=True)

    failures = 0
    done_files = []
    for i, cif in enumerate(files, 1):
        tag = f'[{i}/{len(files)}] {cif.name}'
        data = None
        try:
            data, steps, why, timings = run_one(cif, cfg,
                                                verbose=args.verbose)
            text = format_summary(data, cif, cfg, cfg['methods'], steps,
                                  timings)
            print(f'--- {tag} ' + '-' * max(0, 60 - len(tag)))
            print(text, flush=True)
            summary_path = outdir / (cif.stem + cfg['output']['summary_suffix'])
            summary_path.write_text(text + '\n')
            done_files.append(summary_path)
            if cfg['output']['save_grids']:
                grid_path = outdir / (cif.stem + cfg['output']['grid_suffix'])
                save_grid(data, grid_path, cfg)
                print(f'  grid properties ->  {grid_path}')
                done_files.append(grid_path)
        except Exception as exc:                    # batch resilience
            failures += 1
            print(f'{tag}  FAILED: {type(exc).__name__}: {exc}',
                  file=sys.stderr)
            import traceback
            traceback.print_exc()
            err_path = outdir / (cif.stem + cfg['output']['summary_suffix'])
            err_path.write_text(f'{cif.name}: ERROR {type(exc).__name__} '
                                f'{exc}\n')
        finally:
            data = None
            torch.cuda.empty_cache()

    if len(files) > 1:
        print(f'\nprocessed {len(files) - failures}/{len(files)} structures '
              f'-> {outdir.resolve()}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
