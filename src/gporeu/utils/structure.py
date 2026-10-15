"""
Structure input and pipeline preparation.

CIF reading (fast built-in P1 parser, with an Open Babel fallback for
symmetry-expanded files), analysis-grid generation, and packing of
force-field and probe constants into the dict consumed by every pipeline
stage.
"""

import math
import torch
from . import force_fields
from . import prof
import numpy as np
try:
    import pybel
except:
    try:
        from openbabel import pybel
    except ImportError as e:
        prof.vprint(f"ImportError: {e}")
        prof.vprint('Something appears to be wrong with the openbabel pybel installation. Aborting...')

from openbabel import openbabel as ob
ob.obErrorLog.StopLogging()


# Chemical symbols indexed by atomic number (table adapted from ASE).
chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']



def fill_uc(coord, numbers, cell,  charges, types, sg='P 1'):

    """Expand the asymmetric unit to the full unit cell using Open
        Babel symmetry handling; returns
        (coord, numbers, charges, types, cell).
    """

    coord = np.array(coord).reshape(-1, 3).tolist()
    numbers = np.array(numbers).reshape(-1).tolist()
    cell = np.array(cell).reshape(3, 3).tolist()

    mol = pybel.ob.OBMol()
    for c, n, ch, typ in zip(coord, numbers, charges, types):
        atom = pybel.ob.OBAtom()
        atom.SetVector(*c)
        atom.SetAtomicNum(n)
        atom.SetPartialCharge(ch)
        atom.SetType(typ)
        mol.AddAtom(atom)
    ucell = pybel.ob.OBUnitCell()
    ucell.SetData(pybel.ob.vector3(*cell[0]), pybel.ob.vector3(*cell[1]), pybel.ob.vector3(*cell[2]))
    ucell.SetSpaceGroup(sg)
    assert ucell.GetSpaceGroup().GetHallName() == sg
    ucell.FillUnitCell(mol)
    mol = pybel.Molecule(mol)
    coord = np.array([a.coords for a in mol.atoms])
    numbers = np.array([a.atomicnum for a in mol.atoms])
    charges = np.array([a.partialcharge for a in mol.atoms]).flatten()
    types = np.array([a.type for a in mol.atoms]).flatten()
    return coord, numbers, charges, types, np.array(cell)


_SYMMETRY_TAGS = ('_symmetry_equiv_pos',)

def _symbol_to_z(sym, label):
    import re as _re
    s = _re.sub(r'[^A-Za-z]', '', sym or '')
    if not s and label:
        m = _re.match(r'^[A-Za-z]+', label)
        s = m.group(0) if m else ''
    if not s:
        raise ValueError('no element symbol')
    key = s[0].upper() + s[1:].lower()
    if key in chemical_symbols:
        return chemical_symbols.index(key)
    key = s[0].upper()
    if key in chemical_symbols:
        return chemical_symbols.index(key)
    raise ValueError('unknown element ' + repr(sym))


def _fast_read_cif(fpath):
    """Fast parser for plain P1 CIFs (no symmetry operators).

    Replicates the pybel + fill_uc result to within 1 ulp (double
    precision).  Returns (coord, numbers, types, charges, cell).
    """
    txt = open(fpath).read().split('\n')
    par, rows, cols = {}, [], None
    i, n = 0, len(txt)
    while i < n:
        t = txt[i].strip()
        if t.startswith('_cell_'):
            p = t.split()
            try:
                par[p[0]] = float(p[-1])
            except ValueError:
                pass
            i += 1
            continue
        if t == 'loop_':
            headers = []
            j = i + 1
            while j < n and txt[j].strip().startswith('_'):
                headers.append(txt[j].split()[0].lower())
                j += 1
            if any(h.startswith('_atom_site') for h in headers):
                cols = {h: k for k, h in enumerate(headers)}
                while (j < n and txt[j].strip()
                       and txt[j].strip() not in ('loop_', '#')
                       and not txt[j].strip().startswith('_')):
                    rows.append(txt[j].split())
                    j += 1
            i = j
            continue
        i += 1

    if cols is None or not rows:
        raise ValueError('no atom_site loop')
    # Symmetry operator columns (loop-level symops or per-site symmetry_xyz)
    # are accepted ONLY if every operation is the identity; anything else
    # must go through Open Babel so expansion happens as before.
    def _is_identity(op):
        s = op.lower().replace(' ', '').replace('+', '')
        s = s.strip('\'"').rstrip('.;')
        return s == 'x,y,z'
    for h in list(cols):
        if 'symop_operation' in h or 'symmetry_xyz' in h:
            for r in rows:
                if not _is_identity(r[cols[h]]):
                    raise ValueError('non-identity symmetry operation')
        # walk every loop in the file, recording symop tables
        k = 0
        while k < len(txt):
            t = txt[k].strip()
            if t == 'loop_':
                headers = []
                j = k + 1
                while j < len(txt) and txt[j].strip().startswith('_'):
                    headers.append(txt[j].split()[0].lower())
                    j += 1
                symcol = [m for m, h in enumerate(headers)
                          if 'symop_operation' in h
                          or ('symmetry' in h and h.endswith('_xyz'))]
                if symcol:
                    single = len(headers) == 1
                    while j < len(txt) and txt[j].strip() \
                            and txt[j].strip() != 'loop_' \
                            and not txt[j].strip().startswith('_'):
                        cells = txt[j].split()
                        if single:
                            if not _is_identity(txt[j].strip()):
                                raise ValueError(
                                    'non-identity symmetry operation')
                        else:
                            for m in symcol:
                                if m < len(cells):
                                    c = cells[m]
                                    # a quoted op containing spaces would
                                    # have been split; be conservative and
                                    # defer that file to Open Babel
                                    if c[:1] in '\'"' and c[-1:] != c[:1]:
                                        raise ValueError(
                                            'unparsable symmetry operation')
                                    if not _is_identity(c):
                                        raise ValueError(
                                            'non-identity symmetry '
                                            'operation')
                        j += 1
                k = j
                continue
            k += 1

    a_ = par['_cell_length_a']
    b_ = par['_cell_length_b']
    c_ = par['_cell_length_c']
    al = math.radians(par.get('_cell_angle_alpha', 90.0))
    be = math.radians(par.get('_cell_angle_beta', 90.0))
    ga = math.radians(par.get('_cell_angle_gamma', 90.0))
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sg = math.sin(ga)
    cx = c_ * cb
    cy = c_ * (ca - cb * cg) / sg
    cz = math.sqrt(max(0.0, c_ * c_ - cx * cx - cy * cy))
    cell = np.array([[a_, 0.0, 0.0],
                     [b_ * cg, b_ * sg, 0.0],
                     [cx, cy, cz]], dtype=np.float64)

    fkeys = ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z')
    ckeys = ('_atom_site_cartn_x', '_atom_site_cartn_y', '_atom_site_cartn_z')
    usef = all(k in cols for k in fkeys)
    keys = fkeys if usef else ckeys
    if not all(k in cols for k in keys):
        raise ValueError('no coordinates')

    coord = np.empty((len(rows), 3), dtype=np.float64)
    numbers = np.empty(len(rows), dtype=np.int64)
    types, charges = [], []
    for r_i, r in enumerate(rows):
        p = np.array([float(r[cols[k]]) for k in keys], dtype=np.float64)
        coord[r_i] = p @ cell if usef else p
        sym = r[cols['_atom_site_type_symbol']] \
            if '_atom_site_type_symbol' in cols else ''
        lbl = r[cols['_atom_site_label']] \
            if '_atom_site_label' in cols else ''
        numbers[r_i] = _symbol_to_z(sym, lbl)
        types.append(sym or lbl)
        charges.append(float(r[cols['_atom_site_charge']])
                       if '_atom_site_charge' in cols else 0.0)

    return coord, numbers, types, charges, cell


def read_cif(fpath, _id=None, check_duplicates=True):
    """Read a CIF structure; returns (coord, numbers, types, charges, cell).

    Plain P1 CIFs are handled by a fast built-in parser that matches the
    Open Babel route exactly through float32 conversion (see
    :func:`_fast_read_cif`); files carrying non-identity symmetry
    operations -- or any parse hiccup -- fall through to Open Babel.
    """
    try:
        raw = open(fpath).read()
        if not any(tag in raw for tag in _SYMMETRY_TAGS):
            return _fast_read_cif(fpath)
    except Exception:
        pass

    # Initialize outputs.
    cell = []
    coord = []
    numbers = []
    types = []
    charges = []

    # Parse the unit-cell vectors.
    mol = next(pybel.readfile('cif', fpath))
    cell_matrix = mol.unitcell.GetCellMatrix()
    for i in range(3):
        cell.append([cell_matrix.Get(i, 0), cell_matrix.Get(i, 1), cell_matrix.Get(i, 2)])

    for atom in mol.atoms:
        coord.append(list(atom.coords))
        numbers.append(atom.atomicnum)
        try:
            types.append(atom.type)
        except:
            pass
        try:
            charges.append(atom.partialcharge)
        except:
            charges.append(0.0)

    # Apply symmetry expansion (Open Babel path).
    space_group = mol.unitcell.GetSpaceGroup().GetHallName()
    coord, numbers, charges, types, cell = fill_uc(coord,
                                                    numbers,
                                                    cell,
                                                    charges,
                                                    types,
                                                    space_group)
    return coord, numbers, types, charges, cell


def init_grid(cell, coord, spacing):

    """Create a nearly-uniform grid of the requested spacing over the
        cell; returns (Cartesian points, nx, ny, nz, trials).
    """


    # Number of grid points per dimension.
    # Step along the LATTICE vectors: the row lengths |a|, |b|, |c|,
    # NOT the Cartesian diagonal entries cell[i, i]: for tilted
    # cells the diagonals are projections, which would coarsen the
    # grid along the axis and drop planes.
    lengths = torch.linalg.norm(cell, dim=1)
    nx =  int(lengths[0] / spacing)
    ny =  int(lengths[1] / spacing)
    nz =  int(lengths[2] / spacing)

    # Uniform spacing in fractional coordinates.
    underfrac = (lengths[0] - nx*spacing) / lengths[0]
    _x = torch.linspace(0, 1-underfrac, nx+1)
    underfrac = (lengths[1] - ny*spacing) / lengths[1]
    _y = torch.linspace(0, 1-underfrac, ny+1)
    underfrac = (lengths[2] - nz*spacing) / lengths[2]
    _z = torch.linspace(0, 1-underfrac, nz+1)    
    x, y, z = torch.meshgrid(_x, _y, _z,   indexing='ij')

    # Build the full mesh.
    x = x.flatten().reshape(-1,1)
    y = y.flatten().reshape(-1,1)
    z = z.flatten().reshape(-1,1)
    grid = torch.concatenate([x,y,z], dim=1).to('cuda')
    grid = (grid @ cell.unsqueeze(0)).view(-1,3)
    trials=int(coord.shape[0]*512)
    return grid, nx+1, ny+1, nz+1,trials

def prepare_input(fname, spacing): 

    """Read a CIF and build the complete pipeline input dict: structure
        tensors, analysis grid, force-field parameters and probe constants.
    """


    # Read the CIF structure.
    coord, numbers, types, charges, cell = read_cif(fname)

    # Convert structure info to tensors.
    coord   = torch.tensor(coord   , device='cuda', dtype=torch.float32)
    numbers = torch.tensor(numbers , device='cuda', dtype=torch.long)
    charges = torch.tensor(charges , device='cuda', dtype=torch.float32)
    cell    = torch.tensor(cell    , device='cuda', dtype=torch.float32)

    # Create the analysis grid.
    grid, nx, ny, nz,trials = init_grid(cell, coord, spacing)
    psd_cycles = torch.tensor([10000], dtype=torch.long, device='cuda')
    bin_size = 0.25
    max_pore = torch.tensor([20], dtype=torch.float32, device='cuda')
    wall_thickness= torch.tensor([1], dtype=torch.float32, device='cuda')

    # Probe constants (He, N2, CH4).
    UFF     =  force_fields.UFF().to('cuda')
    MASS    =  force_fields.MASS().to('cuda')
    mass    =  MASS(numbers).sum()
    volume  =  torch.linalg.det(cell)
    density =  mass * 1.660539 / volume
  
    sigmas  , epsilons   = UFF(numbers)

    he_sigma             = torch.tensor([2.58]  , device='cuda')
    he_epsilon           = torch.tensor([10.22] , device='cuda')
    n_sigma              = torch.tensor([3.314]  , device='cuda')
    n_epsilon            = torch.tensor([10.22] , device='cuda')
    probe_sigma          = torch.tensor([3.73] , device='cuda')
    probe_epsilon        = torch.tensor([148.0] , device='cuda')
    
    epsilons   /= 1.987e-3 # TEMPERATURE UNITS

    for num in torch.unique(numbers):
       mask = numbers == num
       prof.vprint("atomic number:", num.item(), "sigma:", sigmas[mask][0].item(), "epsilon K:", epsilons[mask][0].item())
    # Pack results into the input dict.
    _input = dict(coord        =        coord, # FRAMEWORK
                  numbers      =      numbers, # FRAMEWORK
                  charges      =      charges, # FRAMEWORK
                  cell         =         cell, # FRAMEWORK
                  grid         =         grid, # GRID
                  nx           =           nx, # GRID
                  ny           =           ny, # GRID
                  nz           =           nz, # GRID
                  he_sigma     =     he_sigma, # PROBE
                  he_epsilon   =   he_epsilon, # PROBE
                  n_sigma      =      n_sigma, # PROBE
                  n_epsilon    =    n_epsilon, # PROBE
                  sigmas       =       sigmas, # FRAMEWORK
                  epsilons     =     epsilons, # FRAMEWORK
                  mass         =         mass, # FRAMEWORK
                  volume       =       volume, # FRAMEWORK
                  density      =      density, # FRAMEWORK
                  psd_cycles   =   psd_cycles, # DEFAULT SETTING
		  bin_size     =     bin_size, # DEFAULT SETTING
                  trials       =       trials, # DEFAULT SETTING
                  max_pore     =     max_pore, # DEFAULT SETTING
                  probe_sigma  =  probe_sigma, # DEFAULT SETTING
                  probe_epsilon=probe_epsilon, # DEFAULT SETTING
                  grid_spacing = spacing, # GRID
                  wall_thickness= wall_thickness )    
    return _input


def cif_2_xyz(f_in, f_out):

    """Convert a CIF file to XYZ via Open Babel (utility).
    """

    coord, numbers, types, charges, cell = read_cif(f_in)
    f = open(f_out, 'w')
    f.write(str(len(coord)) + '\n')
    f.write('this is a comment line\n')
    for i in range(len(coord)):
        f.write(chemical_symbols[numbers[i]] + ' ')
        f.write(str(coord[i][0]) + ' ')
        f.write(str(coord[i][1]) + ' ')
        f.write(str(coord[i][2]) + '\n')
    f.close()
    return
