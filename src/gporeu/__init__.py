"""GPoreU -- GPU pore characterization for porous crystals (MOFs).

Computes from a CIF file: accessible pore volumes (helium / geometric),
percolation dimensionality, fractional free volume and pore size
distribution, accessible surface area, pore network (H-maxima + watershed
flooding), pore limiting diameter (PLD) and largest cavity diameter (LCD)
via a winding-vector union-find, and CH4 interaction-energy / gradient
grids with a 2D (E, |grad E|) histogram.

Command line:  ``gporeu`` (see ``gporeu -h`` / ``gporeu --list-methods``)
or ``python -m gporeu``.
"""

__version__ = '1.0.0'
__all__ = ['__version__']
