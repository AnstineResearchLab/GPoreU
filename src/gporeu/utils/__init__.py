"""
Shared utilities for the GPoreU pipeline stages.

``structure``  CIF reading, grid preparation and input packing
``force_fields``  UFF lookup tables
``union_find``  Triton graph primitives (components, maximum spanning
                forest) used by more than one stage
``prof``  opt-in verbosity and profiling diagnostics
"""
