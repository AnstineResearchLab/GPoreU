"""Tiny opt-in profiler; enable with GPOREU_PROFILE=1."""
import os
import sys
import time

import torch

_PROF = os.environ.get('GPOREU_PROFILE') == '1'
_VERBOSE = os.environ.get('GPOREU_VERBOSE') == '1'


def vprint(*args, **kw):
    """Print stage diagnostics; active only with GPOREU_VERBOSE=1.

    (Tensor reprs cost milliseconds each, so they are kept behind this
    gate by default.)
    """
    if _VERBOSE:
        print(*args, **kw)
_last = [None]


def reset():
    """Reset the profiler clock (call before a new stage sequence)."""
    if _PROF:
        _last[0] = None


def mark(name):
    """Record the interval since the previous mark (GPOREU_PROFILE=1)."""
    if not _PROF:
        return
    torch.cuda.synchronize()
    now = time.perf_counter()
    if _last[0] is not None:
        print(f'  [prof] {name}: {(now - _last[0]) * 1e3:7.2f} ms', file=sys.stderr, flush=True)
    _last[0] = now
