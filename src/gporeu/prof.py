"""Tiny opt-in profiler; enable with GPOREU_PROFILE=1."""
import os
import sys
import time

import torch

_PROF = os.environ.get('GPOREU_PROFILE') == '1'
_VERBOSE = os.environ.get('GPOREU_VERBOSE') == '1'


def vprint(*args, **kw):
    # diagnostics (tensor reprs cost milliseconds each); enable with
    # GPOREU_VERBOSE=1
    if _VERBOSE:
        print(*args, **kw)
_last = [None]


def reset():
    if _PROF:
        _last[0] = None


def mark(name):
    if not _PROF:
        return
    torch.cuda.synchronize()
    now = time.perf_counter()
    if _last[0] is not None:
        print(f'  [prof] {name}: {(now - _last[0]) * 1e3:7.2f} ms', file=sys.stderr, flush=True)
    _last[0] = now
