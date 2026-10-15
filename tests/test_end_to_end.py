"""End-to-end CLI test on a real CIF (requires CUDA + test CIFs).

    GPOREU_TEST_CIF_DIR=/path/to/cifs pytest tests/test_end_to_end.py

Uses WEI.cif (small, fully characterized reference structure):
PLD 1.852 A / dimension D2, LCD 2.574 A, N2-impermeable, all at 0.4 A
grid spacing.  When WEI.cif is absent, the numeric assertions are skipped
and only the pipeline contract is checked.
"""
import os
from pathlib import Path

import pytest

CIF_DIR = os.environ.get('GPOREU_TEST_CIF_DIR')
CIF = Path(CIF_DIR) / 'WEI.cif' if CIF_DIR else None

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

if not (CIF and CIF.is_file()):
    pytest.skip('set GPOREU_TEST_CIF_DIR to a directory containing '
                'WEI.cif', allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip('CUDA GPU required', allow_module_level=True)

from gporeu.cli import (DEFAULT_CONFIG, format_summary, main, run_one)  # noqa: E402


def test_wei_reference_values_and_determinism():
    data, steps, why, timings = run_one(CIF, DEFAULT_CONFIG)
    assert steps[0] == 'grids'
    assert float(data['pld']) == pytest.approx(1.852, abs=0.05)
    assert int(data['pld_dimension']) == 2
    assert float(data['lcd']) == pytest.approx(2.574, abs=0.12)
    assert not (data['n_perc_x'] or data['n_perc_y'] or data['n_perc_z'])
    assert float(data['ffv'].item()) == 0.0           # N2-impermeable

    text = format_summary(data, CIF, DEFAULT_CONFIG,
                          DEFAULT_CONFIG['methods'], steps, timings)
    assert 'PLD' in text and 'summary - WEI.cif' in text

    # seeded determinism: same seed, same surface area
    again, *_ = run_one(CIF, DEFAULT_CONFIG)
    assert (again['total_surface_area']
            == data['total_surface_area'])


def test_cli_writes_summary_and_grid(tmp_path):
    out = tmp_path / 'out'
    rc = main([str(CIF), '-o', str(out), '-m', 'porosity,pld',
               '--save-grids'])
    assert rc == 0
    assert (out / 'WEI.summary').is_file()
    assert (out / 'WEI.grid').is_file()
    summary = (out / 'WEI.summary').read_text()
    assert 'pore volume' in summary
    assert 'pipeline (auto-resolved): grids -> connectivity -> pld' \
        in summary

    import numpy as np
    with open(out / 'WEI.grid', 'rb') as fh:
        arrays = np.load(fh)
        for key in ('points', 'occ_geo', 'free_radius'):
            assert key in arrays.files
