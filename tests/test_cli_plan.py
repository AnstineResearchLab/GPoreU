"""Method/dependency-resolution logic tests (no GPU required)."""
import pytest

from gporeu.cli import (DEFAULT_CONFIG, METHODS, STEP_ORDER, STEP_REQUIRES,
                        deep_merge, load_config, plan_steps, resolve_method)


def test_all_methods_resolve():
    for m in METHODS:
        assert resolve_method(m) == m
    for m, spec in METHODS.items():
        for alias in spec.get('aliases', []):
            assert resolve_method(alias) == m


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        resolve_method('does-not-exist')


def test_plan_is_topological():
    steps, _ = plan_steps(list(METHODS))
    pos = {s: i for i, s in enumerate(steps)}
    for s in steps:
        for dep in STEP_REQUIRES[s]:
            assert pos[dep] < pos[s], f'{dep} must precede {s}'


def test_energy_histogram_pulls_energy_grid_and_grids():
    steps, _ = plan_steps(['energy_histogram'])
    assert steps == ['grids', 'energy_grid', 'energy_hist']


def test_pld_pulls_connectivity_for_adjacency():
    steps, _ = plan_steps(['pld'])
    assert steps == ['grids', 'connectivity', 'pld']


def test_subset_skips_unrelated_steps():
    steps, _ = plan_steps(['pore_size_distribution', 'pld'])
    assert 'network' not in steps
    assert 'energy_grid' not in steps
    assert 'energy_hist' not in steps
    assert steps == ['grids', 'connectivity', 'psd', 'pld']


def test_default_plan_covers_everything():
    # The default method set excludes opt-in steps (histogram variants);
    # the default plan must equal the canonical STEP_ORDER restricted to
    # steps the defaults actually need, in canonical order.
    steps, _ = plan_steps(DEFAULT_CONFIG['methods'])
    needed = set()
    for m in DEFAULT_CONFIG['methods']:
        for s in METHODS[m]['steps']:
            frontier = [s]
            while frontier:
                cur = frontier.pop()
                if cur not in needed:
                    needed.add(cur)
                    frontier += STEP_REQUIRES[cur]
    expected = [s for s in STEP_ORDER if s in needed]
    assert steps == expected


def test_every_step_reachable_from_some_method():
    reachable = set()
    for m, d in METHODS.items():
        frontier = list(d['steps'])
        while frontier:
            cur = frontier.pop()
            if cur not in reachable:
                reachable.add(cur)
                frontier += STEP_REQUIRES[cur]
    assert reachable == set(STEP_ORDER)


def test_why_explains_chains():
    from gporeu.cli import STEP_LABELS
    steps, why = plan_steps(['energy_histogram'])
    assert why['energy_hist'].startswith('method')
    assert STEP_LABELS['energy_hist'] in why['energy_grid']
    assert STEP_LABELS['energy_grid'] in why['grids']


def test_deep_merge_nested_override():
    cfg = deep_merge(DEFAULT_CONFIG,
                     {'energy': {'rcut': 9.0}, 'grid_spacing': 0.5})
    assert cfg['energy']['rcut'] == 9.0
    assert cfg['energy']['images'] is True          # untouched key kept
    assert cfg['grid_spacing'] == 0.5


def test_load_config_ignores_comment_keys(tmp_path):
    p = tmp_path / 's.json'
    p.write_text('{"_note": "hi", "grid_spacing": 0.6}')
    cfg = load_config(p)
    assert '_note' not in cfg
    assert cfg['grid_spacing'] == 0.6
    assert cfg['methods'] == DEFAULT_CONFIG['methods']
