import numpy as np
from conftest import ts

from fluxopt import Carrier, Effect, Flow, Port, Storage, optimize


def test_stats_summary_quickstart():
    """`result.stats.summary` exposes objective, effect totals and capacity factor."""
    demand = Flow('elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
    source = Flow('elec', size=200, effects_per_flow_hour={'cost': 0.04})

    result = optimize(
        timesteps=ts(3),
        carriers=[Carrier('elec')],
        effects=[Effect('cost')],
        objective_effects='cost',
        ports=[Port('grid', imports=[source]), Port('demand', exports=[demand])],
    )

    summary = result.stats.summary

    # KPIs are present...
    assert 'objective' in summary
    assert 'effect_totals' in summary
    assert 'total_duration' in summary
    assert 'size' in summary
    assert 'total_flow_hours' in summary
    assert 'capacity_factor' in summary
    # No storages -> no storage KPIs.
    assert 'capacity' not in summary
    assert 'relative_mean_level' not in summary

    # ...and carry meaningful content, not just keys.
    assert np.isfinite(summary['objective'].item())
    assert 'cost' in summary['effect_totals'].coords['effect'].values
    assert np.isfinite(summary['effect_totals'].sel(effect='cost').item())

    # The per-flow KPIs share the `flow` dim, while effect totals live on
    # `effect` — it's a KPI namespace, not a flat table.
    assert 'effect' in summary['effect_totals'].dims
    for var in ('size', 'total_flow_hours', 'capacity_factor'):
        assert summary[var].dims == ('flow',)

    flow = 'grid(elec)'
    size = summary['size'].sel(flow=flow).item()
    tfh = summary['total_flow_hours'].sel(flow=flow).item()
    cf = summary['capacity_factor'].sel(flow=flow).item()
    duration = summary['total_duration'].item()

    # size and throughput are transparent; CF is the dimensionless quotient.
    assert size == 200
    assert np.isfinite(tfh) and tfh >= 0
    assert 0 <= cf <= 1
    assert np.isclose(cf, tfh / (size * duration))


def test_stats_summary_with_storage():
    """With storages, `summary` adds capacity and relative mean level."""
    source = Flow('elec', size=100, effects_per_flow_hour={'cost': [0.1, 0.9, 0.1, 0.9]})
    demand = Flow('elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5, 0.5])
    storage = Storage('batt', charging=Flow('elec', size=80), discharging=Flow('elec', size=80), capacity=80)

    result = optimize(
        timesteps=ts(4),
        carriers=[Carrier('elec')],
        effects=[Effect('cost')],
        objective_effects='cost',
        ports=[Port('grid', imports=[source]), Port('load', exports=[demand])],
        storages=[storage],
    )

    stats = result.stats
    summary = stats.summary

    # Storage KPIs appear, named after the storage `level` vocabulary.
    assert 'capacity' in summary
    assert 'relative_mean_level' in summary

    assert summary['capacity'].sel(storage='batt').item() == 80

    rml = stats.relative_mean_level.sel(storage='batt').item()
    # Mean fill is a horizon-independent fraction in [0, 1].
    assert 0 <= rml <= 1
