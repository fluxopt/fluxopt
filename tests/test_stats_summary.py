import numpy as np
from conftest import ts

from fluxopt import Carrier, Effect, Flow, Port, optimize


def test_stats_summary_quickstart():
    """`result.stats.summary` exposes objective, effect totals and full-load hours."""
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
    assert 'size' in summary
    assert 'total_flow_hours' in summary
    assert 'full_load_hours' in summary

    # ...and carry meaningful content, not just keys.
    assert np.isfinite(summary['objective'].item())
    assert 'cost' in summary['effect_totals'].coords['effect'].values
    assert np.isfinite(summary['effect_totals'].sel(effect='cost').item())

    # The per-flow KPIs share the `flow` dim, while effect totals live on
    # `effect` — it's a KPI namespace, not a flat table.
    assert 'effect' in summary['effect_totals'].dims
    for var in ('size', 'total_flow_hours', 'full_load_hours'):
        assert summary[var].dims == ('flow',)

    flow = 'grid(elec)'
    size = summary['size'].sel(flow=flow).item()
    tfh = summary['total_flow_hours'].sel(flow=flow).item()
    flh = summary['full_load_hours'].sel(flow=flow).item()

    # size and throughput are transparent, and FLH is exactly their quotient.
    assert size == 200
    assert np.isfinite(tfh) and tfh >= 0
    assert np.isfinite(flh) and flh >= 0
    assert np.isclose(flh, tfh / size)
