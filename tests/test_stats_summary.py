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
    assert 'full_load_hours' in summary

    # ...and carry meaningful content, not just keys.
    assert np.isfinite(summary['objective'].item())
    assert 'cost' in summary['effect_totals'].coords['effect'].values
    assert np.isfinite(summary['effect_totals'].sel(effect='cost').item())

    flh = summary['full_load_hours']
    assert 'grid(elec)' in flh.coords['flow'].values
    source_flh = flh.sel(flow='grid(elec)').item()
    assert np.isfinite(source_flh)
    assert source_flh >= 0
