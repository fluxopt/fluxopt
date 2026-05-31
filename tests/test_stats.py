import numpy as np
import xarray as xr

from fluxopt import FlowSystem, Flow, Effect

def test_stats_summary_quickstart():
    time = xr.DataArray(np.arange(2), dims=['time'], coords={'time': np.arange(2)})
    sys = FlowSystem(time=time, dt=1.0)
    sys.build(flows=[Flow(id='f1', size=10)], effects=[Effect(id='cost')])
    res = sys.solve()

    summary = res.stats.summary()

    assert 'objective' in summary
    assert 'effect_totals' in summary
    assert 'full_load_hours' in summary
    assert 'f1' in summary['full_load_hours'].coords['flow'].values
