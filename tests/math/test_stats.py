from __future__ import annotations

import pytest
from conftest import ts

from fluxopt import Carrier, Effect, Flow, Port, optimize

_elec = [Carrier(id='elec')]


class TestFlowHours:
    def test_flow_hours_nonnegative(self):

        result = optimize(
            timesteps=ts(3),
            carriers=_elec,
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='grid', imports=[Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        assert (result.stats.flow_hours >= 0).all()

    def test_total_flow_hours_matches_manual(self):

        demand = [50.0, 80.0, 60.0]
        result = optimize(
            timesteps=ts(3),
            carriers=_elec,
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='grid', imports=[Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        grid_total = float(result.stats.total_flow_hours.sel(flow='grid(elec)').values)
        assert grid_total == pytest.approx(sum(demand), abs=1e-6)


class TestCaching:
    def test_stats_accessor_cached(self):

        result = optimize(
            timesteps=ts(3),
            carriers=_elec,
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        assert result.stats is result.stats
        assert result.stats.flow_hours is result.stats.flow_hours
