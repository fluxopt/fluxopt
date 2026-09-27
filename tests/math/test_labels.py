"""The program is bound on the labels the user wrote, not on positions.

A caller who reads or edits a table sees the timestamp and the year; a result
comes back on the same labels without being mapped back.
"""

from __future__ import annotations

import polars as pl
import pytest
from conftest import ts

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port


def _system(timesteps: list, **kwargs: object) -> FlowSystem:
    return FlowSystem(
        timesteps=timesteps,
        carriers=[Carrier(id='e')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(id='grid', imports=[Flow(carrier='e', size=10, effects_per_flow_hour={'cost': 2})]),
            Port(id='demand', exports=[Flow(carrier='e', size=1, fixed_relative_profile=[1, 2, 3])]),
        ],
        **kwargs,
    )


def test_a_table_carries_the_timestamps_and_the_years() -> None:
    params = _system(ts(3), periods=[2020, 2030], period_weights=[10, 10]).parameters()
    assert params['dt']['time'].to_list() == ts(3), 'time is keyed by the timestamps the system was given'
    rates = params['effects_per_flow_hour']
    assert sorted(set(rates['period'].to_list())) == [2020, 2030], 'period is keyed by the years'


def test_numbered_steps_keep_their_numbers() -> None:
    """Steps numbered from 10 would read as positions 0, 1, 2 if the labels were lost."""
    system = _system([10, 20, 30])
    assert system.math().dimensions['time'].dtype == 'int'
    assert system.parameters()['dt']['time'].to_list() == [10, 20, 30]
    result = system.optimize()
    assert result.objective == pytest.approx(12.0), 'demand of 1 + 2 + 3 at a price of 2'
    assert result.flow_rate('grid(e)').coords['time'].values.tolist() == [10, 20, 30]


def test_a_supplied_table_is_keyed_by_the_same_timestamps() -> None:
    """What a caller adds is joined on labels, so a position would match nothing."""
    system = _system(ts(3))
    math = system.math()
    math.parameters['cap'] = type(math.parameters['carrier_sign'])(dims=['time'])
    math.constraints['capped'] = type(math.constraints['carrier_balance'])(
        dims=['flow', 'time', 'period'], where='carrier_sign > 0', expression='rate <= cap'
    )
    cap = pl.DataFrame({'time': ts(3), 'value': [5.0, 5.0, 5.0]})
    result = system.optimize(math=math, parameters={'cap': cap})
    assert result.objective == pytest.approx(12.0), 'a cap of 5 does not bind on a demand of at most 3'
