"""The program is bound on the labels the user wrote, not on positions.

A caller who reads or edits a table sees the timestamp and the year; a result
comes back on the same labels without being mapped back.
"""

from __future__ import annotations

import mathspec as ms
import polars as pl
import pytest
import specsolve
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
    params = _system(ts(3), periods=[2020, 2030], period_weights=[10, 10]).sources()
    assert params['dt']['time'].to_list() == ts(3), 'time is keyed by the timestamps the system was given'
    rates = params['effects_per_flow_hour']
    assert sorted(set(rates['period'].to_list())) == [2020, 2030], 'period is keyed by the years'


@pytest.mark.parametrize('steps', [pytest.param([10, 20, 30], id='ints'), pytest.param([1.5, 2.5], id='floats')])
def test_numbered_steps_are_refused_rather_than_read_as_1970(steps: list) -> None:
    """pydantic reads a number as seconds since 1970, which would bind 10 s steps as 0.003 h."""
    with pytest.raises(TypeError, match=r'must be timestamps.*pl\.datetime_range'):
        _system(steps)


def test_a_dumped_system_loads_its_timestamps_again() -> None:
    """ISO strings are how timestamps come back from a dump, so they are not refused as numbers are."""
    system = _system(ts(3))
    assert FlowSystem.from_dict(system.to_dict()).sources()['dt']['time'].to_list() == ts(3)


def test_a_supplied_table_is_keyed_by_the_same_timestamps() -> None:
    """What a caller adds is joined on labels, so a position would match nothing."""
    system = _system(ts(3))
    patch = """
parameters:
  cap: {dims: [time]}
constraints:
  capped:
    dims: [flow, time, period]
    where: carrier_sign > 0
    expression: rate <= cap
"""
    spec = ms.override(system.spec(), [patch])
    cap = pl.DataFrame({'time': ts(3), 'value': [5.0, 5.0, 5.0]})
    result = specsolve.solve(spec, system.sources() | {'cap': cap})
    assert result.objective == pytest.approx(12.0), 'a cap of 5 does not bind on a demand of at most 3'
