"""What the program assumes of the numbers bound to it.

A range the element layer refuses is stated again as an `assumptions:` entry,
so specsolve checks it on whatever reaches the program: a resolved
`ProfileRef`, a reloaded file, or a table the caller edited.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
import specsolve as lpspec
from conftest import ts

from fluxopt import Carrier, Effect, Flow, FlowSystem, Investment, Port, Sizing, Status, Storage
from fluxopt.math import build_sources, objective_weights, program


def _system() -> FlowSystem:
    """One of every entity an assumption is about."""
    return FlowSystem(
        timesteps=ts(3),
        carriers=[Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(
                id='grid',
                imports=[
                    Flow(
                        carrier='heat',
                        size=Sizing(size_min=5, size_max=50),
                        relative_rate_min=0.1,
                        effects_per_flow_hour={'cost': 1},
                    )
                ],
            ),
            Port(
                id='plant',
                imports=[
                    Flow(
                        carrier='heat',
                        size=Investment(size_min=0, size_max=50, prior_size=10),
                        effects_per_flow_hour={'cost': 2},
                    )
                ],
            ),
            Port(
                id='boiler',
                imports=[
                    Flow(
                        carrier='heat',
                        size=20,
                        relative_rate_min=0.2,
                        status=Status(uptime_min=2),
                        effects_per_flow_hour={'cost': 3},
                    )
                ],
            ),
            Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=[10, 20, 15])]),
        ],
        storages=[
            Storage(
                id='tank',
                charging=Flow(carrier='heat', size=10),
                discharging=Flow(carrier='heat', size=10),
                capacity=Sizing(size_min=1, size_max=100),
                eta_charge=0.9,
                eta_discharge=0.9,
                relative_loss_per_hour=0.01,
            )
        ],
        periods=[2020],
        period_weights=[1],
    )


def _bound() -> dict[str, object]:
    data = _system().build_data()
    sources, coords = build_sources(data, objective_weights(data, 'cost'))
    return {**sources, **coords}


def _edit(table: object, value: float, side: str | None = None) -> pl.DataFrame:
    """The table with every value, or every value on one storage side, replaced."""
    frame = pl.from_pandas(table) if isinstance(table, pd.DataFrame) else table
    assert isinstance(frame, pl.DataFrame)
    hit = pl.lit(True) if side is None else pl.col('side') == side
    return frame.with_columns(pl.when(hit).then(pl.lit(value)).otherwise(pl.col('value')).alias('value'))


def test_the_shipped_numbers_hold_every_assumption() -> None:
    lpspec.build(program().expand('sos'), _bound()).close()


@pytest.mark.parametrize(
    ('parameter', 'value', 'side', 'assumption'),
    [
        pytest.param('relative_rate_min', -0.1, None, 'rate_floor_is_not_negative', id='a negative floor'),
        pytest.param('relative_rate_min', 2.0, None, 'rate_bounds_do_not_cross', id='a floor above the ceiling'),
        pytest.param('size_min', 60.0, None, 'size_bounds_are_ordered', id='a size minimum above its maximum'),
        pytest.param('prior_capacity', -1.0, None, 'prior_capacity_is_not_negative', id='a negative prior'),
        pytest.param('duration_min', -1.0, None, 'durations_are_ordered', id='a negative minimum stay'),
        pytest.param('level_max', -1.0, None, 'capacity_is_not_negative', id='a negative capacity'),
        pytest.param('capacity_min', 200.0, None, 'capacity_bounds_are_ordered', id='a capacity minimum above max'),
        pytest.param('storage_coeff', 1.5, 'charge', 'charging_efficiency_is_a_fraction', id='eta_charge above 1'),
        pytest.param('storage_coeff', -0.5, 'discharge', 'discharging_efficiency_is_a_fraction', id='eta_d above 1'),
        pytest.param('retention', 1.2, None, 'loss_is_a_fraction', id='a negative loss'),
    ],
)
def test_an_edited_table_that_breaks_an_assumption_is_refused(
    parameter: str, value: float, side: str | None, assumption: str
) -> None:
    """The program names the assumption the numbers break, whichever path they came by."""
    bound = _bound()
    bound[parameter] = _edit(bound[parameter], value, side)
    with pytest.raises(lpspec.DataError, match=assumption):
        lpspec.build(program().expand('sos'), bound)
