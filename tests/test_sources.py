"""The sources are the numbers a spec is bound to: one table per declared name, the caller's to edit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mathspec as ms
import numpy as np
import polars as pl
import pytest
import specsolve
from conftest import read, ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port

if TYPE_CHECKING:
    from pathlib import Path


def _system() -> FlowSystem:
    return FlowSystem(
        timesteps=ts(3),
        carriers=[Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=np.array([5, 8, 6]))]),
            Port(id='grid', imports=[Flow(carrier='heat', size=100, effects_per_flow_hour={'cost': 2})]),
        ],
    )


def _frame(table: object) -> pl.DataFrame:
    return table if isinstance(table, pl.DataFrame) else pl.from_pandas(table)  # type: ignore[arg-type]


class TestReadable:
    def test_the_sources_cover_every_declaration_of_the_spec(self) -> None:
        """`spec()` is the equations, `sources()` the numbers bound to them."""
        system = _system()
        spec, sources = system.spec(), system.sources()
        declared = set(spec.parameters) | set(spec.dimensions) | set(spec.relations)
        assert declared <= set(sources), f'no table for {sorted(declared - set(sources))}'

    def test_a_table_carries_the_numbers_the_elements_declared(self) -> None:
        """`effects_per_flow_hour` is the flow's rate times the step duration."""
        charged = _frame(_system().sources()['effects_per_flow_hour'])
        assert charged['flow'].unique().to_list() == ['grid(heat)']
        assert charged['value'].to_list() == [2.0, 2.0, 2.0]

    def test_a_relation_is_a_map_from_one_dimension_into_another(self) -> None:
        """`carrier_of: {key: flow, values: carrier}` is a `(flow, carrier)` table."""
        carrier_of = _frame(_system().sources()['carrier_of'])
        assert carrier_of.columns == ['flow', 'carrier']
        assert carrier_of.sort('flow')['carrier'].to_list() == ['heat', 'heat']

    def test_a_relation_holds_only_the_labels_it_is_defined_at(self) -> None:
        """No storage here, so nothing charges one: an absent map, not null rows."""
        sources = _system().sources()
        assert _frame(sources['port_of']).is_empty()
        assert _frame(sources['carrier_of']).height == 2


class TestArchive:
    def test_the_archive_carries_the_spec_its_sources_and_the_answer(self, tmp_path: Path) -> None:
        system = _system()
        result = system.optimize(archive=tmp_path / 'run.zip')
        back = specsolve.load_archive(tmp_path / 'run.zip')

        assert back.answer.objective == result.objective
        assert set(back.sources) >= set(system.spec().parameters), 'every bound parameter travels with it'
        assert specsolve.solve(back.spec, back.sources).objective == result.objective, 'and solves again as it was'


class TestDerivedNotAuthored:
    def test_a_coefficient_binds_as_the_step_charge_and_a_cross_effect_as_its_share(self) -> None:
        """The step length is multiplied in; `contribution_from` binds as the chained share."""
        system = FlowSystem(
            timesteps=ts(2),
            carriers=[Carrier(id='heat')],
            effects=[Effect(id='co2'), Effect(id='cost', contribution_from={'co2': 10.0})],
            objective='cost',
            ports=[
                Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=np.array([1, 1]))]),
                Port(id='grid', imports=[Flow(carrier='heat', size=10, effects_per_flow_hour={'co2': 3})]),
            ],
        )
        sources = system.sources()
        charged = _frame(sources['effects_per_flow_hour'])
        assert dict(zip(charged['effect'], charged['value'], strict=True)) == {'co2': 3.0}, 'only what the flow emits'
        share = _frame(sources['share'])
        assert set(zip(share['effect'], share['source'], share['value'], strict=True)) == {('cost', 'co2', 10.0)}


REGIONS = """
dimensions:
  region: {dtype: str}
relations:
  region_of: {key: flow, values: region}
parameters:
  region_cap: {dims: [region, time, period]}
constraints:
  region_limit:
    dims: [region, time, period]
    where: region_cap
    expression: sum(rate, by=region_of, over=flow, into=region) <= region_cap
"""


def test_a_caller_s_own_relation_reaches_the_constraint_that_reads_it() -> None:
    """Capping the cheap region forces the expensive one, which changes the cost."""
    system = FlowSystem(
        timesteps=ts(2),
        carriers=[Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(id='demand', exports=[Flow(carrier='heat', size=1, fixed_relative_profile=np.array([4, 4]))]),
            Port(id='north', imports=[Flow(carrier='heat', size=10, effects_per_flow_hour={'cost': 1})]),
            Port(id='south', imports=[Flow(carrier='heat', size=10, effects_per_flow_hour={'cost': 5})]),
        ],
    )
    assert_allclose(read(system.optimize(), 'effect_total').sel(effect='cost').item(), 8.0, rtol=1e-6)

    spec = ms.override(system.spec(), [REGIONS])
    sources = system.sources() | {
        'region': pl.DataFrame({'region': ['cheap']}),
        'region_of': pl.DataFrame({'flow': ['north(heat)'], 'region': ['cheap']}),
        'region_cap': pl.DataFrame({'region': ['cheap'], 'time': ts(2)[:1], 'period': [0], 'value': [1.0]}),
    }
    result = specsolve.solve(spec, sources)
    assert read(result, 'effect_total').sel(effect='cost').item() == pytest.approx(20.0), 't0: 1 + 3 * 5; t1: 4'
