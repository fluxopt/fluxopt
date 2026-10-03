"""The math is composed from one fragment per feature.

`effects.yaml` reads what the features charge directly as sums no file
defines, and each feature file adds its own term to them with `adds_to:`.
These tests hold the files to that shape, and show that a caller extends the
ledger by adding a file rather than by editing one.
"""

from __future__ import annotations

import mathspec as ms
import polars as pl
import pytest
import specsolve
from conftest import read, ts

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    FlowSystem,
    Investment,
    PiecewiseConversion,
    Port,
    Sizing,
    Status,
    Storage,
)
from fluxopt.math import CORE, PROGRAM, build_sources, program
from fluxopt.math.sources import _bind

FRAGMENTS = {path.stem: path for path in sorted(PROGRAM.glob('*.yaml'))}


@pytest.mark.parametrize('name', sorted(FRAGMENTS))
def test_every_fragment_loads_alone(name: str) -> None:
    """A fragment is a whole spec: it declares what it builds and reads the rest under `given:`."""
    ms.to_spec(FRAGMENTS[name])


def test_the_fragments_leave_nothing_given() -> None:
    """Every name a fragment reads, another fragment declares."""
    assert not program().to_dict().get('given'), 'a fragment reads a name no fragment declares'


def test_the_ledger_is_the_sum_of_the_feature_terms() -> None:
    """What the features charge directly is written by merge, one term per feature."""
    math = program()
    assert math.expressions['direct_step'].expression == 'flow_hour_costs + status_costs'
    assert math.expressions['direct_lump'].expression == 'investment_costs + sizing_costs + storage_costs'


GRID_FEE = """
description: A fee on every unit a flow carries, charged to one effect.
dimensions:
  time: {dtype: datetime}
  period: {dtype: int}
  effect: {dtype: str}
  flow: {dtype: str}
parameters:
  fee: {dims: [flow, effect]}
given:
  parameters:
    dt: {dims: [time]}
  variables:
    rate: {dims: [flow, time, period]}
  expressions:
    direct_step: {dims: [effect, time, period]}
expressions:
  fees:
    expression: sum(rate * dt * fee, over=flow)
    adds_to: direct_step
"""


def test_a_new_fragment_adds_to_the_ledger_without_editing_it() -> None:
    """A fee in a file of its own reaches the objective; no shipped file changes."""
    system = FlowSystem(
        timesteps=ts(3),
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[
            Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
            Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
        ],
    )
    base = read(system.optimize(), 'effect_total').sel(effect='cost').item()

    math = ms.merge([system.spec(), GRID_FEE])
    fee = pl.DataFrame({'flow': ['grid(elec)'], 'effect': ['cost'], 'value': [2.0]})
    charged = read(specsolve.solve(math, system.sources() | {'fee': fee}), 'effect_total').sel(effect='cost').item()

    assert charged == pytest.approx(3 * base, rel=1e-6), 'a fee of 2 on a unit cost of 1 triples the bill'


def _heat(**elements: object) -> FlowSystem:
    """Gas or district heat for a fixed demand; *elements* add what a case needs."""
    ports = [
        Port(id='gas', imports=[Flow(carrier='gas', size=100, effects_per_flow_hour={'cost': 1.0})]),
        Port(id='district', imports=[Flow(carrier='heat', size=100, effects_per_flow_hour={'cost': 5.0})]),
        Port(id='demand', exports=[Flow(carrier='heat', size=100, fixed_relative_profile=[0.2, 0.6, 0.4])]),
    ]
    return FlowSystem(
        timesteps=ts(3),
        carriers=[Carrier(id='gas'), Carrier(id='heat')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=ports,
        **elements,
    )


def _boiler(**heat: object) -> Converter:
    """A boiler whose heat flow takes *heat*: a size, a status, a ramp."""
    out = Flow(carrier='heat', **{'size': 100, **heat})
    return Converter.boiler('boiler', 0.9, Flow(carrier='gas', size=100), out)


def _curve(**conversion: object) -> Converter:
    return Converter(
        id='curve',
        inputs=[Flow(carrier='gas', short_id='fuel', size=100)],
        outputs=[Flow(carrier='heat', size=100)],
        conversion=PiecewiseConversion(points={'fuel': [10, 40, 100], 'heat': [8, 35, 80]}, **conversion),
    )


def _tank() -> Storage:
    return Storage(
        id='tank', charging=Flow(carrier='heat', size=20), discharging=Flow(carrier='heat', size=20), capacity=50
    )


ON = {'status': Status(effects_per_startup={'cost': 1.0}), 'relative_rate_min': 0.2}
RAMP = {'ramp_up_per_hour': 0.5, 'ramp_down_per_hour': 0.5}
SIZED = {'size': Sizing(size_min=10, size_max=100, effects_per_size={'cost': 0.1})}
INVESTED = {'size': Investment(size_min=10, size_max=100, effects_per_size_at_build={'cost': 0.1})}
PERIODS = {'periods': [2020, 2030], 'period_weights': [10, 10]}

#: Each system, and the fragments it composes besides the core.
CASES = [
    pytest.param({}, set(), id='flows alone'),
    pytest.param({'converters': [_boiler()]}, {'converters'}, id='a linear converter'),
    pytest.param({'converters': [_curve()]}, {'piecewise'}, id='a piecewise converter'),
    pytest.param({'storages': [_tank()]}, {'storage', 'lump'}, id='a storage'),
    pytest.param({'converters': [_boiler(**RAMP)]}, {'converters', 'ramps'}, id='a ramp'),
    pytest.param({'converters': [_boiler(**SIZED)]}, {'converters', 'sizing', 'lump'}, id='a sizing'),
    pytest.param(
        {'converters': [_boiler(**INVESTED)], **PERIODS},
        {'converters', 'sizing', 'investment', 'lump'},
        id='an investment',
    ),
    pytest.param({'converters': [_boiler(**ON)]}, {'converters', 'status'}, id='a status'),
    pytest.param(
        {'converters': [_boiler(**ON, **SIZED)]},
        {'converters', 'status', 'sizing', 'lump', 'status_sizing'},
        id='a status on a sizing',
    ),
    pytest.param(
        {'converters': [_boiler(**ON, **RAMP)]},
        {'converters', 'status', 'ramps', 'ramps_status'},
        id='a status on a ramp',
    ),
    pytest.param(
        {'converters': [_curve(status=Status())]}, {'piecewise', 'status', 'piecewise_status'}, id='a gated curve'
    ),
]


def _composed(spec: object) -> set[str]:
    """The fragments whose every own name *spec* holds, by file stem."""
    out = set()
    for stem, path in FRAGMENTS.items():
        own = ms.to_spec(path)
        names = {*own.constraints, *own.expressions, *own.variables}
        held = {*spec.constraints, *spec.expressions, *spec.variables}  # type: ignore[attr-defined]
        if names <= held:
            out.add(stem)
        else:
            assert not names & held, f'{stem} is composed in part: {sorted(names & held)}'
    return out


@pytest.mark.parametrize(('elements', 'optional'), CASES)
def test_a_system_composes_the_core_and_the_fragments_its_elements_use(elements: dict, optional: set[str]) -> None:
    """A fragment no element needs is left out of the spec, and so is every table it declares."""
    system = _heat(**elements)
    assert _composed(system.spec()) == CORE | optional
    declared = {*system.spec().parameters, *system.spec().relations}
    assert set(system.sources()) - {*system.spec().dimensions} == declared, 'one table per declared name, no more'


@pytest.mark.parametrize(('elements', 'optional'), CASES)
def test_the_composed_spec_solves_as_the_whole_program_does(elements: dict, optional: set[str]) -> None:
    """Leaving a fragment out changes no answer: every row it would build is absent anyway."""
    system = _heat(**elements)
    whole = build_sources(
        timesteps=system.timesteps,
        carriers=system.carriers,
        effects=system.effects,
        ports=system.ports,
        objective=system.objective,
        converters=system.converters,
        storages=system.storages,
        periods=system.periods,
        period_weights=system.period_weights,
    )
    expected = specsolve.solve(program().expand('sos'), whole).objective
    assert system.optimize().objective == pytest.approx(expected, rel=1e-9)


def test_a_table_with_rows_that_no_composed_fragment_reads_is_refused() -> None:
    """The guard behind every left-out fragment: a dropped row would be a dropped constraint."""
    system = _heat(converters=[_boiler()])
    tables = system.sources() | {'port_of': pl.DataFrame({'flow': ['x'], 'storage': ['tank'], 'side': ['charge']})}
    with pytest.raises(RuntimeError, match=r"\['port_of'\] hold rows"):
        _bind(tables, system.spec())
