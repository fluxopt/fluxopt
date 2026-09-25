"""The math is composed from one fragment per feature.

`effects.yaml` declares the two halves of the ledger as sums with a frame and
no body, and each feature file adds its own term to them. These tests hold
the files to that shape, and show that a caller extends the ledger by adding
a file rather than by editing one.
"""

from __future__ import annotations

import mathspec as ms
import pandas as pd
import pytest
from conftest import ts

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port
from fluxopt.math import PROGRAM, program

FRAGMENTS = {path.stem: path for path in sorted(PROGRAM.glob('*.yaml'))}


@pytest.mark.parametrize('name', sorted(FRAGMENTS))
def test_every_fragment_loads_alone(name: str) -> None:
    """A fragment is a whole spec: it declares what it builds and reads the rest under `given:`."""
    ms.to_spec(FRAGMENTS[name])


def test_the_fragments_leave_nothing_given() -> None:
    """Every name a fragment reads, another fragment declares."""
    assert not program().to_dict().get('given'), 'a fragment reads a name no fragment declares'


def test_the_ledger_is_the_sum_of_the_feature_terms() -> None:
    """Each half of the ledger is written by merge, one term per feature."""
    math = program()
    assert math.expressions['effect_temporal'].expression == 'flow_hour_costs + status_costs'
    assert math.expressions['effect_lump'].expression == 'investment_costs + sizing_costs + storage_costs'


GRID_FEE = """
description: A fee on every unit a flow carries, charged to one effect.
dimensions:
  time: {dtype: int}
  period: {dtype: int}
  effect: {dtype: str}
  flow: {dtype: str}
parameters:
  fee: {dims: [flow, effect]}
given:
  parameters:
    dt: {dims: [time]}
    time_weight: {dims: [time]}
  variables:
    rate: {dims: [flow, time, period]}
  expressions:
    effect_temporal: {dims: [effect, period], term: fees}
expressions:
  fees:
    expression: sum(sum(rate * dt * time_weight, over=time) * fee, over=flow)
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
    base = system.optimize().effect_totals.sel(effect='cost').item()

    math = ms.merge({**FRAGMENTS, 'grid_fee': GRID_FEE})
    fee = pd.DataFrame({'flow': ['grid(elec)'], 'effect': ['cost'], 'value': [2.0]})
    charged = system.optimize(math=math, parameters={'fee': fee}).effect_totals.sel(effect='cost').item()

    assert charged == pytest.approx(3 * base, rel=1e-6), 'a fee of 2 on a unit cost of 1 triples the bill'
