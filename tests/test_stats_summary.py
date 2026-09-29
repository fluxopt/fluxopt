"""The headline quantities a result is read in, as reported expressions.

Each is named in ``math/program/reporting.yaml`` (or, for ``size`` and
``flow_hours``, in the fragment that uses them) and evaluated at the solve.
"""

import numpy as np
from conftest import read, ts

from fluxopt import Carrier, Effect, Flow, Port, Sizing, Storage, optimize

# Fixed three-step demand: 50, 80, 60 MW -> 190 MWh at dt=1.
_DEMAND_PROFILE = [0.5, 0.8, 0.6]
_DEMAND_ENERGY = 190.0
_GRID = 'grid(elec)'


def _solve(source, *, dt=None, storages=()):
    """One-bus electricity model: `source` imports, a fixed demand exports."""
    demand = Flow(carrier='elec', size=100, fixed_relative_profile=_DEMAND_PROFILE)
    return optimize(
        timesteps=ts(3),
        dt=dt,
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[source]), Port(id='demand', exports=[demand])],
        storages=list(storages),
    )


def _reported(result, name):
    return read(result, name, 'expression')


def test_the_headline_quantities_of_a_sized_flow():
    result = _solve(Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04}))

    assert np.isclose(read(result, 'effect_total').sel(effect='cost').item(), _DEMAND_ENERGY * 0.04)
    assert _reported(result, 'total_duration').item() == 3.0
    assert _reported(result, 'size').sel(flow=_GRID).item() == 200
    assert np.isclose(_reported(result, 'flow_hours').sel(flow=_GRID).item(), _DEMAND_ENERGY)
    assert np.isclose(_reported(result, 'capacity_factor').sel(flow=_GRID).item(), _DEMAND_ENERGY / (200 * 3)), (
        'energy carried over running at full size throughout'
    )


def test_an_unsized_flow_reads_size_zero_and_has_no_capacity_factor():
    """No size reads 0, and a share of running at size 0 is not a number: the capacity factor has no row.

    specsolve 0.2.1 leaves a quotient absent where its divisor is zero
    (fluxopt/specsolve#1776); before, this read ``inf``.
    """
    result = _solve(Flow(carrier='elec', effects_per_flow_hour={'cost': 0.04}))

    assert _reported(result, 'size').sel(flow=_GRID).item() == 0
    assert _GRID not in _reported(result, 'capacity_factor').labels('flow')
    assert np.isclose(_reported(result, 'flow_hours').sel(flow=_GRID).item(), _DEMAND_ENERGY)


def test_a_decided_size_is_the_size_the_solver_chose():
    """A per-size cost makes the solver pick the smallest feasible size: the peak."""
    sizing = Sizing(size_min=0, size_max=500, effects_per_size={'cost': 1.0})
    result = _solve(Flow(carrier='elec', size=sizing, effects_per_flow_hour={'cost': 0.04}))

    size = _reported(result, 'size').sel(flow=_GRID).item()
    assert np.isclose(size, read(result, 'chosen_size').sel(flow=_GRID).item())
    assert np.isclose(size, 80), 'the peak demand'
    assert np.isclose(_reported(result, 'capacity_factor').sel(flow=_GRID).item(), _DEMAND_ENERGY / (size * 3))


def test_capacity_factor_is_horizon_independent():
    """Scaling every timestep duration leaves the capacity factor unchanged while throughput scales."""

    def source():
        return Flow(carrier='elec', size=200, effects_per_flow_hour={'cost': 0.04})

    short = _solve(source(), dt=[1.0, 1.0, 1.0])
    long = _solve(source(), dt=[2.0, 2.0, 2.0])

    assert np.isclose(_reported(long, 'total_duration').item(), 2 * _reported(short, 'total_duration').item())
    assert np.isclose(
        _reported(long, 'flow_hours').sel(flow=_GRID).item(),
        2 * _reported(short, 'flow_hours').sel(flow=_GRID).item(),
    )
    assert np.isclose(
        _reported(long, 'capacity_factor').sel(flow=_GRID).item(),
        _reported(short, 'capacity_factor').sel(flow=_GRID).item(),
    )


def test_a_storage_reports_its_capacity_and_mean_level():
    source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': [0.1, 0.9, 0.1]})
    demand = Flow(carrier='elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5])
    storage = Storage(
        id='batt', charging=Flow(carrier='elec', size=80), discharging=Flow(carrier='elec', size=80), capacity=80
    )
    result = optimize(
        timesteps=ts(3),
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[source]), Port(id='load', exports=[demand])],
        storages=[storage],
    )

    assert _reported(result, 'capacity').sel(storage='batt').item() == 80
    level = read(result, 'level').sel(storage='batt')
    mean = _reported(result, 'relative_mean_level').sel(storage='batt').item()
    assert 0 <= mean <= 1
    assert np.isclose(mean, float(level.sum()) / 3 / 80), 'the step-weighted mean level over the capacity'


def test_a_storage_built_at_zero_has_no_mean_level():
    """A storage the solver builds at capacity 0 reports that capacity, and no mean level over it."""
    source = Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.1})
    demand = Flow(carrier='elec', size=50, fixed_relative_profile=[0.5, 0.5, 0.5])
    storage = Storage(
        id='batt',
        charging=Flow(carrier='elec', size=80),
        discharging=Flow(carrier='elec', size=80),
        capacity=Sizing(size_min=0, size_max=80, effects_per_size={'cost': 1.0}),
    )
    result = optimize(
        timesteps=ts(3),
        carriers=[Carrier(id='elec')],
        effects=[Effect(id='cost')],
        objective='cost',
        ports=[Port(id='grid', imports=[source]), Port(id='load', exports=[demand])],
        storages=[storage],
    )

    assert np.isclose(_reported(result, 'capacity').sel(storage='batt').item(), 0), 'a flat price buys no storage'
    assert 'batt' not in _reported(result, 'relative_mean_level').labels('storage')
