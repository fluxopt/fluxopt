from collections.abc import Callable
from typing import Any

from fluxopt.components import Converter, Port
from fluxopt.elements import (
    PENALTY_EFFECT_ID,
    Carrier,
    ConversionCurve,
    Effect,
    Flow,
    Investment,
    Sizing,
    Status,
    Storage,
)
from fluxopt.model import FlowSystem
from fluxopt.model_data import Dims, ModelData
from fluxopt.results import Result
from fluxopt.types import (
    IdList,
    TimeIndex,
    TimeSeries,
    Timesteps,
    as_dataarray,
)


def optimize(
    timesteps: Timesteps,
    carriers: list[Carrier],
    effects: list[Effect],
    ports: list[Port],
    objective_effects: str | list[str],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
    periods: list[int] | None = None,
    period_weights: list[float] | None = None,
    solver: str = 'highs',
    customize: Callable[[FlowSystem], None] | None = None,
    **kwargs: Any,
) -> Result:
    """Build data, build model, optimize, return results.

    Args:
        timesteps: Time index for the optimization horizon.
        carriers: Carrier declarations.
        effects: Effects to track (costs, emissions, etc.).
        ports: System boundary ports with imports/exports.
        objective_effects: Effect name(s) to minimize. Sum of named effect totals.
        converters: Linear converters between carriers.
        storages: Energy storages.
        dt: Timestep duration in hours. Auto-derived if None.
        periods: Integer period labels for multi-period optimization.
        period_weights: Explicit weights per period. Inferred from gaps if None.
        solver: Solver backend name.
        customize: Optional callback to modify the linopy model between build and solve.
            Receives the built FlowSystem; use ``model.m`` to add variables/constraints.
        **kwargs: Passed through to ``linopy.Model.solve()``.
    """
    data = ModelData.build(
        timesteps,
        carriers,
        effects,
        ports,
        converters,
        storages,
        dt,
        periods=periods,
        period_weights=period_weights,
    )
    model = FlowSystem(data)
    return model.optimize(objective_effects=objective_effects, customize=customize, solver=solver, **kwargs)


__all__ = [
    'PENALTY_EFFECT_ID',
    'Carrier',
    'ConversionCurve',
    'Converter',
    'Dims',
    'Effect',
    'Flow',
    'FlowSystem',
    'IdList',
    'Investment',
    'ModelData',
    'Port',
    'Result',
    'Sizing',
    'Status',
    'Storage',
    'TimeIndex',
    'TimeSeries',
    'Timesteps',
    'as_dataarray',
    'optimize',
]
