from collections.abc import Mapping
from typing import Any

from fluxopt.components import Converter, Port
from fluxopt.elements import (
    PENALTY_EFFECT_ID,
    Carrier,
    Effect,
    Flow,
    Investment,
    PiecewiseConversion,
    Sizing,
    Status,
    Storage,
)
from fluxopt.flow_system import FlowSystem
from fluxopt.schema import all_element_schemas, element_schema, from_dict, to_dict
from fluxopt.types import (
    ProfileRef,
    TimeIndex,
    Timesteps,
    Variate,
    as_dataarray,
)


def optimize(
    timesteps: Timesteps,
    carriers: list[Carrier],
    effects: list[Effect],
    ports: list[Port],
    objective: str | dict[str, float],
    converters: list[Converter] | None = None,
    storages: list[Storage] | None = None,
    dt: float | list[float] | None = None,
    periods: list[int] | None = None,
    period_weights: list[float] | None = None,
    profiles: Mapping[str, Any] | None = None,
    solver: str = 'highs',
    **solver_options: Any,
) -> Any:
    """Declare a system and solve it: [`FlowSystem.optimize`][fluxopt.FlowSystem.optimize] in one call.

    Args:
        timesteps: Time index for the optimization horizon.
        carriers: Carrier declarations.
        effects: Effects to track (costs, emissions, etc.).
        ports: System boundary ports with imports/exports.
        objective: Effect(s) to minimize. A single name, or a dict
            mapping effect names to objective weights
            (``{'cost': 1, 'co2': 50}``) — tracked effect totals are
            unaffected by the weighting. The built-in ``'penalty'`` effect
            is added at weight 1.0 unless the dict names it
            (``{'cost': 1, 'penalty': 0}`` opts out).
        converters: Linear converters between carriers.
        storages: Energy storages.
        dt: Timestep duration in hours. Auto-derived if None.
        periods: Integer period labels for multi-period optimization.
        period_weights: Explicit weights per period. Inferred from gaps if None.
        profiles: Mapping from ``ProfileRef.dataset`` to a dataset (or mapping)
            holding referenced time series. Required if any element uses a
            ``ProfileRef``.
        solver: Solver name — ``highs``, or ``gurobi`` with specsolve's extra.
        **solver_options: Passed to the solver verbatim, in its own vocabulary.

    Returns:
        specsolve's result; see [`FlowSystem.optimize`][fluxopt.FlowSystem.optimize].
    """
    system = FlowSystem(
        timesteps=timesteps,
        carriers=carriers,
        effects=effects,
        ports=ports,
        objective=objective,
        converters=converters or [],
        storages=storages or [],
        dt=dt,
        periods=periods,
        period_weights=period_weights,
    )
    return system.optimize(profiles, solver=solver, **solver_options)


__all__ = [
    'PENALTY_EFFECT_ID',
    'Carrier',
    'Converter',
    'Effect',
    'Flow',
    'FlowSystem',
    'Investment',
    'PiecewiseConversion',
    'Port',
    'ProfileRef',
    'Sizing',
    'Status',
    'Storage',
    'TimeIndex',
    'Timesteps',
    'Variate',
    'all_element_schemas',
    'as_dataarray',
    'element_schema',
    'from_dict',
    'optimize',
    'to_dict',
]
