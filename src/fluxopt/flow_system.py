"""``FlowSystem`` — the declarative top of the element layer.

A ``FlowSystem`` is an inert, validated description of a flow system: the
same lists you would pass to :func:`fluxopt.optimize`, gathered into one object
that round-trips to dict/YAML. It carries *structure* (components, effects,
config) and ``ProfileRef`` references to time-series; the actual series are
supplied at solve time via ``sources`` (``system.optimize(sources=...)``) and
resolved into arrays just before the model is built.

The FlowSystem has no modeling behavior of its own — ``.optimize()`` runs the existing
pipeline (:meth:`ModelData.build` → :class:`FlowSystemModel`). Declaration (the system)
and use (building/solving) stay separate.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import field
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from fluxopt.components import Converter, Port
from fluxopt.elements import Carrier, Effect, Storage
from fluxopt.model import FlowSystemModel
from fluxopt.model_data import ModelData
from fluxopt.schema import from_dict, to_dict
from fluxopt.types import IdList, ProfileRef, Timesteps

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from fluxopt.results import Result

_PYDANTIC_CFG = ConfigDict(arbitrary_types_allowed=True)


def _resolve_refs(obj: Any, sources: Mapping[str, Any]) -> Any:
    """Recursively replace every ``ProfileRef`` in *obj* with a resolved array.

    Walks element dataclasses, dicts, lists, and ``IdList`` containers,
    mutating in place. Non-container leaves (scalars, arrays) pass through.

    Args:
        obj: The value or element to walk.
        sources: Mapping passed to :meth:`ProfileRef.resolve`.
    """
    if isinstance(obj, ProfileRef):
        return obj.resolve(sources)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _resolve_refs(value, sources)
        return obj
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _resolve_refs(value, sources)
        return obj
    if isinstance(obj, IdList):
        for item in obj:
            _resolve_refs(item, sources)
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            setattr(obj, f.name, _resolve_refs(getattr(obj, f.name), sources))
        return obj
    return obj


@dataclass(config=_PYDANTIC_CFG)
class FlowSystem:
    """A declarative flow-system description (see module docstring).

    Args:
        timesteps: Time index for the optimization horizon.
        carriers: Carrier declarations.
        effects: Effects to track (costs, emissions, …).
        ports: System boundary ports with imports/exports.
        objective_effects: Effect(s) to minimize — a name or ``{effect: weight}``.
        converters: Linear/piecewise converters between carriers.
        storages: Energy storages.
        dt: Timestep duration in hours. Auto-derived if None.
        periods: Integer period labels for multi-period optimization.
        period_weights: Explicit weights per period. Inferred from gaps if None.
    """

    timesteps: Timesteps
    carriers: list[Carrier]
    effects: list[Effect]
    ports: list[Port]
    objective_effects: str | dict[str, float]
    converters: list[Converter] = field(default_factory=list)
    storages: list[Storage] = field(default_factory=list)
    dt: float | list[float] | None = None
    periods: list[int] | None = None
    period_weights: list[float] | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FlowSystem:
        """Build a system from a mapping (e.g. parsed YAML/JSON)."""
        return from_dict(cls, data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FlowSystem:
        """Load a system from a YAML file.

        Args:
            path: Path to a YAML document describing the system.
        """
        import yaml

        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the system to a JSON-safe dict (profiles as ``ProfileRef``)."""
        return to_dict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write the system to a YAML file.

        Args:
            path: Destination path.
        """
        import yaml

        with open(path, 'w') as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def optimize(
        self,
        sources: Mapping[str, Any] | None = None,
        *,
        solver: str = 'highs',
        customize: Callable[[FlowSystemModel], None] | None = None,
        **kwargs: Any,
    ) -> Result:
        """Resolve profile references, build the model, and solve.

        The system is left untouched — ``ProfileRef`` resolution runs on a copy —
        so it stays reusable across different ``sources``.

        Args:
            sources: Mapping from ``ProfileRef.source`` to a dataset (or mapping)
                holding the referenced variables. Required if the system uses
                any ``ProfileRef``.
            solver: Solver backend name.
            customize: Callback to modify the linopy model between build and
                solve; receives the built ``FlowSystemModel`` (use ``model.m``).
            **kwargs: Passed through to ``linopy.Model.solve()``.
        """
        carriers, effects, ports, converters, storages = (
            copy.deepcopy(self.carriers),
            copy.deepcopy(self.effects),
            copy.deepcopy(self.ports),
            copy.deepcopy(self.converters),
            copy.deepcopy(self.storages),
        )
        srcs = sources or {}
        for group in (carriers, effects, ports, converters, storages):
            _resolve_refs(group, srcs)

        data = ModelData.build(
            self.timesteps,
            carriers,
            effects,
            ports,
            converters,
            storages,
            self.dt,
            periods=self.periods,
            period_weights=self.period_weights,
        )
        model = FlowSystemModel(data)
        return model.optimize(
            objective_effects=self.objective_effects,
            customize=customize,
            solver=solver,
            **kwargs,
        )
