"""``FlowSystem`` — the declarative top of the element layer.

A ``FlowSystem`` is an inert, validated description of a flow system: the
same lists you would pass to [`fluxopt.optimize`][fluxopt.optimize], gathered into one object
that round-trips to dict/YAML. It carries *structure* (components, effects,
config) and ``ProfileRef`` references to time-series; the actual series are
supplied at solve time via ``profiles`` (``system.optimize(profiles=...)``), as
polars tables, and resolved just before the tables are built.

The FlowSystem has no modeling behavior of its own: [`FlowSystem.sources`][fluxopt.FlowSystem.sources]
builds the tables bound to the math program, and ``.optimize()`` solves them.
Declaration (the system) and use (building/solving) stay separate.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fluxopt.components import Converter, Port
from fluxopt.elements import Carrier, Effect, Storage
from fluxopt.schema import from_dict, to_dict
from fluxopt.types import ProfileRef, Timesteps, normalize_timesteps
from fluxopt.validation import validate_system

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import polars as pl
    import specsolve

_PYDANTIC_CFG = ConfigDict(arbitrary_types_allowed=True, extra='forbid')


def _resolve_refs(obj: Any, profiles: Mapping[str, Any]) -> Any:
    """Recursively replace every ``ProfileRef`` in *obj* with a resolved array.

    Walks element dataclasses, dicts and lists,
    mutating in place. Non-container leaves (scalars, arrays) pass through.

    Args:
        obj: The value or element to walk.
        profiles: Mapping passed to [`ProfileRef.resolve`][fluxopt.ProfileRef.resolve].
    """
    if isinstance(obj, ProfileRef):
        return obj.resolve(profiles)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = _resolve_refs(value, profiles)
        return obj
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = _resolve_refs(value, profiles)
        return obj
    if isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            setattr(obj, name, _resolve_refs(getattr(obj, name), profiles))
        return obj
    return obj


def _collect_profile_refs(obj: Any, path: str, out: list[tuple[str, ProfileRef]]) -> None:
    """Recursively collect every ``ProfileRef`` in *obj* with a readable path.

    Path segments name elements by class and id (``Flow('Demand(Heat)')``) and
    descend through fields, dict keys, and list positions.

    Args:
        obj: The value or element to walk.
        path: Path accumulated so far.
        out: Collected ``(path, ref)`` pairs, appended in walk order.
    """
    if isinstance(obj, ProfileRef):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _collect_profile_refs(value, f'{path}[{key!r}]', out)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _collect_profile_refs(value, f'{path}[{i}]', out)
    elif isinstance(obj, BaseModel):
        element_id = getattr(obj, 'id', '') or getattr(obj, 'short_id', '')
        base = f'{type(obj).__name__}({element_id!r})' if element_id else path
        for name in type(obj).model_fields:
            _collect_profile_refs(getattr(obj, name), f'{base}.{name}', out)


def _check_profiles_cover(refs: list[tuple[str, ProfileRef]], profiles: Mapping[str, pl.DataFrame]) -> None:
    """Raise one comprehensive error if any ref cannot be resolved.

    Args:
        refs: ``(path, ref)`` pairs from `_collect_profile_refs`.
        profiles: The solve-time profile supply.

    Raises:
        KeyError: Listing every unresolvable ref with its element/field path.
    """
    missing = []
    for path, ref in refs:
        if ref.table not in profiles:
            missing.append(f'{path}: table {ref.table!r} not supplied (have {sorted(profiles)})')
        elif ref.column not in profiles[ref.table].columns:
            missing.append(f'{path}: column {ref.column!r} not in table {ref.table!r}')
    if missing:
        raise KeyError('unresolvable ProfileRef(s):\n  ' + '\n  '.join(missing))


class FlowSystem(BaseModel):
    """A declarative flow-system description (see module docstring)."""

    model_config = _PYDANTIC_CFG

    timesteps: Timesteps
    """Time index for the optimization horizon."""
    carriers: list[Carrier]
    """Carrier declarations."""
    effects: list[Effect]
    """Effects to track (costs, emissions, …)."""
    ports: list[Port]
    """System boundary ports with imports/exports."""
    objective: str | dict[str, float]
    """Effect(s) to minimize — a name or ``{effect: weight}``. Must name at
    least one non-penalty effect."""
    converters: list[Converter] = Field(default_factory=list)
    """Linear/piecewise converters between carriers."""
    storages: list[Storage] = Field(default_factory=list)
    """Energy storages."""
    dt: float | list[float] | None = None
    """Timestep duration in hours. Auto-derived if None."""
    periods: list[int] | None = None
    """Integer period labels for multi-period optimization."""
    period_weights: list[float] | None = None
    """Explicit weights per period. Inferred from gaps if None."""

    @field_validator('timesteps', mode='before')
    @classmethod
    def _timestamps_only(cls, value: Any) -> Any:
        """Refuse numbered steps before pydantic reads a number as seconds since 1970.

        ISO strings pass through to pydantic's own parsing, which is how a
        dumped system loads again. A ``pl.Series`` is stored as its datetimes,
        so the system serializes the same either way.
        """
        if not (isinstance(value, list) and value and all(isinstance(t, str) for t in value)):
            return normalize_timesteps(value).to_list()
        return value

    @model_validator(mode='after')
    def _validate_references(self) -> FlowSystem:
        """Fail fast on undeclared references and duplicate ids (at construction/load)."""
        validate_system(
            carriers=self.carriers,
            effects=self.effects,
            ports=self.ports,
            converters=self.converters,
            storages=self.storages,
            objective=self.objective,
        )
        return self

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

    def required_profiles(self) -> dict[str, set[str]]:
        """Enumerate the external data this system needs, as ``{table: columns}``.

        The contract a ``profiles`` supply must cover before
        [`sources`][fluxopt.FlowSystem.sources] or [`optimize`][fluxopt.FlowSystem.optimize] can run. Empty when every value
        is inline.
        """
        refs: list[tuple[str, ProfileRef]] = []
        for group in (self.carriers, self.effects, self.ports, self.converters, self.storages):
            _collect_profile_refs(group, '', refs)
        out: dict[str, set[str]] = {}
        for _, ref in refs:
            out.setdefault(ref.table, set()).add(ref.column)
        return out

    def spec(self) -> Any:
        """The equations this system is solved as, before any number is bound.

        A `mathspec.Spec`, composed from the fragments under
        [`fluxopt.math.PROGRAM`][fluxopt.math.PROGRAM] that the system uses: the
        [`CORE`][fluxopt.math.CORE], and a storage, converter, piecewise or ramp fragment only
        where an element needs it. The piecewise special-ordered sets are
        written out as binaries so every solver takes it. Read it, typeset it (``mathspec.to_latex``), or
        extend it — ``mathspec.override`` it with a patch, or ``merge`` a
        fragment of your own onto the shipped ones — and solve the result
        with `specsolve.solve` against [`sources`][fluxopt.FlowSystem.sources].
        """
        return self._program().expand('sos')

    def _program(self) -> Any:
        from fluxopt.math import program
        from fluxopt.math.sources import _features

        return program(_features(self.ports, self.converters, self.storages))

    def sources(self, profiles: Mapping[str, pl.DataFrame] | None = None) -> dict[str, Any]:
        """The numbers [`spec`][fluxopt.FlowSystem.spec] is bound to, one table per declared name.

        Every parameter, relation and dimension the spec declares, keyed by
        the timestamps, years and ids the elements were written with. Edit a
        table, or add one for a declaration of your own, and hand the dict to
        `specsolve.solve`: the spec's ``assumptions:`` check whatever
        arrives.

        Args:
            profiles: Mapping from ``ProfileRef.table`` to a ``pl.DataFrame``
                holding the referenced columns, keyed by its ``time`` (and
                ``period``) columns. Required if the system uses any
                ``ProfileRef`` — see [`required_profiles`][fluxopt.FlowSystem.required_profiles].

        Raises:
            KeyError: If any ``ProfileRef`` cannot be resolved from *profiles*;
                lists every unresolvable ref with its element/field path.
        """
        from fluxopt.math import build_sources
        from fluxopt.math.sources import _bind

        refs: list[tuple[str, ProfileRef]] = []
        for group in (self.carriers, self.effects, self.ports, self.converters, self.storages):
            _collect_profile_refs(group, '', refs)
        _check_profiles_cover(refs, profiles or {})
        # Resolved on a copy, so the system stays reusable across profiles.
        groups = (self.carriers, self.effects, self.ports, self.converters, self.storages)
        if refs:
            groups = copy.deepcopy(groups)
            for group in groups:
                _resolve_refs(group, profiles or {})
        carriers, effects, ports, converters, storages = groups
        tables = build_sources(
            timesteps=self.timesteps,
            carriers=carriers,
            effects=effects,
            ports=ports,
            objective=self.objective,
            converters=converters,
            storages=storages,
            dt=self.dt,
            periods=self.periods,
            period_weights=self.period_weights,
        )
        return _bind(tables, self._program())

    def optimize(
        self,
        profiles: Mapping[str, pl.DataFrame] | None = None,
        *,
        solver: str = 'highs',
        archive: str | Path | None = None,
        **solver_options: Any,
    ) -> specsolve.Result:
        """Solve [`spec`][fluxopt.FlowSystem.spec] against [`sources`][fluxopt.FlowSystem.sources].

        The same as ``specsolve.solve(system.spec(), system.sources(profiles))``.
        A solver other than HiGHS gets the piecewise sets as special-ordered
        sets, which it takes natively.

        Args:
            profiles: Mapping from ``ProfileRef.table`` to a ``pl.DataFrame``
                holding the referenced columns, keyed by its ``time`` (and
                ``period``) columns. Required if the system uses any
                ``ProfileRef``.
            solver: Solver name — ``highs``, or ``gurobi`` with specsolve's extra.
            archive: Where specsolve writes the spec, its data and the answer,
                as a ``.zip`` or a directory; ``specsolve.load_archive`` reads
                it back.
            **solver_options: Passed to the solver verbatim, in its own vocabulary.

        Returns:
            specsolve's result: ``objective``, ``primal(name)`` for a
            variable and ``evaluate(name)`` for a reported expression such as
            ``flow_hours`` or ``priced_flow_hour``, each a tidy polars table.
        """
        import specsolve

        return specsolve.solve(
            self.spec() if solver == 'highs' else self._program(),
            self.sources(profiles),
            solver,
            solver_options=solver_options or None,
            archive=archive,
        )
