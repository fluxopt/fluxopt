from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from fluxopt.types import PiecewiseMethod, Variate

PENALTY_EFFECT_ID = 'penalty'

NODE_SEP = ':'
QUAL_FMT = '{component}({flow})'


def qualified_id(component: str, flow: str) -> str:
    """Format a qualified flow id: ``component(flow)``."""
    return QUAL_FMT.format(component=component, flow=flow)


def node_id(carrier: str, node: str) -> str:
    """Format a carrier-node id: ``carrier:node``."""
    return f'{carrier}{NODE_SEP}{node}'


@dataclass
class Carrier:
    """Physical energy medium (electricity, heat, gas, …).

    Args:
        id: Unique identifier used as xarray coordinate.
        nodes: Sub-nodes for multi-node balancing. Empty means single-node.
        unit: Energy unit label.
        color: Optional color for plotting.
        description: Human-readable description.
    """

    id: str
    nodes: list[str] = field(default_factory=list)
    unit: str = 'MWh'
    color: str | None = None
    description: str = ''


@dataclass
class Sizing:
    """Capacity optimization parameters.

    The solver decides the optimal size within [size_min, size_max].

    - ``mandatory=True``: continuous, size in [min, max], no binary.
    - ``mandatory=False``: binary indicator gates size: 0 or [min, max].
    - ``size_min == size_max`` with ``mandatory=False``: binary invest
      at exact size (yes/no).

    See: docs/math/sizing.md

    Args:
        size_min: Minimum capacity if invested.
        size_max: Maximum capacity.
        mandatory: If True, must be built (no binary indicator).
        effects_per_size: Effect cost per unit size (e.g. €/MW).
        effects_fixed: Fixed effect cost if built (optional only).
    """

    size_min: float
    size_max: float
    mandatory: bool = True
    effects_per_size: dict[str, Variate] = field(default_factory=dict)
    effects_fixed: dict[str, Variate] = field(default_factory=dict)


@dataclass
class Investment:
    """Singular discrete build-timing optimization.

    The solver decides WHEN to build (which period) and at what size.
    Once built, capacity is available for ``lifetime`` periods.
    Size is decided once — no growth or partial retirement.

    Args:
        size_min: Minimum capacity if built.
        size_max: Maximum capacity.
        mandatory: If True, must build exactly once; if False, may build at most once.
        lifetime: Periods active after build; None = forever.
        prior_size: Pre-existing capacity available from period 0.
        effects_per_size_at_build: One-time per-MW costs charged in the build period.
        effects_fixed_at_build: One-time fixed costs charged in the build period.
        effects_per_size_recurring: Recurring per-MW costs charged every active period.
        effects_fixed_recurring: Recurring fixed costs charged every active period.
    """

    size_min: float
    size_max: float
    mandatory: bool = True
    lifetime: int | None = None
    prior_size: float = 0.0
    effects_per_size_at_build: dict[str, Variate] = field(default_factory=dict)
    effects_fixed_at_build: dict[str, Variate] = field(default_factory=dict)
    effects_per_size_recurring: dict[str, Variate] = field(default_factory=dict)
    effects_fixed_recurring: dict[str, Variate] = field(default_factory=dict)


@dataclass
class Status:
    """Binary on/off behavior parameters.

    Together with relative bounds, gives semi-continuous behavior:
    ``{0} U [min, max] * size``.

    See: docs/math/status.md

    Args:
        uptime_min: Minimum consecutive on-hours.
        uptime_max: Maximum consecutive on-hours.
        downtime_min: Minimum consecutive off-hours.
        downtime_max: Maximum consecutive off-hours.
        effects_per_running_hour: Effect cost per running hour.
        effects_per_startup: Effect cost per startup event.
    """

    uptime_min: float | None = None  # [h]
    uptime_max: float | None = None  # [h]
    downtime_min: float | None = None  # [h]
    downtime_max: float | None = None  # [h]
    effects_per_running_hour: dict[str, Variate] = field(default_factory=dict)
    effects_per_startup: dict[str, Variate] = field(default_factory=dict)


@dataclass(eq=False)
class Flow:
    """A single energy flow on a carrier.

    ``short_id`` defaults to ``carrier`` (or ``carrier:node`` when ``node``
    is set).  Set explicitly to disambiguate multiple flows on the same
    carrier::

        Flow('elec')  # short_id='elec'
        Flow('heat', node='A')  # short_id='heat:A'
        Flow('elec', short_id='base')  # short_id='base'

    ``short_id`` must be unique within a component.  Storage renames
    colliding short_ids to ``charge`` / ``discharge`` before qualification.
    ``id`` is the qualified form set by the parent component:
    ``component(short_id)``.

    See: docs/math/flows.md

    Args:
        carrier: Carrier this flow connects to.
        short_id: Component-local identifier; defaults to ``carrier``
            (or ``carrier:node``). The qualified form ``component(short_id)``
            is stored in ``id``.
        node: Sub-node for multi-node carrier balancing.
        size: Nominal capacity [MW], ``Sizing`` for investment optimization,
            or None (unsized / unbounded).
        relative_rate_min: Lower bound as fraction of size.
        relative_rate_max: Upper bound as fraction of size.
        fixed_relative_profile: Fixed profile as fraction of size; sets both
            lower and upper bounds equal to the profile value.
        effects_per_flow_hour: Effect coefficients per flow-hour
            (e.g. €/MWh).
        status: On/off behavior (semi-continuous, startup costs, durations).
        prior_rates: Flow rates [MW] before the horizon, used for
            status initial conditions.
    """

    carrier: str
    short_id: str = ''
    id: str = field(init=False, default='')
    node: str | None = None
    size: float | Sizing | Investment | None = None  # P̄_f  [MW]
    relative_rate_min: Variate = 0.0  # p̲_f  [-]
    relative_rate_max: Variate = 1.0  # p̄_f  [-]
    fixed_relative_profile: Variate | None = None  # π_f  [-]
    effects_per_flow_hour: dict[str, Variate] = field(default_factory=dict)  # c_{f,k}  [varies]
    status: Status | None = None
    prior_rates: list[float] | None = None  # flow rates before horizon [MW]

    def __post_init__(self) -> None:
        """Default short_id from carrier/node, set id = short_id."""
        if not self.short_id:
            self.short_id = node_id(self.carrier, self.node) if self.node else self.carrier
        self.id = self.short_id
        if self.status is not None and isinstance(self.relative_rate_min, (int, float)) and self.relative_rate_min <= 0:
            msg = (
                f'Flow {self.short_id!r}: relative_rate_min must be > 0 when status is set, '
                f'otherwise on/off is indistinguishable (got {self.relative_rate_min})'
            )
            raise ValueError(msg)


@dataclass
class Effect:
    """A tracked quantity across the optimization horizon (cost, CO₂, …).

    One effect is designated as the objective to minimize via the
    ``objective_effects`` argument of ``optimize()``. Others can be bounded
    to enforce budgets (e.g. emission caps).

    Effects accumulate contributions from two domains:

    - **Temporal** — per-timestep flow costs, running costs, startup costs.
    - **Lump** — one-time sizing costs and fixed costs.

    Cross-effect chains (e.g. CO₂ → cost) are supported via
    ``contribution_from``.

    See: docs/math/effects.md

    Args:
        id: Unique identifier.
        unit: Unit label (e.g. ``'€'``, ``'kg'``).
        total_max: Upper bound on weighted total across all periods.
        total_min: Lower bound on weighted total across all periods.
        periodic_max: Upper bound applied to each period independently.
            Scalar or per-period values (multi-period only).
        periodic_min: Lower bound applied to each period independently.
            Scalar or per-period values (multi-period only).
        rate_max: Upper bound rate [unit/h], scaled by Δt.
        rate_min: Lower bound rate [unit/h], scaled by Δt.
        contribution_from: Cross-effect factors ``{source_effect: factor}``.
            Scalar factors apply identically to both domains; time-varying
            factors are averaged for the lump domain.
        period_weights: Per-period weights ω for total aggregation;
            overrides global ``period_weights``.
    """

    id: str
    unit: str = ''
    total_max: float | None = None  # Φ̄_k  [unit] — weighted total across all periods
    total_min: float | None = None  # Φ̲_k  [unit] — weighted total across all periods
    periodic_max: Variate | None = None  # Φ̄_{k,p}  [unit] — each period independently
    periodic_min: Variate | None = None  # Φ̲_{k,p}  [unit] — each period independently
    rate_max: Variate | None = None  # Φ̄_{k,t}  [unit/h] — rate, scaled by dt
    rate_min: Variate | None = None  # Φ̲_{k,t}  [unit/h] — rate, scaled by dt
    contribution_from: dict[str, Variate] = field(default_factory=dict)
    period_weights: list[float] | None = None  # ω[p] — scales total across periods


@dataclass
class Storage:
    """Energy storage with level dynamics.

    Flow ids are qualified as ``storage(flow)``. When both flows connect
    to the same carrier, they are renamed to ``charge`` / ``discharge``::

        Storage('bat', Flow('elec'), Flow('elec'))  # bat(charge), bat(discharge)
        Storage('bat', Flow('elec'), Flow('heat'))  # bat(elec), bat(heat)

    Level balance::

        E_{s,t+1} = E_{s,t} (1 - δ)^Δt + P^c η^c Δt - P^d / η^d Δt

    See: docs/math/storage.md

    Args:
        id: Storage identifier.
        charging: Charging flow.
        discharging: Discharging flow.
        capacity: Maximum stored energy [MWh], ``Sizing`` for investment
            optimization, or None.
        eta_charge: Charging efficiency.
        eta_discharge: Discharging efficiency.
        relative_loss_per_hour: Self-discharge rate [1/h].
        prior_level: Initial energy level [MWh]; None = unconstrained.
        cyclic: If True, end level must equal start level.
        relative_level_min: Min SOC as fraction of capacity.
        relative_level_max: Max SOC as fraction of capacity.
        status: Component-level on/off behavior gating both charging and
            discharging. Forbids flow-level ``status`` on the child flows
            (the two switches would have no defined precedence).
    """

    id: str
    charging: Flow
    discharging: Flow
    capacity: float | Sizing | Investment | None = None  # Ē_s  [MWh]
    eta_charge: Variate = 1.0  # η^c_s  [-]
    eta_discharge: Variate = 1.0  # η^d_s  [-]
    relative_loss_per_hour: Variate = 0.0  # δ_s  [1/h]
    prior_level: float | None = None  # E_{s,0}  [MWh]
    cyclic: bool = True  # E_{s,first} == E_{s,last}
    relative_level_min: Variate = 0.0  # e̲_s  [-]
    relative_level_max: Variate = 1.0  # ē_s  [-]
    status: Status | None = None

    def __post_init__(self) -> None:
        """Validate carrier match, rename colliding flow ids, and qualify."""
        if self.charging.carrier != self.discharging.carrier:
            msg = (
                f'Storage {self.id!r}: charging carrier {self.charging.carrier!r} '
                f'!= discharging carrier {self.discharging.carrier!r}'
            )
            raise ValueError(msg)
        if self.charging.short_id == self.discharging.short_id:
            self.charging.short_id = 'charge'
            self.discharging.short_id = 'discharge'
        self.charging.id = qualified_id(self.id, self.charging.short_id)
        self.discharging.id = qualified_id(self.id, self.discharging.short_id)
        if self.status is not None:
            for f in (self.charging, self.discharging):
                if f.status is not None:
                    msg = (
                        f'Storage {self.id!r}: flow {f.short_id!r} cannot have flow-level '
                        f'status when Storage.status is set; the component status already gates both flows'
                    )
                    raise ValueError(msg)
                if f.size is None:
                    msg = (
                        f'Storage {self.id!r}: flow {f.short_id!r} must have a size when '
                        f'Storage.status is set — without it, the on/off binary cannot gate '
                        f'the rate (no upper bound to scale)'
                    )
                    raise ValueError(msg)


_CurveTuple = tuple[str, 'list[Variate]'] | tuple[str, 'list[Variate]', Literal['==', '<=', '>=']]


@dataclass
class PiecewiseConversion:
    """Piecewise-linear conversion linking N flows.

    Wraps :func:`linopy.piecewise.add_piecewise_formulation`. All flows
    share interpolation weights — every operating point lies on the same
    piece of the curve.

    Two input forms:

    - **Dict** — equality-only, terse for the common case::

          PiecewiseConversion({'fuel': [0, 50, 100], 'Heat': [0, 45, 70]})

    - **List of tuples** — supports per-flow inequality bounds::

          PiecewiseConversion(
              [
                  ('fuel', [0, 50, 100]),
                  ('Heat', [0, 45, 70], '>='),
              ]
          )

    See: docs/math/converters.md

    Args:
        points: Per-flow breakpoints. Either ``{flow: [bp...]}`` (equality
            only) or a list of ``(flow, [bp...])`` / ``(flow, [bp...], '<='|'>=')``
            tuples. Need >=2 flows; all breakpoint lists must share the same
            length (>=2). At most one tuple may carry a non-equality bound,
            and only when exactly two flows are present.
        method: Formulation. ``"auto"`` picks LP (2 flows + bounded +
            matching convexity), else incremental (monotonic) or sos2.
            Override with ``"sos2"`` / ``"incremental"`` / ``"lp"``.
        status: Component-level on/off behavior gating the curve.
        availability: Time-varying scaling of the upper breakpoint.
    """

    points: dict[str, list[Variate]] | list[_CurveTuple]
    method: PiecewiseMethod = 'auto'
    status: Status | None = None
    availability: Variate = 1.0

    def __post_init__(self) -> None:
        """Validate normalized breakpoints and bound combinations."""
        flows_pts_bounds = list(self._iter_normalized())

        if len(flows_pts_bounds) < 2:
            msg = f'PiecewiseConversion needs >=2 flows, got {len(flows_pts_bounds)}'
            raise ValueError(msg)

        n = len(flows_pts_bounds[0][1])
        if n < 2:
            msg = f'PiecewiseConversion needs >=2 breakpoints per flow, got {n}'
            raise ValueError(msg)

        if any(len(pts) != n for _, pts, _ in flows_pts_bounds):
            lengths = {flow: len(pts) for flow, pts, _ in flows_pts_bounds}
            msg = f'PiecewiseConversion breakpoint lists must all have the same length, got {lengths}'
            raise ValueError(msg)

        flows = [flow for flow, _, _ in flows_pts_bounds]
        if len(set(flows)) != len(flows):
            dupes = [f for f in flows if flows.count(f) > 1]
            msg = f'PiecewiseConversion has duplicate flow ids: {sorted(set(dupes))}'
            raise ValueError(msg)

        nonequal = [b for _, _, b in flows_pts_bounds if b != '==']
        if len(nonequal) > 1:
            msg = f'At most one bounded flow per PiecewiseConversion, got {len(nonequal)}'
            raise ValueError(msg)
        if nonequal and len(flows_pts_bounds) > 2:
            msg = f'Inequality bounds require exactly 2 flows, got {len(flows_pts_bounds)}'
            raise ValueError(msg)
        if self.method == 'lp' and not nonequal:
            msg = "method='lp' requires one flow with bound '<=' or '>='"
            raise ValueError(msg)

    def _iter_normalized(
        self,
    ) -> list[tuple[str, list[Variate], Literal['==', '<=', '>=']]]:
        """Return the curve as a list of ``(flow, breakpoints, bound)`` tuples."""
        if isinstance(self.points, dict):
            return [(flow, list(pts), '==') for flow, pts in self.points.items()]
        result: list[tuple[str, list[Variate], Literal['==', '<=', '>=']]] = []
        for i, t in enumerate(self.points):
            match t:
                case (flow, pts):
                    result.append((flow, list(pts), '=='))
                case (flow, pts, bound):
                    if bound not in ('==', '<=', '>='):
                        msg = f'PiecewiseConversion tuple {i} has invalid bound {bound!r}; expected one of (==, <=, >=)'
                        raise ValueError(msg)
                    result.append((flow, list(pts), bound))
                case _:
                    msg = f'PiecewiseConversion tuple {i} has length {len(t)}; expected 2 or 3'
                    raise ValueError(msg)
        return result
