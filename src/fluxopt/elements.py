from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluxopt.types import TimeSeries

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

    The solver decides the optimal size within [min_size, max_size].

    - ``mandatory=True``: continuous, size in [min, max], no binary.
    - ``mandatory=False``: binary indicator gates size: 0 or [min, max].
    - ``min_size == max_size``: binary invest at exact size (yes/no).
    """

    min_size: float
    max_size: float
    mandatory: bool = True
    effects_per_size: dict[str, TimeSeries] = field(default_factory=dict)
    effects_fixed: dict[str, TimeSeries] = field(default_factory=dict)


@dataclass
class Investment:
    """Singular discrete build-timing optimization.

    The solver decides WHEN to build (which period) and at what size.
    Once built, capacity is available for ``lifetime`` periods.
    Size is decided once — no growth or partial retirement.

    Args:
        min_size: Minimum capacity if built.
        max_size: Maximum capacity.
        mandatory: If True, must build exactly once; if False, may build at most once.
        lifetime: Periods active after build; None = forever.
        prior_size: Pre-existing capacity available from period 0.
        effects_per_size: One-time per-MW costs charged in the build period.
        effects_fixed: One-time fixed costs charged in the build period.
        effects_per_size_periodic: Recurring per-MW costs charged every active period.
        effects_fixed_periodic: Recurring fixed costs charged every active period.
    """

    min_size: float
    max_size: float
    mandatory: bool = True
    lifetime: int | None = None
    prior_size: float = 0.0
    effects_per_size: dict[str, TimeSeries] = field(default_factory=dict)
    effects_fixed: dict[str, TimeSeries] = field(default_factory=dict)
    effects_per_size_periodic: dict[str, TimeSeries] = field(default_factory=dict)
    effects_fixed_periodic: dict[str, TimeSeries] = field(default_factory=dict)


@dataclass
class Status:
    """Binary on/off behavior parameters.

    Together with relative bounds, gives semi-continuous behavior:
    ``{0} U [min, max] * size``.
    """

    min_uptime: float | None = None  # [h]
    max_uptime: float | None = None  # [h]
    min_downtime: float | None = None  # [h]
    max_downtime: float | None = None  # [h]
    effects_per_running_hour: dict[str, TimeSeries] = field(default_factory=dict)
    effects_per_startup: dict[str, TimeSeries] = field(default_factory=dict)


@dataclass
class PiecewiseSizing:
    """Sizing for piecewise converters. Size = max breakpoint (implicit).

    Args:
        mandatory: When True, must operate. When False, a silent binary
            allows the converter to be completely off.
        effects_per_size: Per-MW costs charged every period ``{effect_id: value}``.
        effects_fixed: Fixed costs charged every period ``{effect_id: value}``.
    """

    mandatory: bool = True
    effects_per_size: dict[str, TimeSeries] = field(default_factory=dict)
    effects_fixed: dict[str, TimeSeries] = field(default_factory=dict)


@dataclass
class PiecewiseInvestment:
    """Build-timing for piecewise converters. Size = max breakpoint (implicit).

    The solver decides WHEN to build (which period). Once built, the
    converter is available for ``lifetime`` periods at its implicit size.

    Args:
        mandatory: If True, must build exactly once; if False, may build at most once.
        lifetime: Periods active after build; None = forever.
        prior_size: Pre-existing capacity available from period 0.
        effects_per_size: One-time per-MW costs charged in the build period.
        effects_fixed: One-time fixed costs charged in the build period.
        effects_per_size_periodic: Recurring per-MW costs charged every active period.
        effects_fixed_periodic: Recurring fixed costs charged every active period.
    """

    mandatory: bool = True
    lifetime: int | None = None
    prior_size: float = 0.0
    effects_per_size: dict[str, TimeSeries] = field(default_factory=dict)
    effects_fixed: dict[str, TimeSeries] = field(default_factory=dict)
    effects_per_size_periodic: dict[str, TimeSeries] = field(default_factory=dict)
    effects_fixed_periodic: dict[str, TimeSeries] = field(default_factory=dict)


@dataclass
class ConversionCurve:
    """Piecewise-linear conversion defined by breakpoints.

    Each key in ``breakpoints`` maps a flow short_id to a list of operating
    points.  All lists must have the same length (>= 2).  The solver picks
    the optimal operating point, interpolating between adjacent breakpoints.

    Args:
        breakpoints: ``{flow_short_id: [bp0, bp1, …]}`` per flow.
        size: Sizing/investment for the piecewise converter. ``None`` means
            mandatory operation, ``float`` sets an explicit scale,
            ``PiecewiseSizing`` adds effects and optional binary,
            ``PiecewiseInvestment`` enables build-timing optimization.
        status: Component-level on/off behavior. Governs all flows.
        availability: Maximum fraction of the reference flow's last
            breakpoint that can be dispatched each timestep.
    """

    breakpoints: dict[str, list[TimeSeries]]
    size: float | PiecewiseSizing | PiecewiseInvestment | None = None
    status: Status | None = None
    availability: TimeSeries = 1.0

    def __post_init__(self) -> None:
        """Validate breakpoint structure."""
        if len(self.breakpoints) < 2:
            msg = 'ConversionCurve requires breakpoints for at least 2 flows'
            raise ValueError(msg)
        lengths = {k: len(v) for k, v in self.breakpoints.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            msg = f'ConversionCurve breakpoint lists must all have the same length, got {lengths}'
            raise ValueError(msg)
        n = unique_lengths.pop()
        if n < 2:
            msg = f'ConversionCurve requires at least 2 breakpoints, got {n}'
            raise ValueError(msg)

    @property
    def mandatory(self) -> bool:
        """Whether the converter must always operate."""
        if self.size is None or isinstance(self.size, (int, float)):
            return True
        return self.size.mandatory


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
    """

    carrier: str
    short_id: str = ''
    id: str = field(init=False, default='')
    node: str | None = None
    size: float | Sizing | Investment | None = None  # P̄_f  [MW]
    relative_minimum: TimeSeries = 0.0  # p̲_f  [-]
    relative_maximum: TimeSeries = 1.0  # p̄_f  [-]
    fixed_relative_profile: TimeSeries | None = None  # π_f  [-]
    effects_per_flow_hour: dict[str, TimeSeries] = field(default_factory=dict)  # c_{f,k}  [varies]
    status: Status | None = None
    prior_rates: list[float] | None = None  # flow rates before horizon [MW]

    def __post_init__(self) -> None:
        """Default short_id from carrier/node, set id = short_id."""
        if not self.short_id:
            self.short_id = node_id(self.carrier, self.node) if self.node else self.carrier
        self.id = self.short_id
        if self.status is not None and isinstance(self.relative_minimum, (int, float)) and self.relative_minimum <= 0:
            msg = (
                f'Flow {self.short_id!r}: relative_minimum must be > 0 when status is set, '
                f'otherwise on/off is indistinguishable (got {self.relative_minimum})'
            )
            raise ValueError(msg)


@dataclass
class Effect:
    id: str
    unit: str = ''
    is_objective: bool = False
    maximum_total: float | None = None  # Φ̄_k  [unit]
    minimum_total: float | None = None  # Φ̲_k  [unit]
    maximum_per_hour: TimeSeries | None = None  # Φ̄_{k,t}  [unit]
    minimum_per_hour: TimeSeries | None = None  # Φ̲_{k,t}  [unit]
    contribution_from: dict[str, TimeSeries] = field(default_factory=dict)
    contribution_from_per_hour: dict[str, TimeSeries] = field(default_factory=dict)
    period_weights_periodic: list[float] | None = None  # ω_periodic[p] — scales temporal+periodic
    period_weights_once: list[float] | None = None  # ω_once[p] — scales once


@dataclass
class Storage:
    """Energy storage with level dynamics.

    Flow ids are qualified as ``storage(flow)``. When both flows connect
    to the same carrier, they are renamed to ``charge`` / ``discharge``::

        Storage('bat', Flow('elec'), Flow('elec'))  # bat(charge), bat(discharge)
        Storage('bat', Flow('elec'), Flow('heat'))  # bat(elec), bat(heat)

    Level balance::

        E_{s,t+1} = E_{s,t} (1 - δ)^Δt + P^c η^c Δt - P^d / η^d Δt
    """

    id: str
    charging: Flow
    discharging: Flow
    capacity: float | Sizing | Investment | None = None  # Ē_s  [MWh]
    eta_charge: TimeSeries = 1.0  # η^c_s  [-]
    eta_discharge: TimeSeries = 1.0  # η^d_s  [-]
    relative_loss_per_hour: TimeSeries = 0.0  # δ_s  [1/h]
    prior_level: float | None = None  # E_{s,0}  [MWh]
    cyclic: bool = True  # E_{s,first} == E_{s,last}
    relative_minimum_level: TimeSeries = 0.0  # e̲_s  [-]
    relative_maximum_level: TimeSeries = 1.0  # ē_s  [-]
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
                    msg = f'Storage {self.id!r}: flow {f.short_id!r} cannot have status when Storage.status is set'
                    raise ValueError(msg)
                if f.fixed_relative_profile is not None:
                    msg = f'Storage {self.id!r}: flow {f.short_id!r} cannot have fixed_relative_profile when Storage.status is set'
                    raise ValueError(msg)
