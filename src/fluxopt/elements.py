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
    effects_per_size: dict[str, float] = field(default_factory=dict)
    effects_fixed: dict[str, float] = field(default_factory=dict)


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


@dataclass(eq=False)
class Flow:
    """A single energy flow on a carrier.

    ``id`` is optional: defaults to the carrier name. Set explicitly to
    disambiguate multiple flows on the same carrier::

        Flow('elec')  # id → 'elec' → boiler(elec)
        Flow('elec', id='base')  # id → 'base' → chp(base)

    After ``__post_init__`` of the parent component, ``id`` is expanded
    to the qualified form ``component(id)``.
    """

    carrier: str
    id: str = ''
    node: str | None = None
    size: float | Sizing | None = None  # P̄_f  [MW]
    relative_minimum: TimeSeries = 0.0  # p̲_f  [-]
    relative_maximum: TimeSeries = 1.0  # p̄_f  [-]
    fixed_relative_profile: TimeSeries | None = None  # π_f  [-]
    effects_per_flow_hour: dict[str, TimeSeries] = field(default_factory=dict)  # c_{f,k}  [varies]
    status: Status | None = None
    prior_rates: list[float] | None = None  # flow rates before horizon [MW]

    def __post_init__(self) -> None:
        """Default id to carrier name, including node if set."""
        if not self.id:
            self.id = node_id(self.carrier, self.node) if self.node else self.carrier
        if self.status is not None and isinstance(self.relative_minimum, (int, float)) and self.relative_minimum <= 0:
            msg = (
                f'Flow {self.id!r}: relative_minimum must be > 0 when status is set, '
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
    contribution_from: dict[str, float] = field(default_factory=dict)
    contribution_from_per_hour: dict[str, TimeSeries] = field(default_factory=dict)


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
    capacity: float | Sizing | None = None  # Ē_s  [MWh]
    eta_charge: TimeSeries = 1.0  # η^c_s  [-]
    eta_discharge: TimeSeries = 1.0  # η^d_s  [-]
    relative_loss_per_hour: TimeSeries = 0.0  # δ_s  [1/h]
    prior_level: float | None = None  # E_{s,0}  [MWh]
    cyclic: bool = True  # E_{s,first} == E_{s,last}
    relative_minimum_level: TimeSeries = 0.0  # e̲_s  [-]
    relative_maximum_level: TimeSeries = 1.0  # ē_s  [-]

    def __post_init__(self) -> None:
        """Validate carrier match, rename colliding flow ids, and qualify."""
        if self.charging.carrier != self.discharging.carrier:
            msg = (
                f'Storage {self.id!r}: charging carrier {self.charging.carrier!r} '
                f'!= discharging carrier {self.discharging.carrier!r}'
            )
            raise ValueError(msg)
        if self.charging.id == self.discharging.id:
            self.charging.id = 'charge'
            self.discharging.id = 'discharge'
        self.charging.id = qualified_id(self.id, self.charging.id)
        self.discharging.id = qualified_id(self.id, self.discharging.id)
