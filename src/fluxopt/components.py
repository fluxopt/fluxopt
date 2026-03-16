from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fluxopt.elements import qualified_id
from fluxopt.types import IdList

if TYPE_CHECKING:
    from fluxopt.elements import ConversionCurve, Flow
    from fluxopt.types import TimeSeries


def _qualify_flows(component_id: str, flows: list[Flow]) -> IdList[Flow]:
    """Set qualified id on each flow and return as IdList.

    Args:
        component_id: Parent component id used as prefix.
        flows: Flows to qualify.
    """
    for f in flows:
        f.id = qualified_id(component_id, f.id)
    return IdList(flows)


@dataclass
class Port:
    """System boundary that imports from or exports to buses."""

    id: str
    imports: list[Flow] | IdList[Flow] = field(default_factory=list)
    exports: list[Flow] | IdList[Flow] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Qualify flow ids with the port id."""
        self.imports = _qualify_flows(self.id, list(self.imports))
        self.exports = _qualify_flows(self.id, list(self.exports))


@dataclass
class Converter:
    """Conversion between input and output flows.

    Supports two modes:

    **Linear** (default): ``conversion_factors`` is a list of dicts, each
    defining one equation ``sum_f(a_f * P_{f,t}) = 0``.

    **Piecewise**: set ``conversion`` to a :class:`ConversionCurve`.
    Breakpoint keys must match flow ``short_id`` values. Individual flows
    must not carry ``size`` or ``status`` (the curve governs sizing/status
    at the component level).
    """

    id: str
    inputs: list[Flow] | IdList[Flow]
    outputs: list[Flow] | IdList[Flow]
    conversion_factors: list[dict[str, TimeSeries]] = field(default_factory=list)  # a_f
    conversion: ConversionCurve | None = None
    _short_to_id: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Qualify flow ids, build short→qualified mapping, validate."""
        self.inputs = _qualify_flows(self.id, list(self.inputs))
        self.outputs = _qualify_flows(self.id, list(self.outputs))
        self._short_to_id = {f.short_id: f.id for f in (*self.inputs, *self.outputs)}

        if self.conversion is not None:
            if self.conversion_factors:
                msg = f'Converter {self.id!r}: cannot specify both conversion_factors and conversion'
                raise ValueError(msg)
            # Validate breakpoint keys match flow short_ids
            bp_keys = set(self.conversion.breakpoints.keys())
            flow_keys = set(self._short_to_id.keys())
            if not bp_keys.issubset(flow_keys):
                unknown = bp_keys - flow_keys
                msg = f'Converter {self.id!r}: ConversionCurve breakpoint keys {unknown} do not match flow short_ids {flow_keys}'
                raise ValueError(msg)
            # Validate no flow-level size or status on piecewise flows
            from fluxopt.elements import Investment, PiecewiseInvestment, PiecewiseSizing, Sizing

            for f in (*self.inputs, *self.outputs):
                if f.short_id in bp_keys:
                    if isinstance(f.size, (Sizing, Investment, PiecewiseSizing, PiecewiseInvestment)):
                        msg = f'Converter {self.id!r}: flow {f.short_id!r} cannot have Sizing/Investment when using ConversionCurve'
                        raise ValueError(msg)
                    if f.status is not None:
                        msg = (
                            f'Converter {self.id!r}: flow {f.short_id!r} cannot have status when using ConversionCurve'
                        )
                        raise ValueError(msg)

    @classmethod
    def _single_io(cls, id: str, coefficient: TimeSeries, input_flow: Flow, output_flow: Flow) -> Converter:
        """Create a single-input/single-output converter: input * coefficient = output."""
        return cls(
            id,
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.short_id: coefficient, output_flow.short_id: -1}],
        )

    @classmethod
    def boiler(cls, id: str, thermal_efficiency: TimeSeries, fuel_flow: Flow, thermal_flow: Flow) -> Converter:
        """Create a boiler converter: fuel * eta = thermal.

        Args:
            id: Converter id.
            thermal_efficiency: Thermal efficiency eta.
            fuel_flow: Input fuel flow.
            thermal_flow: Output thermal flow.
        """
        return cls._single_io(id, thermal_efficiency, fuel_flow, thermal_flow)

    @classmethod
    def heat_pump(
        cls,
        id: str,
        cop: TimeSeries,
        electrical_flow: Flow,
        source_flow: Flow,
        thermal_flow: Flow,
    ) -> Converter:
        """Create a heat pump converter with source heat.

        Two conversion equations:
            electrical * COP = thermal
            electrical + source = thermal

        Args:
            id: Converter id.
            cop: Coefficient of performance.
            electrical_flow: Input electrical flow.
            source_flow: Input environmental heat flow (air, ground, water).
            thermal_flow: Output thermal flow.
        """
        return cls(
            id,
            inputs=[electrical_flow, source_flow],
            outputs=[thermal_flow],
            conversion_factors=[
                {electrical_flow.short_id: cop, thermal_flow.short_id: -1},
                {electrical_flow.short_id: 1, source_flow.short_id: 1, thermal_flow.short_id: -1},
            ],
        )

    @classmethod
    def power2heat(cls, id: str, efficiency: TimeSeries, electrical_flow: Flow, thermal_flow: Flow) -> Converter:
        """Create an electric resistance heater: electrical * eta = thermal.

        Args:
            id: Converter id.
            efficiency: Electrical-to-thermal efficiency.
            electrical_flow: Input electrical flow.
            thermal_flow: Output thermal flow.
        """
        return cls._single_io(id, efficiency, electrical_flow, thermal_flow)

    @classmethod
    def chp(
        cls,
        id: str,
        eta_el: TimeSeries,
        eta_th: TimeSeries,
        fuel_flow: Flow,
        electrical_flow: Flow,
        thermal_flow: Flow,
    ) -> Converter:
        """Create a CHP converter with separate electrical and thermal outputs.

        Args:
            id: Converter id.
            eta_el: Electrical efficiency.
            eta_th: Thermal efficiency.
            fuel_flow: Input fuel flow.
            electrical_flow: Output electrical flow.
            thermal_flow: Output thermal flow.
        """
        return cls(
            id,
            inputs=[fuel_flow],
            outputs=[electrical_flow, thermal_flow],
            conversion_factors=[
                {fuel_flow.short_id: eta_el, electrical_flow.short_id: -1},
                {fuel_flow.short_id: eta_th, thermal_flow.short_id: -1},
            ],
        )
