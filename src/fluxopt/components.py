from __future__ import annotations

from typing import Any, override

from pydantic import Field, PrivateAttr

from fluxopt.elements import Element, Flow, PiecewiseConversion, qualified_id
from fluxopt.types import IdList, Variate


def _qualify_flows(component_id: str, flows: list[Flow]) -> IdList[Flow]:
    """Set qualified id on each flow and return as IdList.

    Args:
        component_id: Parent component id used as prefix.
        flows: Flows to qualify.
    """
    for f in flows:
        f.id = qualified_id(component_id, f.id)
    return IdList(flows)


class Port(Element):
    """System boundary that imports from or exports to buses."""

    id: str
    imports: list[Flow] | IdList[Flow] = Field(default_factory=list)
    exports: list[Flow] | IdList[Flow] = Field(default_factory=list)

    @override
    def model_post_init(self, __context: Any) -> None:
        """Qualify flow ids with the port id."""
        self.imports = _qualify_flows(self.id, list(self.imports))
        self.exports = _qualify_flows(self.id, list(self.exports))


class Converter(Element):
    """Conversion between input and output flows.

    Two mutually exclusive modes:

    - **Linear** — ``conversion_factors=[{flow_short_id: a_f}, ...]``,
      one dict per equation; constraint ``sum_f(a_f * P_{f,t}) = 0``.
    - **Piecewise** — ``conversion=PiecewiseConversion(...)``; the solver
      interpolates between breakpoints, optionally with on/off via
      ``PiecewiseConversion.status``.
    """

    id: str
    """Converter id."""
    inputs: list[Flow] | IdList[Flow]
    """Input flows."""
    outputs: list[Flow] | IdList[Flow]
    """Output flows."""
    conversion_factors: list[dict[str, Variate]] = Field(default_factory=list)  # a_f
    """Linear-mode equations. Empty when
    ``conversion`` is set.
    """
    conversion: PiecewiseConversion | None = None
    """Piecewise-mode curve. ``None`` for linear mode."""
    _short_to_id: dict[str, str] = PrivateAttr(default_factory=dict)

    @override
    def model_post_init(self, __context: Any) -> None:
        """Qualify flow ids and validate mode exclusivity."""
        self.inputs = _qualify_flows(self.id, list(self.inputs))
        self.outputs = _qualify_flows(self.id, list(self.outputs))
        self._short_to_id = {f.short_id: f.id for f in (*self.inputs, *self.outputs)}

        if self.conversion is not None:
            if self.conversion_factors:
                msg = (
                    f'Converter {self.id!r}: cannot set both conversion_factors and conversion '
                    f'(they are mutually exclusive linear vs piecewise modes)'
                )
                raise ValueError(msg)
            curve_flows = {flow for flow, _, _ in self.conversion._iter_normalized()}
            unknown = curve_flows - set(self._short_to_id)
            if unknown:
                msg = (
                    f'Converter {self.id!r}: PiecewiseConversion references unknown flow short_ids '
                    f'{sorted(unknown)}; known: {sorted(self._short_to_id)}'
                )
                raise ValueError(msg)
            if self.conversion.status is not None:
                for f in (*self.inputs, *self.outputs):
                    if f.status is not None:
                        msg = (
                            f'Converter {self.id!r}: flow {f.short_id!r} cannot have flow-level '
                            f'status when PiecewiseConversion.status is set'
                        )
                        raise ValueError(msg)
        else:
            for eq_i, equation in enumerate(self.conversion_factors):
                unknown = set(equation) - set(self._short_to_id)
                if unknown:
                    msg = (
                        f'Converter {self.id!r}: conversion_factors[{eq_i}] references unknown '
                        f'flow short_ids {sorted(unknown)}; known: {sorted(self._short_to_id)}'
                    )
                    raise ValueError(msg)

    @classmethod
    def _single_io(cls, id: str, coefficient: Variate, input_flow: Flow, output_flow: Flow) -> Converter:
        """Create a single-input/single-output converter: input * coefficient = output."""
        return cls(
            id=id,
            inputs=[input_flow],
            outputs=[output_flow],
            conversion_factors=[{input_flow.short_id: coefficient, output_flow.short_id: -1}],
        )

    @classmethod
    def boiler(cls, id: str, thermal_efficiency: Variate, fuel_flow: Flow, thermal_flow: Flow) -> Converter:
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
        cop: Variate,
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
            id=id,
            inputs=[electrical_flow, source_flow],
            outputs=[thermal_flow],
            conversion_factors=[
                {electrical_flow.short_id: cop, thermal_flow.short_id: -1},
                {electrical_flow.short_id: 1, source_flow.short_id: 1, thermal_flow.short_id: -1},
            ],
        )

    @classmethod
    def power2heat(cls, id: str, efficiency: Variate, electrical_flow: Flow, thermal_flow: Flow) -> Converter:
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
        eta_el: Variate,
        eta_th: Variate,
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
            id=id,
            inputs=[fuel_flow],
            outputs=[electrical_flow, thermal_flow],
            conversion_factors=[
                {fuel_flow.short_id: eta_el, electrical_flow.short_id: -1},
                {fuel_flow.short_id: eta_th, thermal_flow.short_id: -1},
            ],
        )
