from __future__ import annotations

from collections import Counter
from typing import Any, override

from pydantic import Field

from fluxopt.elements import Element, Flow, PiecewiseConversion, _BoundFlow, qualified_id
from fluxopt.types import Variate


def _check_unique_short_ids(owner: str, flows: list[Flow]) -> None:
    """Raise when two flows of one component share a short_id.

    Args:
        owner: Component label for the error message (e.g. ``"Port 'grid'"``).
        flows: All flows of the component, across both directions.
    """
    if dupes := sorted(s for s, n in Counter(f.short_id for f in flows).items() if n > 1):
        msg = (
            f'{owner}: duplicate flow short_id(s) {dupes} — '
            f'set short_id explicitly to disambiguate flows on the same carrier'
        )
        raise ValueError(msg)


class Port(Element):
    """System boundary importing and exporting carriers.

    ``imports`` bring flow into the system (sources); ``exports`` send it out
    of the system (sinks).
    """

    id: str
    """Port id (prefixes qualified flow ids)."""
    imports: list[Flow] = Field(default_factory=list)
    """Flows bringing the carrier into the system (sources, e.g. grid supply)."""
    exports: list[Flow] = Field(default_factory=list)
    """Flows sending the carrier out of the system (sinks, e.g. demand)."""

    @override
    def model_post_init(self, __context: Any) -> None:
        """Reject duplicate short_ids across imports and exports."""
        _check_unique_short_ids(f'Port {self.id!r}', [*self.imports, *self.exports])

    def _qualified_flows(self) -> list[_BoundFlow]:
        """All flows with build-time qualified ids; imports produce, exports consume."""
        return [
            *(_BoundFlow(qualified_id(self.id, f.short_id), f, 1) for f in self.imports),
            *(_BoundFlow(qualified_id(self.id, f.short_id), f, -1) for f in self.exports),
        ]


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
    inputs: list[Flow]
    """Input flows."""
    outputs: list[Flow]
    """Output flows."""
    conversion_factors: list[dict[str, Variate]] = Field(default_factory=list)  # a_f
    """Linear-mode equations. Empty when
    ``conversion`` is set.
    """
    conversion: PiecewiseConversion | None = None
    """Piecewise-mode curve. ``None`` for linear mode."""

    def _qualified_flows(self) -> list[_BoundFlow]:
        """All flows with build-time qualified ids; inputs consume, outputs produce."""
        return [
            *(_BoundFlow(qualified_id(self.id, f.short_id), f, -1) for f in self.inputs),
            *(_BoundFlow(qualified_id(self.id, f.short_id), f, 1) for f in self.outputs),
        ]

    @override
    def model_post_init(self, __context: Any) -> None:
        """Validate short_id uniqueness, mode exclusivity, and factor references."""
        flows = [*self.inputs, *self.outputs]
        _check_unique_short_ids(f'Converter {self.id!r}', flows)
        known = {f.short_id for f in flows}

        if self.conversion is not None:
            if self.conversion_factors:
                msg = (
                    f'Converter {self.id!r}: cannot set both conversion_factors and conversion '
                    f'(they are mutually exclusive linear vs piecewise modes)'
                )
                raise ValueError(msg)
            curve_flows = {flow for flow, _, _ in self.conversion._iter_normalized()}
            unknown = curve_flows - known
            if unknown:
                msg = (
                    f'Converter {self.id!r}: PiecewiseConversion references unknown flow short_ids '
                    f'{sorted(unknown)}; known: {sorted(known)}'
                )
                raise ValueError(msg)
            if self.conversion.status is not None:
                for f in flows:
                    if f.status is not None:
                        msg = (
                            f'Converter {self.id!r}: flow {f.short_id!r} cannot have flow-level '
                            f'status when PiecewiseConversion.status is set'
                        )
                        raise ValueError(msg)
        else:
            for eq_i, equation in enumerate(self.conversion_factors):
                unknown = set(equation) - known
                if unknown:
                    msg = (
                        f'Converter {self.id!r}: conversion_factors[{eq_i}] references unknown '
                        f'flow short_ids {sorted(unknown)}; known: {sorted(known)}'
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
