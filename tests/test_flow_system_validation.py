"""Fail-fast aggregate validation on FlowSystem (Phase 4)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fluxopt import Carrier, Converter, Effect, Flow, FlowSystem, Port

_BASE = {
    'timesteps': [0, 1],
    'carriers': [Carrier(id='Heat')],
    'effects': [Effect(id='cost')],
    'objective_effects': 'cost',
}


def _system(**overrides: object) -> FlowSystem:
    return FlowSystem(
        **{
            **_BASE,
            'ports': [Port(id='S', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})])],
            **overrides,
        }
    )


class TestValidReferences:
    def test_valid_system_constructs(self) -> None:
        assert _system() is not None

    def test_penalty_effect_is_allowed(self) -> None:
        # 'penalty' is a built-in objective target even when not declared.
        assert _system(objective_effects={'cost': 1, 'penalty': 0}) is not None

    def test_nested_effect_refs_pass(self) -> None:
        # effect keys inside Sizing / Status still validate.
        from fluxopt import Sizing, Status

        assert (
            _system(
                effects=[Effect(id='cost'), Effect(id='co2')],
                ports=[
                    Port(
                        id='S',
                        imports=[
                            Flow(
                                carrier='Heat',
                                size=Sizing(size_min=0, size_max=10, effects_per_size={'co2': 1}),
                                relative_rate_min=0.1,
                                status=Status(effects_per_startup={'cost': 5}),
                            )
                        ],
                    )
                ],
            )
            is not None
        )


class TestFailFast:
    def test_unknown_effect_in_flow(self) -> None:
        with pytest.raises(ValidationError, match='undeclared effect'):
            _system(ports=[Port(id='S', imports=[Flow(carrier='Heat', effects_per_flow_hour={'co2': 1})])])

    def test_unknown_effect_in_contribution_from(self) -> None:
        with pytest.raises(ValidationError, match='undeclared effect'):
            _system(effects=[Effect(id='cost', contribution_from={'co2': 1})])

    def test_unknown_carrier(self) -> None:
        with pytest.raises(ValidationError, match='undeclared carrier'):
            _system(ports=[Port(id='S', imports=[Flow(carrier='gas')])])

    def test_bad_objective(self) -> None:
        with pytest.raises(ValidationError, match='objective_effects references undeclared'):
            _system(objective_effects='nope')

    def test_duplicate_effect_id(self) -> None:
        with pytest.raises(ValidationError, match='Duplicate effect'):
            _system(effects=[Effect(id='cost'), Effect(id='cost')])

    def test_duplicate_component_id(self) -> None:
        with pytest.raises(ValidationError, match='Duplicate component'):
            _system(
                ports=[Port(id='S', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})])],
                converters=[Converter.boiler('S', 0.9, Flow(carrier='gas'), Flow(carrier='Heat'))],
                carriers=[Carrier(id='Heat'), Carrier(id='gas')],
            )

    def test_validation_runs_from_dict(self) -> None:
        good = _system().to_dict()
        good['ports'][0]['imports'][0]['effects_per_flow_hour'] = {'co2': 1}
        with pytest.raises(ValidationError, match='undeclared effect'):
            FlowSystem.from_dict(good)
