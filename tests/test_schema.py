"""Element-layer validation and JSON Schema (pydantic-backed)."""

from __future__ import annotations

import pytest
import xarray as xr
from pydantic import ValidationError

from fluxopt import (
    Converter,
    Effect,
    Flow,
    all_element_schemas,
    element_schema,
)
from fluxopt.schema import ELEMENT_TYPES


class TestValidation:
    def test_rejects_wrong_scalar_type(self) -> None:
        with pytest.raises(ValidationError):
            Effect(id='co2', total_max='not a number')  # type: ignore[arg-type]

    def test_rejects_wrong_id_type(self) -> None:
        with pytest.raises(ValidationError):
            Flow(carrier=123)  # type: ignore[arg-type]

    def test_accepts_scalar_and_array_variate(self) -> None:
        f = Flow(carrier='gas', effects_per_flow_hour={'cost': 0.04})
        assert f.effects_per_flow_hour['cost'] == 0.04
        da = xr.DataArray([1.0, 2.0], dims=['time'])
        f2 = Flow(carrier='gas', effects_per_flow_hour={'cost': da})
        assert isinstance(f2.effects_per_flow_hour['cost'], xr.DataArray)

    def test_post_init_validation_still_runs(self) -> None:
        # __post_init__ guards survive the pydantic migration.
        with pytest.raises(ValueError, match='must be > 0 when status is set'):
            from fluxopt import Status

            Flow(carrier='gas', relative_rate_min=0.0, size=10.0, status=Status())

    def test_nested_identity_preserved_without_mutation(self) -> None:
        gas, heat = Flow(carrier='gas'), Flow(carrier='heat')
        conv = Converter.boiler('b', 0.9, gas, heat)
        assert conv.inputs[0] is gas  # pydantic keeps the instance
        assert gas.short_id == 'gas'  # the declaration is never modified
        assert conv._qualified_flows()[0].id == 'b(gas)'  # qualification is derived, not stored


class TestSchema:
    @pytest.mark.parametrize('element_type', ELEMENT_TYPES, ids=lambda t: t.__name__)
    def test_schema_generates_with_properties(self, element_type: type) -> None:
        schema = element_schema(element_type)
        assert schema['type'] == 'object'
        assert schema['properties']

    def test_all_element_schemas_covers_every_type(self) -> None:
        schemas = all_element_schemas()
        assert {t.__name__ for t in ELEMENT_TYPES} == set(schemas)

    def test_effect_schema_exposes_bounds(self) -> None:
        props = element_schema(Effect)['properties']
        assert {'total_max', 'total_min', 'contribution_from'} <= set(props)

    def test_schema_is_json_serializable(self) -> None:
        import json

        # arbitrary Variate fields degrade to permissive {}, but the schema
        # itself must always be plain JSON.
        assert json.dumps(all_element_schemas())
