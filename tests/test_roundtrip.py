"""Structural dict/JSON round-trip for the element layer (Phase 2)."""

from __future__ import annotations

import json

import pytest
import xarray as xr

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    Investment,
    Port,
    ProfileRef,
    Sizing,
    Status,
    Storage,
    from_dict,
    to_dict,
)


def _elements() -> list[object]:
    return [
        Carrier(id='elec', unit='MWh'),
        Effect(id='co2', unit='kg', total_max=1000.0, contribution_from={'cost': 0.05}),
        Flow(carrier='gas', effects_per_flow_hour={'cost': 0.04}),
        Sizing(size_min=0.0, size_max=100.0, effects_per_size={'cost': 50.0}),
        Investment(size_min=0.0, size_max=100.0, lifetime=20, effects_fixed_at_build={'cost': 5.0}),
        Status(uptime_min=3.0, effects_per_startup={'cost': 10.0}),
        Storage(
            id='bat',
            charging=Flow(carrier='elec'),
            discharging=Flow(carrier='elec'),
            capacity=Sizing(size_min=0.0, size_max=100.0),
        ),
        Port(id='grid', imports=[Flow(carrier='elec')], exports=[Flow(carrier='elec')]),
        Converter.boiler('boiler', 0.9, Flow(carrier='gas'), Flow(carrier='heat')),
        Converter.chp('chp', 0.4, 0.45, Flow(carrier='gas'), Flow(carrier='elec'), Flow(carrier='heat')),
    ]


class TestRoundTrip:
    @pytest.mark.parametrize('element', _elements(), ids=lambda e: type(e).__name__)
    def test_dict_is_json_safe(self, element: object) -> None:
        d = to_dict(element)
        assert json.dumps(d)  # no arbitrary objects leaked into the dict

    @pytest.mark.parametrize('element', _elements(), ids=lambda e: type(e).__name__)
    def test_from_dict_rebuilds_same_type(self, element: object) -> None:
        rebuilt = from_dict(type(element), to_dict(element))
        assert type(rebuilt) is type(element)

    def test_component_idlist_serializes_to_list(self) -> None:
        d = to_dict(Converter.boiler('b', 0.9, Flow(carrier='gas'), Flow(carrier='heat')))
        assert isinstance(d['inputs'], list)
        assert isinstance(d['outputs'], list)

    def test_component_requalifies_flows_on_rebuild(self) -> None:
        conv = from_dict(Converter, to_dict(Converter.boiler('b', 0.9, Flow(carrier='gas'), Flow(carrier='heat'))))
        assert [f.id for f in conv.inputs] == ['b(gas)']
        assert [f.id for f in conv.outputs] == ['b(heat)']

    def test_sizing_investment_union_disambiguates(self) -> None:
        f = from_dict(Flow, to_dict(Flow(carrier='g', size=Investment(size_min=0.0, size_max=100.0, lifetime=20))))
        assert isinstance(f.size, Investment)
        assert f.size.lifetime == 20


class TestProfileRef:
    def test_profileref_roundtrips_as_variate(self) -> None:
        f = Flow(carrier='gas', effects_per_flow_hour={'cost': ProfileRef(source='prices', variable='gas')})
        d = to_dict(f)
        assert d['effects_per_flow_hour']['cost'] == {'source': 'prices', 'variable': 'gas', 'dim': 'time'}
        rebuilt = from_dict(Flow, d)
        assert isinstance(rebuilt.effects_per_flow_hour['cost'], ProfileRef)

    def test_resolve_pulls_from_sources(self) -> None:
        ref = ProfileRef(source='prices', variable='gas')
        da = xr.DataArray([1.0, 2.0, 3.0], dims=['time'])
        resolved = ref.resolve({'prices': {'gas': da}})
        assert list(resolved.values) == [1.0, 2.0, 3.0]

    def test_resolve_missing_source_raises(self) -> None:
        with pytest.raises(KeyError, match='source'):
            ProfileRef(source='nope', variable='x').resolve({'prices': {}})

    def test_resolve_missing_variable_raises(self) -> None:
        with pytest.raises(KeyError, match='variable'):
            ProfileRef(source='prices', variable='nope').resolve({'prices': {}})

    def test_unresolved_ref_rejected_at_build(self) -> None:
        from fluxopt import as_dataarray

        with pytest.raises(ValueError, match='Unresolved ProfileRef'):
            as_dataarray(ProfileRef(source='p', variable='x'), {'time': [0, 1, 2]})
