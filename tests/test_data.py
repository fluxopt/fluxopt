from __future__ import annotations

import polars as pl
import pytest
import xarray as xr
from conftest import read, ts
from pydantic import ValidationError
from specsolve import DataError

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    FlowSystem,
    Port,
    ProfileRef,
    Storage,
    optimize,
)
from fluxopt.math import build_sources


def _sources(ports, carriers=None, converters=None) -> dict:
    """The sources of a three-step system that minimizes `cost`."""
    return build_sources(
        timesteps=ts(3),
        carriers=carriers or [Carrier(id='b')],
        effects=[Effect(id='cost')],
        ports=ports,
        converters=converters,
        objective='cost',
    )


def _values(table: pl.DataFrame, flow: str) -> list[float]:
    return table.filter(pl.col('flow') == flow)['value'].to_list()


class TestFlowsTable:
    def test_bounds_with_size(self):
        flow = Flow(carrier='b', size=100, relative_rate_min=0.2, relative_rate_max=0.8)
        sources = _sources([Port(id='src', imports=[flow])])
        assert _values(sources['rate_min'], 'src(b)') == [20.0] * 3, 'the relative floor times the size'
        assert _values(sources['rate_max'], 'src(b)') == [80.0] * 3, 'the relative ceiling times the size'
        assert _values(sources['size_bound'], 'src(b)') == [100.0]
        assert sources['is_bounded']['flow'].to_list() == ['src(b)']

    def test_fixed_profile(self):
        flow = Flow(carrier='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        sources = _sources([Port(id='sink', exports=[flow])])
        assert _values(sources['rate_min'], 'sink(b)') == [50.0, 80.0, 60.0]
        assert _values(sources['rate_max'], 'sink(b)') == [50.0, 80.0, 60.0], 'a profile pins the rate'
        assert sources['is_profile']['flow'].to_list() == ['sink(b)']

    def test_unsized_flow(self):
        sources = _sources([Port(id='src', imports=[Flow(carrier='b')])])
        assert sources['is_bounded'].is_empty(), 'no size to bound against'
        assert sources['is_profile'].is_empty(), 'no profile to follow'


class TestCarriersData:
    def test_coefficients(self):
        out_flow = Flow(carrier='b', size=100)
        in_flow = Flow(carrier='b', size=100)
        sources = _sources([Port(id='src', imports=[out_flow]), Port(id='sink', exports=[in_flow])])
        sign = sources['carrier_sign']
        assert dict(zip(sign['flow'], sign['value'], strict=True)) == {'src(b)': 1.0, 'sink(b)': -1.0}, (
            'an import feeds the carrier, an export draws from it'
        )
        assert set(sources['carrier_of']['carrier'].to_list()) == {'b'}


class TestBuildValidation:
    def test_undeclared_effect_rejected_without_flow_system(self) -> None:
        """Building the sources directly rejects undeclared effect references."""
        with pytest.raises(ValueError, match=r"undeclared effect\(s\) \['co2'\]"):
            _sources(
                [Port(id='grid', imports=[Flow(carrier='elec', size=10, effects_per_flow_hour={'co2': 1.0})])],
                carriers=[Carrier(id='elec')],
            )

    def test_a_status_floor_of_zero_from_a_profile_is_refused_at_build(self) -> None:
        """`Flow` refuses a zero floor under a status, but cannot see one a `ProfileRef` supplies."""
        from fluxopt import ProfileRef, Status

        system = FlowSystem(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='grid',
                    imports=[
                        Flow(
                            carrier='elec',
                            size=100,
                            relative_rate_min=ProfileRef(dataset='p', variable='floor'),
                            status=Status(),
                        )
                    ],
                ),
                Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        with pytest.raises(ValueError, match='on/off is indistinguishable'):
            system.sources({'p': {'floor': xr.DataArray([0.3, 0.0, 0.3], dims=['time'])}})


class TestConvertersTable:
    def test_scalar_factors(self):
        boiler = Converter.boiler('boiler', 0.9, Flow(carrier='gas', size=200), Flow(carrier='heat', size=100))
        sources = _sources(
            [Port(id='src', imports=[Flow(carrier='gas', size=200)])],
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            converters=[boiler],
        )
        first = sources['conversion_factor'].filter(pl.col('eq_idx') == 0).group_by('flow').first()
        assert dict(zip(first['flow'], first['value'], strict=True)) == {'boiler(gas)': 0.9, 'boiler(heat)': -1.0}, (
            'only the flows the equation names have rows at all'
        )


class TestEffectsTable:
    def test_flow_coefficients(self):
        flow = Flow(carrier='b', size=100, effects_per_flow_hour={'cost': 0.04})
        pairs = _sources([Port(id='src', imports=[flow])])['effects_per_flow_hour']
        # One row per (flow, effect) the flow actually charges — not a dense
        # product over every flow and every effect.
        assert pairs['flow'].unique().to_list() == ['src(b)']
        assert pairs['effect'].unique().to_list() == ['cost']
        assert pairs['value'].to_list() == [0.04, 0.04, 0.04]


class TestFlowNodeId:
    def test_node_included_in_default_short_id(self):
        """Flow with node set auto-generates carrier:node short_id."""
        f = Flow(carrier='heat', node='A')
        assert f.short_id == 'heat:A'

    def test_node_without_node_uses_carrier(self):
        """Flow without node uses carrier as short_id."""
        f = Flow(carrier='heat')
        assert f.short_id == 'heat'


class TestStorageValidation:
    def test_mismatched_carriers_raises(self):
        """Storage with different charging/discharging carriers raises ValueError."""
        with pytest.raises(ValueError, match='charging carrier'):
            Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='heat'))

    def test_same_short_id_resolves_to_charge_discharge(self):
        """Colliding short_ids resolve to charge/discharge in the qualified ids only."""
        s = Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='elec'))
        assert s.charging.short_id == 'elec'  # declaration untouched
        assert s.discharging.short_id == 'elec'
        assert s._charging_id == 'bat(charge)'
        assert s._discharging_id == 'bat(discharge)'

    def test_distinct_short_ids_preserved(self):
        """Storage with explicit different short_ids keeps them in qualified id."""
        s = Storage(
            id='bat', charging=Flow(carrier='elec', short_id='in'), discharging=Flow(carrier='elec', short_id='out')
        )
        assert s._charging_id == 'bat(in)'
        assert s._discharging_id == 'bat(out)'


class TestFlowQualification:
    def test_declarations_are_never_mutated(self):
        """Placing a flow in a component leaves the flow object untouched."""
        f = Flow(carrier='elec')
        Port(id='grid', imports=[f])
        assert f.short_id == 'elec'
        assert not hasattr(f, 'id')

    def test_port_qualified_flows_carry_signs(self):
        buy, sell = Flow(carrier='elec', short_id='buy'), Flow(carrier='elec', short_id='sell')
        port = Port(id='grid', imports=[buy], exports=[sell])
        assert [(bf.id, bf.sign) for bf in port._qualified_flows()] == [
            ('grid(buy)', 1),
            ('grid(sell)', -1),
        ]

    def test_flow_reused_across_components_gets_two_entries(self):
        """One flow declaration placed in two components yields two dataset columns."""
        f = Flow(carrier='b', size=100)
        sources = _sources([Port(id='src', imports=[f]), Port(id='sink', exports=[f])])
        assert sources['flow']['flow'].to_list() == ['src(b)', 'sink(b)']

    def test_port_duplicate_short_ids_raise_at_construction(self):
        with pytest.raises(ValueError, match=r"Port 'grid': duplicate flow short_id\(s\) \['elec'\]"):
            Port(id='grid', imports=[Flow(carrier='elec')], exports=[Flow(carrier='elec')])

    def test_converter_duplicate_short_ids_raise_at_construction(self):
        with pytest.raises(ValueError, match=r"Converter 'c': duplicate flow short_id\(s\) \['gas'\]"):
            Converter(
                id='c',
                inputs=[Flow(carrier='gas'), Flow(carrier='gas')],
                outputs=[Flow(carrier='heat')],
                conversion_factors=[{'gas': 0.9, 'heat': -1}],
            )


class TestConverterValidation:
    def test_unknown_short_id_in_conversion_factors_raises(self):
        with pytest.raises(ValueError, match=r"unknown flow short_ids \['gas'\]"):
            Converter(
                id='boiler',
                inputs=[Flow(carrier='Gas')],
                outputs=[Flow(carrier='Heat')],
                conversion_factors=[{'gas': 0.9, 'Heat': -1}],
            )

    def test_unknown_short_id_reports_equation_index(self):
        with pytest.raises(ValueError, match=r'conversion_factors\[1\]'):
            Converter(
                id='chp',
                inputs=[Flow(carrier='Gas')],
                outputs=[Flow(carrier='Heat'), Flow(carrier='Elec')],
                conversion_factors=[
                    {'Gas': 0.5, 'Heat': -1},
                    {'Gas': 0.4, 'Electricity': -1},
                ],
            )

    def test_known_short_ids_pass(self):
        conv = Converter(
            id='boiler',
            inputs=[Flow(carrier='Gas')],
            outputs=[Flow(carrier='Heat')],
            conversion_factors=[{'Gas': 0.9, 'Heat': -1}],
        )
        assert conv.conversion_factors[0]['Gas'] == 0.9


class TestCarrierValidation:
    def test_undeclared_carrier_raises(self):
        """Flow referencing an undeclared carrier raises ValueError."""
        with pytest.raises(ValueError, match='undeclared carrier'):
            optimize(
                timesteps=ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                objective='cost',
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_undeclared_carrier_without_flow_system(self):
        """Building the sources directly rejects flows with undeclared carriers."""
        with pytest.raises(ValueError, match=r"undeclared carrier\(s\) \['elec'\]"):
            build_sources(
                timesteps=ts(2),
                objective='cost',
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_duplicate_carrier_raises(self):
        """Duplicate carrier declarations raise ValueError."""
        with pytest.raises(ValueError, match='Duplicate carrier id'):
            build_sources(
                timesteps=ts(2),
                objective='cost',
                carriers=[Carrier(id='elec'), Carrier(id='elec')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_flow_node_on_nodeless_carrier_raises(self):
        """Flow with node on a carrier without nodes raises ValueError."""
        with pytest.raises(ValueError, match='has no nodes'):
            build_sources(
                timesteps=ts(2),
                objective='cost',
                carriers=[Carrier(id='heat')],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='A', size=100)])],
            )

    def test_flow_node_not_in_carrier_nodes_raises(self):
        """Flow with node not declared on carrier raises ValueError."""
        with pytest.raises(ValueError, match="node='C'"):
            build_sources(
                timesteps=ts(2),
                objective='cost',
                carriers=[Carrier(id='heat', nodes=['A', 'B'])],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='C', size=100)])],
            )


class TestCarrierBalance:
    def test_carrier_balance_cancels_per_carrier(self):
        """Each flow's signed share, grouped by the carrier the sources map it to, sums to zero."""
        system = FlowSystem(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='src', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='sink', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        balance = read(system.optimize(), 'carrier_balance', 'expression')
        carrier_of = system.sources()['carrier_of']
        carrier = dict(zip(carrier_of['flow'], carrier_of['carrier'], strict=True))
        balance = balance.assign_coords(carrier=('flow', [carrier[str(f)] for f in balance.coords['flow'].values]))
        assert balance.sel(flow='src(elec)').values.tolist() == pytest.approx([50.0, 80.0, 60.0]), 'a source produces'
        assert balance.groupby('carrier').sum().sel(carrier='elec').values.tolist() == pytest.approx([0.0] * 3)


class TestMultiNodeCarrier:
    def test_independent_node_balance(self):
        """Two flows on the same carrier but different nodes get independent balance equations."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='src_a', imports=[Flow(carrier='heat', node='A', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='src_b', imports=[Flow(carrier='heat', node='B', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='sink_a',
                    exports=[Flow(carrier='heat', node='A', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])],
                ),
                Port(
                    id='sink_b',
                    exports=[Flow(carrier='heat', node='B', size=100, fixed_relative_profile=[0.8, 0.8, 0.8])],
                ),
            ],
        )
        # Source A matches sink A demand (50 MW)
        rate_a = read(result, 'rate').sel(flow='src_a(heat:A)').values
        for val in rate_a:
            assert val == pytest.approx(50.0, abs=1e-4)

        # Source B matches sink B demand (80 MW)
        rate_b = read(result, 'rate').sel(flow='src_b(heat:B)').values
        for val in rate_b:
            assert val == pytest.approx(80.0, abs=1e-4)

    def test_node_in_carrier_dim_id(self):
        """Carrier dimension coordinates contain 'heat:A' and 'heat:B'."""
        sources = build_sources(
            timesteps=ts(3),
            objective='cost',
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            ports=[
                Port(
                    id='src_a', imports=[Flow(carrier='heat', node='A', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='src_b', imports=[Flow(carrier='heat', node='B', size=100, effects_per_flow_hour={'cost': 0.04})]
                ),
                Port(
                    id='sink_a',
                    exports=[Flow(carrier='heat', node='A', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])],
                ),
                Port(
                    id='sink_b',
                    exports=[Flow(carrier='heat', node='B', size=100, fixed_relative_profile=[0.8, 0.8, 0.8])],
                ),
            ],
        )
        assert sources['carrier']['carrier'].to_list() == ['heat:A', 'heat:B']


class TestContributionsAreDeclared:
    def test_the_program_names_every_contribution_the_ledger_sums(self):
        """The breakdown and the ledger are one declaration, so they must agree.

        `direct_step` and `direct_lump` sum exactly the `contribution_*`
        expressions, and `reporting.yaml` prices each of them as `priced_*`; a
        contribution the ledger sums but the report does not price would
        charge a cost nobody is attributed.
        """

        from fluxopt.math import program

        math = program()
        declared = {name for name in math.expressions if name.startswith('contribution_')}
        missing = {name for name in declared if name.replace('contribution_', 'priced_', 1) not in math.expressions}
        assert not missing, 'every contribution is reported again with its cross-effects charged'
        # Each feature adds one term to the ledger, and the term sums that
        # feature's contributions, so the ledger reaches them through the terms.
        ledger = [math.expressions[half].expression for half in ('direct_step', 'direct_lump')]
        terms = [term.strip() for body in ledger for term in body.split('+')]
        summed = ' '.join(ledger + [math.expressions[term].expression for term in terms])
        for name in declared:
            assert name in summed, f'{name} is read back but the ledger never sums it'


class TestStorageRanges:
    """Physical ranges are refused where the value is written."""

    def _storage(self, **kwargs):
        return Storage(id='b', charging=Flow(carrier='e'), discharging=Flow(carrier='e'), **kwargs)

    @pytest.mark.parametrize(
        ('kwargs', 'match'),
        [
            ({'capacity': -5}, 'capacity is negative'),
            ({'eta_charge': 0}, r'eta_charge must be in \(0.0, 1.0\]'),
            ({'eta_discharge': 1.5}, r'eta_discharge must be in \(0.0, 1.0\]'),
            ({'relative_loss_per_hour': 1.4}, r'relative_loss_per_hour must be in \[0.0, 1.0\]'),
            ({'relative_loss_per_hour': [0.1, 1.4]}, 'relative_loss_per_hour must be in'),
        ],
    )
    def test_refused_at_construction(self, kwargs, match):
        with pytest.raises(ValidationError, match=match):
            self._storage(**kwargs)

    def test_a_profile_ref_is_checked_when_it_is_bound(self):
        """Its numbers live elsewhere, so the element cannot see them.

        The element accepts the reference, and the values only exist once
        profiles are resolved; the program's assumption refuses them at bind.
        """
        system = FlowSystem(
            timesteps=ts(3),
            carriers=[Carrier(id='e')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[Port(id='g', imports=[Flow(carrier='e', size=10, effects_per_flow_hour={'cost': 1.0})])],
            storages=[self._storage(capacity=10, eta_charge=ProfileRef(dataset='p', variable='eta'))],
        )
        with pytest.raises(DataError, match='charging_efficiency_is_a_fraction'):
            system.optimize({'p': {'eta': xr.DataArray([0.9, 0.9, 1.7], dims=['time'])}})
