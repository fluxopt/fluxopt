from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import ts

from fluxopt import Carrier, Converter, Dims, Effect, Flow, ModelData, Port, Storage, optimize


class TestFlowsTable:
    def test_bounds_with_size(self):
        flow = Flow(carrier='b', size=100, relative_rate_min=0.2, relative_rate_max=0.8)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        ds = data.flows
        lb = ds.rel_lb.sel(flow='src(b)').values
        ub = ds.rel_ub.sel(flow='src(b)').values
        assert list(lb) == [0.2, 0.2, 0.2]
        assert list(ub) == [0.8, 0.8, 0.8]
        assert float(ds.size.sel(flow='src(b)').values) == 100.0
        assert str(ds.bound_type.sel(flow='src(b)').values) == 'bounded'

    def test_fixed_profile(self):
        flow = Flow(carrier='b', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='sink', exports=[flow])],
        )
        fixed = data.flows.fixed_profile.sel(flow='sink(b)').values
        assert list(fixed) == [0.5, 0.8, 0.6]
        assert str(data.flows.bound_type.sel(flow='sink(b)').values) == 'profile'

    def test_unsized_flow(self):
        flow = Flow(carrier='b')
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        assert str(data.flows.bound_type.sel(flow='src(b)').values) == 'unsized'


class TestCarriersData:
    def test_coefficients(self):
        out_flow = Flow(carrier='b', size=100)
        in_flow = Flow(carrier='b', size=100)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[out_flow]), Port(id='sink', exports=[in_flow])],
        )
        coeffs = data.carriers.flow_coeff
        out_coeff = float(coeffs.sel(carrier='b', flow='src(b)').values)
        in_coeff = float(coeffs.sel(carrier='b', flow='sink(b)').values)
        assert out_coeff == 1.0  # output to carrier
        assert in_coeff == -1.0  # input from carrier

    def test_metadata(self):
        data = ModelData.build(
            ts(2),
            carriers=[Carrier(id='elec', unit='kWh', color='blue', description='Electricity')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='elec', size=100)])],
        )
        assert str(data.carriers.unit.sel(carrier='elec').values) == 'kWh'
        assert str(data.carriers.color.sel(carrier='elec').values) == 'blue'
        assert str(data.carriers.description.sel(carrier='elec').values) == 'Electricity'

    def test_from_dataset_roundtrip(self):
        from fluxopt.model_data import CarriersData

        data = ModelData.build(
            ts(2),
            carriers=[Carrier(id='elec', unit='kWh', color='red', description='Power')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='elec', size=100)])],
        )
        ds = data.carriers.to_dataset()
        loaded = CarriersData.from_dataset(ds)
        assert str(loaded.unit.sel(carrier='elec').values) == 'kWh'
        assert str(loaded.color.sel(carrier='elec').values) == 'red'
        assert str(loaded.description.sel(carrier='elec').values) == 'Power'


class TestConvertersTable:
    def test_scalar_factors(self):
        fuel = Flow(carrier='gas', size=200)
        heat_flow = Flow(carrier='heat', size=100)
        boiler = Converter.boiler('boiler', 0.9, fuel, heat_flow)
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='gas'), Carrier(id='heat')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[Flow(carrier='gas', size=200)])],
            converters=[boiler],
        )
        ds = data.converters
        assert ds is not None
        fuel_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(gas)', time=data.dims.time[0]).values
        )
        heat_coeff = float(
            ds.flow_coeff.sel(converter='boiler', eq_idx=0, flow='boiler(heat)', time=data.dims.time[0]).values
        )
        assert fuel_coeff == 0.9
        assert heat_coeff == -1.0


class TestEffectsTable:
    def test_flow_coefficients(self):
        flow = Flow(carrier='b', size=100, effects_per_flow_hour={'cost': 0.04})
        data = ModelData.build(
            ts(3),
            carriers=[Carrier(id='b')],
            effects=[Effect(id='cost')],
            ports=[Port(id='src', imports=[flow])],
        )
        coeff = data.flows.effect_coeff.sel(flow='src(b)', effect='cost')
        assert all(v == 0.04 for v in coeff.values)


class TestFlowNodeId:
    def test_node_included_in_default_id(self):
        """Flow with node set auto-generates carrier:node id."""
        f = Flow(carrier='heat', node='A')
        assert f.id == 'heat:A'

    def test_node_without_node_uses_carrier(self):
        """Flow without node uses carrier as id."""
        f = Flow(carrier='heat')
        assert f.id == 'heat'


class TestStorageValidation:
    def test_mismatched_carriers_raises(self):
        """Storage with different charging/discharging carriers raises ValueError."""
        with pytest.raises(ValueError, match='charging carrier'):
            Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='heat'))

    def test_same_short_id_renamed_to_charge_discharge(self):
        """Storage with same short_id renames both short_id and id."""
        s = Storage(id='bat', charging=Flow(carrier='elec'), discharging=Flow(carrier='elec'))
        assert s.charging.short_id == 'charge'
        assert s.discharging.short_id == 'discharge'
        assert s.charging.id == 'bat(charge)'
        assert s.discharging.id == 'bat(discharge)'

    def test_distinct_short_ids_preserved(self):
        """Storage with explicit different short_ids keeps them in qualified id."""
        s = Storage(
            id='bat', charging=Flow(carrier='elec', short_id='in'), discharging=Flow(carrier='elec', short_id='out')
        )
        assert s.charging.id == 'bat(in)'
        assert s.discharging.id == 'bat(out)'


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
        with pytest.raises(ValueError, match='not in the declared carriers'):
            optimize(
                timesteps=ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                objective_effects='cost',
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_undeclared_carrier_in_model_data_build(self):
        """ModelData.build rejects flows with undeclared carriers."""
        with pytest.raises(ValueError, match="carrier 'elec'"):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='gas')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_duplicate_carrier_raises(self):
        """Duplicate carrier declarations raise ValueError."""
        with pytest.raises(ValueError, match='Duplicate carrier id'):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='elec'), Carrier(id='elec')],
                effects=[Effect(id='cost')],
                ports=[Port(id='grid', imports=[Flow(carrier='elec', size=100)])],
            )

    def test_flow_node_on_nodeless_carrier_raises(self):
        """Flow with node on a carrier without nodes raises ValueError."""
        with pytest.raises(ValueError, match='has no nodes'):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='heat')],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='A', size=100)])],
            )

    def test_flow_node_not_in_carrier_nodes_raises(self):
        """Flow with node not declared on carrier raises ValueError."""
        with pytest.raises(ValueError, match="node='C'"):
            ModelData.build(
                ts(2),
                carriers=[Carrier(id='heat', nodes=['A', 'B'])],
                effects=[Effect(id='cost')],
                ports=[Port(id='src', imports=[Flow(carrier='heat', node='C', size=100)])],
            )


class TestCarrierBalance:
    def test_carrier_balance_property(self):
        """StatsAccessor.carrier_balance returns signed balance per carrier."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='elec')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(id='src', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 0.04})]),
                Port(id='sink', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])]),
            ],
        )
        balance = result.stats.carrier_balance
        assert 'carrier' in balance.dims
        assert 'flow' in balance.dims
        # Source has positive coeff, sink negative — balance should sum to ~0
        total = balance.sum('flow')
        for val in total.sel(carrier='elec').values:
            assert val == pytest.approx(0.0, abs=1e-6)


class TestMultiNodeCarrier:
    def test_independent_node_balance(self):
        """Two flows on the same carrier but different nodes get independent balance equations."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='heat', nodes=['A', 'B'])],
            effects=[Effect(id='cost')],
            objective_effects='cost',
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
        rate_a = result.flow_rate('src_a(heat:A)').values
        for val in rate_a:
            assert val == pytest.approx(50.0, abs=1e-4)

        # Source B matches sink B demand (80 MW)
        rate_b = result.flow_rate('src_b(heat:B)').values
        for val in rate_b:
            assert val == pytest.approx(80.0, abs=1e-4)

    def test_node_in_carrier_dim_id(self):
        """Carrier dimension coordinates contain 'heat:A' and 'heat:B'."""
        data = ModelData.build(
            ts(3),
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
        carrier_ids = list(data.carriers.flow_coeff.coords['carrier'].values)
        assert 'heat:A' in carrier_ids
        assert 'heat:B' in carrier_ids
        assert len(carrier_ids) == 2


class TestDimsValidation:
    def test_mismatched_dim_raises(self):
        """Dims rejects arrays that are not 1D with dims=('time',)."""
        time = xr.DataArray([0, 1], dims=['time'], coords={'time': [0, 1]})
        bad_dt = xr.DataArray([1.0, 1.0], dims=['other'])
        with pytest.raises(ValueError, match='must be 1D'):
            Dims(time=time, dt=bad_dt, weights=time)

    def test_mismatched_coords_raises(self):
        """Dims rejects arrays with different time coordinates."""
        time = xr.DataArray([0, 1], dims=['time'], coords={'time': [0, 1]})
        dt = xr.DataArray([1.0, 1.0], dims=['time'], coords={'time': [0, 1]})
        bad_weights = xr.DataArray(np.ones(3), dims=['time'], coords={'time': [0, 1, 2]})
        with pytest.raises(ValueError, match='does not match'):
            Dims(time=time, dt=dt, weights=bad_weights)


class TestFlatTimeIndex:
    """Dims builds one flat time axis; periods ride along as time_period."""

    def test_single_period_unchanged(self):
        dims = Dims.build(ts(3))
        assert dims.period is None
        assert dims.time_period is None
        assert list(dims.episode_starts.values) == [True, False, False]

    def test_uniform_periods_shift_calendar_years(self):
        dims = Dims.build(ts(3), periods=[2020, 2025])
        assert len(dims.time) == 6
        years = pd.DatetimeIndex(dims.time.values).year
        assert list(years) == [2024, 2024, 2024, 2029, 2029, 2029]
        assert list(dims.time_period.values) == [2020, 2020, 2020, 2025, 2025, 2025]
        assert list(dims.start_positions) == [0, 3]
        assert list(dims.last_positions) == [2, 5]

    def test_integer_timesteps_get_running_index(self):
        dims = Dims.build([0, 1, 2], periods=[1, 2], period_weights=[1, 1])
        assert list(dims.time.values) == [0, 1, 2, 3, 4, 5]

    def test_ragged_periods_have_own_dt(self):
        dims = Dims.build(
            {
                2030: pd.date_range('2030-01-01', periods=4, freq='h'),
                2040: pd.date_range('2040-01-01', periods=2, freq='4h'),
            }
        )
        assert list(dims.dt.values) == [1, 1, 1, 1, 4, 4]
        assert list(dims.time_period.values) == [2030] * 4 + [2040] * 2
        # gap-inferred period weights
        assert list(dims.period_weights.values) == [10, 10]

    def test_leap_year_base_replicates_safely(self):
        # Feb 29 in the base grid must not collide when shifted into
        # non-leap years (whole-day offset, not calendar-year arithmetic).
        base = pd.date_range('2024-02-28 22:00', periods=6, freq='h')
        dims = Dims.build(base, periods=[2030, 2040], period_weights=[1, 1])
        flat = pd.DatetimeIndex(dims.time.values)
        assert flat.is_monotonic_increasing and flat.is_unique
        # constant whole-day offset preserves time-of-day and dt
        assert list(flat.hour[:6]) == list(flat.hour[6:])
        assert list(dims.dt.values) == [1.0] * 12

    def test_ragged_requires_datetime(self):
        with pytest.raises(TypeError, match='datetime'):
            Dims.build({2030: [0, 1, 2], 2040: [0, 1]})

    def test_mapping_forbids_periods_arg(self):
        with pytest.raises(ValueError, match='periods must not be given'):
            Dims.build({2030: ts(2), 2040: ts(2)}, periods=[2030, 2040])

    def test_overlapping_period_grids_raise(self):
        overlapping = {
            2030: pd.date_range('2030-01-01', periods=2, freq='h'),
            2040: pd.date_range('2029-01-01', periods=2, freq='h'),
        }
        with pytest.raises(ValueError, match='increasing'):
            Dims.build(overlapping)

    def test_map_to_time_strips_period_coord_from_variables(self):
        import linopy

        dims = Dims.build(ts(2), periods=[2030, 2040], period_weights=[1, 1])
        m = linopy.Model()
        var = m.add_variables(coords=dims.coords(period=True), name='size')
        mapped = dims.map_to_time(var)
        assert 'time' in mapped.dims
        assert 'period' not in mapped.coords

    def test_replicas_overlapping_raise(self):
        base = pd.DatetimeIndex(['2024-01-01', '2025-06-01'])
        with pytest.raises(ValueError, match='increasing'):
            Dims.build(base, periods=[2020, 2021], period_weights=[1, 1])


class TestOperationalInputAlignment:
    """Operational profiles must align to the flat axis — no silent resampling."""

    def _build(self, profile, timesteps=None, periods=(2030, 2040)):
        return ModelData.build(
            timesteps if timesteps is not None else ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=profile)]),
                Port(id='Grid', imports=[Flow(carrier='Heat')]),
            ],
            periods=list(periods) if periods else None,
            period_weights=[1] * len(periods) if periods else None,
        )

    def test_within_period_profile_tiles(self):
        data = self._build([3.0, 4.0])
        assert list(data.flows.fixed_profile.sel(flow='Demand(Heat)').values) == [3, 4, 3, 4]

    def test_flat_profile_used_as_is(self):
        data = self._build([1.0, 2.0, 3.0, 4.0])
        assert list(data.flows.fixed_profile.sel(flow='Demand(Heat)').values) == [1, 2, 3, 4]

    def test_period_mapping_aligns_per_period(self):
        ragged = {
            2030: pd.date_range('2030-01-01', periods=3, freq='h'),
            2040: pd.date_range('2040-01-01', periods=2, freq='4h'),
        }
        data = self._build({2030: [1.0, 2.0, 3.0], 2040: [7.0, 8.0]}, timesteps=ragged, periods=None)
        assert list(data.flows.fixed_profile.sel(flow='Demand(Heat)').values) == [1, 2, 3, 7, 8]

    def test_period_mapping_accepts_base_grid_labels(self):
        # Uniform mode shifts later periods' labels internally; users only
        # know the base grid, so {period: series} indexed by it must work.
        base = pd.DatetimeIndex(ts(2), name='time')
        data = self._build(
            {
                2030: pd.Series([1.0, 2.0], index=base),
                2040: pd.Series([7.0, 8.0], index=base),
            }
        )
        profile = data.flows.fixed_profile.sel(flow='Demand(Heat)')
        assert list(profile.values) == [1, 2, 7, 8]

    def test_series_with_foreign_index_name_raises(self):
        bad = pd.Series([1.0, 2.0], index=pd.Index(['a', 'b'], name='timestamp'))
        with pytest.raises(ValueError, match="'timestamp'"):
            self._build(bad)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match='matches no time grid'):
            self._build([1.0, 2.0, 3.0])

    def test_mapping_key_mismatch_raises(self):
        with pytest.raises(ValueError, match='do not match periods'):
            self._build({2030: [1.0, 2.0], 2035: [3.0, 4.0]})

    def test_bare_list_is_always_a_time_profile(self):
        # 3 periods x 3 within-period timesteps: the period-count collision
        # must not change the meaning — bare lists are time profiles, period
        # values require a named form.
        data = self._build([1.0, 2.0, 3.0], timesteps=ts(3), periods=(2030, 2040, 2050))
        profile = data.flows.fixed_profile.sel(flow='Demand(Heat)')
        assert list(profile.values) == [1, 2, 3] * 3

    def test_bare_list_of_period_count_length_errors_with_hint(self):
        # ts(3) x 2 periods: a bare list of 2 matches no time grid; the error
        # points to the named per-period forms instead of guessing.
        with pytest.raises(ValueError, match=r'\{period: value\} mapping'):
            self._build([1.0, 2.0], timesteps=ts(3))

    def test_per_period_values_via_period_dataarray(self):
        per_period = xr.DataArray([1.0, 2.0, 3.0], dims=['period'], coords={'period': [2030, 2040, 2050]})
        data = self._build(per_period, timesteps=ts(3), periods=(2030, 2040, 2050))
        profile = data.flows.fixed_profile.sel(flow='Demand(Heat)')
        assert list(profile.values) == [1, 1, 1, 2, 2, 2, 3, 3, 3]

    def test_per_period_values_via_mapping(self):
        data = self._build(
            {2030: [1.0, 2.0, 3.0], 2040: [1.0, 2.0, 3.0], 2050: [1.0, 2.0, 3.0]},
            timesteps=ts(3),
            periods=(2030, 2040, 2050),
        )
        profile = data.flows.fixed_profile.sel(flow='Demand(Heat)')
        assert list(profile.values) == [1, 2, 3] * 3

    def test_time_period_frame_requires_uniform_grid(self):
        ragged = {
            2030: pd.date_range('2030-01-01', periods=3, freq='h'),
            2040: pd.date_range('2040-01-01', periods=2, freq='4h'),
        }
        frame = pd.DataFrame(
            [[1.0, 2.0]] * 3,
            index=pd.DatetimeIndex(pd.date_range('2030-01-01', periods=3, freq='h'), name='time'),
            columns=pd.Index([2030, 2040], name='period'),
        )
        with pytest.raises(ValueError, match='uniform grid'):
            self._build(frame, timesteps=ragged, periods=None)
