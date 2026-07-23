"""Mathematical correctness tests for COMBINATIONS of features.

These tests verify that status parameters, investment sizing, and effects
work correctly when combined — catching interaction bugs that single-feature
tests miss.
"""

import numpy as np
import pytest
from conftest import assert_off_blocks, assert_on_blocks
from numpy.testing import assert_allclose

from fluxopt import Carrier, Converter, Effect, Flow, PiecewiseConversion, Port, Sizing, Status

from .conftest import ts, waste


class TestPiecewiseWithInvestment:
    """Tests combining PiecewiseConversion with InvestParameters."""

    def test_piecewise_conversion_with_investment_sizing(self, optimize):
        """Proves: PiecewiseConversion and Sizing on the same converter's flow work together.

        Seg1: fuel 0→30, heat 0→20. Seg2: fuel 30→80, heat 20→70. Demand=[40,40] falls in seg2:
        fuel = 30 + (40-20)/(70-20) * (80-30) = 50. Heat flow sized to peak 40 at 1€/size.
        cost = 40 invest + 2*50 fuel = 140, unique to both mechanisms cooperating.
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([40, 40]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[
                        Flow(carrier='Gas', short_id='fuel', size=Sizing(size_min=0, size_max=100, mandatory=False))
                    ],
                    outputs=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=0, size_max=100, mandatory=False, effects_per_size={'cost': 1}),
                        )
                    ],
                    conversion=PiecewiseConversion(points={'fuel': [0, 30, 80], 'Heat': [0, 20, 70]}),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert_allclose(result.sizes.sel(flow='Converter(Heat)').item(), 40.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 140.0, rtol=1e-4)

    @pytest.mark.skip(reason='piecewise investment effects not supported — issue #26')
    def test_piecewise_invest_cost_with_optional_skip(self, optimize):
        """Proves: Piecewise investment cost function works with optional investment."""


class TestPiecewiseWithStatus:
    """Tests combining PiecewiseConversion with StatusParameters."""

    def test_piecewise_nonlinear_conversion_with_startup_cost(self, optimize):
        """Proves: PiecewiseConversion (non-1:1 ratio) and startup costs interact correctly.

        flixopt's off piece [0,0] becomes a component-level Status providing the off-state.
        Operating piece: fuel 30→60, heat 30→50. Startup=100€. Demand=[0,40,0,40] → 2 startups.
        heat=40 → fuel = 30 + (40-30)/(50-30) * (60-30) = 45. cost = 2*45 + 2*100 = 290
        (280 with 1:1 conversion, 90 without startup cost).
        """
        demand = np.array([0, 40, 0, 40])
        result = optimize(
            timesteps=ts(4),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=demand)]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [30, 60], 'Heat': [30, 50]}, status=Status(effects_per_startup={'cost': 100})
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert_allclose(result.solution['flow--rate'].sel(flow='Converter(fuel)').values[1], 45.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 290.0, rtol=1e-4)

    def test_piecewise_minimum_load_with_status(self, optimize):
        """Proves: Piecewise gap enforces minimum load, interacting with status on/off.

        flixopt's off piece [0,0] becomes a component-level Status; operating piece fuel/heat
        20→50 creates min load 20. Demand=[15,40]: t=0 below min load → converter OFF, backup
        at 5€/kWh covers. cost = 15*5 + 40 = 115 (55 without the gap).
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([15, 40]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(points={'fuel': [20, 50], 'Heat': [20, 50]}, status=Status()),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 115.0, rtol=1e-4)
        conv_heat = result.solution['flow--rate'].sel(flow='Converter(Heat)').values[0]
        assert conv_heat < 1e-5, f'Converter should be off at t=0 (demand < min_load), got {conv_heat}'

    def test_piecewise_no_zero_point_with_status(self, optimize):
        """Proves: Piecewise WITHOUT off-state breakpoint plus Status lets the converter turn OFF.

        Curve: fuel 20→60, heat 10→40 — a mandatory operating range when ON; without Status the
        converter would be forced to produce ≥10 at t=0. Demand=[5,35], backup at 5€/kWh.
        t=0: demand 5 < min heat 10 → OFF, backup = 25. t=1: fuel = 20 + (35-10)/(40-10)*40 ≈ 53.3
        (35 if the piecewise ratio were ignored).
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([5, 35]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 5})]),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(points={'fuel': [20, 60], 'Heat': [10, 40]}, status=Status()),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        expected_fuel_t1 = 20 + (25 / 30) * 40
        fuel = result.solution['flow--rate'].sel(flow='Converter(fuel)').values
        assert_allclose(fuel[1], expected_fuel_t1, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 25.0 + expected_fuel_t1, rtol=1e-4)
        assert fuel[0] < 1e-5  # OFF at t=0 despite no zero point in the curve

    def test_piecewise_no_zero_point_startup_cost(self, optimize):
        """Proves: Piecewise without zero point + startup cost work together.

        Curve: fuel 30→80, heat 20→60 (no off point), Status allows OFF. Startup=200€.
        Demand=[0,40,0,40] → 2 startups. heat=40 → fuel = 30 + (40-20)/(60-20) * (80-30) = 55.
        cost = 2*55 + 2*200 = 510 (480 with 1:1 conversion, 110 without startup cost).
        """
        demand = np.array([0, 40, 0, 40])
        result = optimize(
            timesteps=ts(4),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=demand)]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
                Port(id='Backup', imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 100})]),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [30, 80], 'Heat': [20, 60]}, status=Status(effects_per_startup={'cost': 200})
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        expected_fuel = 30 + (20 / 40) * 50
        assert_allclose(result.solution['flow--rate'].sel(flow='Converter(fuel)').values[1], expected_fuel, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 2 * expected_fuel + 400, rtol=1e-4)


class TestPiecewiseThreeSegments:
    """Tests for piecewise conversion with 3+ segments."""

    def _run_three_segment(self, optimize, demand: float):
        """Optimize the shared 3-segment setup for a constant demand level.

        Seg1: fuel 0→10, heat 0→10. Seg2: fuel 10→30, heat 10→25. Seg3: fuel 30→60, heat 25→55.
        """
        profile = np.array([demand, demand])
        return optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=profile)]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat')],
                    conversion=PiecewiseConversion(points={'fuel': [0, 10, 30, 60], 'Heat': [0, 10, 25, 55]}),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )

    def test_three_segment_piecewise(self, optimize):
        """Proves: 3-segment PiecewiseConversion correctly selects the optimal segment.

        Demand=40 falls in seg3: fuel = 30 + (40-25)/(55-25) * (60-30) = 45, cost = 2*45 = 90.
        """
        result = self._run_three_segment(optimize, 40.0)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 90.0, rtol=1e-4)
        assert_allclose(result.solution['flow--rate'].sel(flow='Converter(fuel)').values[0], 45.0, rtol=1e-4)

    def test_three_segment_low_load_selection(self, optimize):
        """Proves: With 3 segments, low demand correctly uses segment 1.

        Demand=5 falls in seg1 (1:1): fuel = 5, cost = 2*5 = 10; other segments' ratios would differ.
        """
        result = self._run_three_segment(optimize, 5.0)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-4)

    def test_three_segment_mid_load_selection(self, optimize):
        """Proves: With 3 segments, mid demand correctly uses segment 2.

        Demand=18 falls in seg2: fuel = 10 + (18-10)/(25-10) * (30-10) ≈ 20.67, unique to seg2.
        """
        result = self._run_three_segment(optimize, 18.0)
        expected_fuel = 10 + (8 / 15) * 20
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 2 * expected_fuel, rtol=1e-4)


class TestStatusWithEffects:
    """Tests for StatusParameters contributing to non-standard effects."""

    def test_startup_cost_on_co2_effect(self, optimize):
        """Proves: effects_per_startup can contribute to a non-cost effect (CO2),
        and that this correctly interacts with effect constraints.

        CO2 capped at total_max=60. Boiler startup emits 50kg CO2.
        Demand=[0,20,0,20] → 2 startups = 100kg CO2. Exceeds cap!
        Optimizer must reduce startups by keeping boiler running continuously.

        Sensitivity: Without CO2 cap, 2 startups optimal. With cap=60, forced to 1 startup.
        """

        result = optimize(
            timesteps=ts(4),
            effects=[
                Effect(id='cost'),
                Effect(id='CO2', total_max=60),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 20, 0, 20])),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=100,
                        relative_rate_min=0.1,
                        prior_rates=[0],
                        status=Status(effects_per_startup={'CO2': 50}),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 60.0 + 1e-5
        # Verify only 1 startup (continuous operation)
        on = result.solution['flow--on'].sel(flow='Boiler(Heat)').values
        startups = sum(1 for i in range(len(on)) if on[i] > 0.5 and (i == 0 or on[i - 1] < 0.5))
        assert startups <= 1, f'Expected ≤1 startup, got {startups}: on={on}'

    def test_effects_per_active_hour_on_multiple_effects(self, optimize):
        """Proves: effects_per_running_hour can contribute to multiple effects simultaneously.

        Boiler with effects_per_running_hour={'cost': 10, 'CO2': 5}.
        Demand=[20,20]. Boiler on 2 hours.

        Sensitivity: Without effects_per_running_hour, costs=40, CO2=0.
        With it, costs = 40 + 2*10 = 60, CO2 = 2*5 = 10.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[
                Effect(id='cost'),
                Effect(id='CO2'),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([20, 20])),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=100,
                        relative_rate_min=0.1,
                        status=Status(effects_per_running_hour={'cost': 10, 'CO2': 5}),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 10.0, rtol=1e-5)


class TestInvestWithRelativeMinimum:
    """Tests combining Sizing with relative_rate_min."""

    def test_invest_sizing_respects_relative_rate_min(self, optimize):
        """Proves: relative_rate_min on an invested flow forces the boiler OFF at
        low-demand timesteps, requiring expensive backup.

        Sensitivity: Without relative_rate_min: size=50, ON both hours, fuel=55, total=80.
        With it: size=50, OFF at t=0, backup=5*10=50, total=125.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([5, 50])),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
                Port(
                    id='Backup',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 10}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        relative_rate_min=0.5,
                        size=Sizing(
                            size_min=0,
                            size_max=100,
                            mandatory=True,
                            effects_per_size={'cost': 0.5},
                        ),
                        status=Status(),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert_allclose(result.sizes.sel(flow='Boiler(Heat)').item(), 50.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 125.0, rtol=1e-4)
        # Verify boiler is OFF at t=0
        assert result.solution['flow--on'].sel(flow='Boiler(Heat)').values[0] < 0.5


class TestConversionWithTimeVaryingEffects:
    """Tests for conversion factors with time-varying effects."""

    def test_time_varying_effects_per_flow_hour(self, optimize):
        """Proves: Time-varying effects_per_flow_hour correctly applies different rates
        per timestep when combined with conversion.

        Boiler eta=0.5. Gas cost = [1, 3]. Demand=[20, 10].
        t=0: fuel=40, cost=40. t=1: fuel=20, cost=60. Total=100.

        Sensitivity: If mean(2) were used: cost=120. Only per-timestep gives 100.
        """

        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([20, 10])),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': np.array([1, 3])}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(carrier='Heat'),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_effects_per_flow_hour_with_dual_output_conversion(self, optimize):
        """Proves: effects_per_flow_hour applied to individual flows of a multi-output
        converter correctly accumulates effects for each flow independently.

        CHP: fuel→heat+elec. Fuel costs 1€/kWh, elec earns -2€/kWh.
        CO2: fuel emits 0.5 kg/kWh, elec avoids -0.3 kg/kWh.
        Demand=50 heat per timestep.

        Sensitivity: Total uniquely determined by conversion factors + effects.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[
                Effect(id='cost'),
                Effect(id='CO2'),
            ],
            objective='cost',
            ports=[
                Port(
                    id='HeatDemand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([50, 50])),
                    ],
                ),
                Port(
                    id='ElecGrid',
                    exports=[
                        Flow(carrier='Elec', effects_per_flow_hour={'cost': -2, 'CO2': -0.3}),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': 1, 'CO2': 0.5}),
                    ],
                ),
            ],
            converters=[
                Converter.chp(
                    'CHP',
                    eta_th=0.5,
                    eta_el=0.4,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(carrier='Heat'),
                    electrical_flow=Flow(carrier='Elec'),
                ),
            ],
            carriers=[Carrier(id='Elec'), Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # Per ts: fuel=100, elec=40. costs: 100-80=20. CO2: 50-12=38. Total: costs=40, CO2=76.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 76.0, rtol=1e-5)


@pytest.mark.skip(reason='piecewise investment effects not supported — issue #26')
class TestPiecewiseInvestWithStatus:
    """Tests combining piecewise investment costs with status parameters."""

    def test_piecewise_invest_with_startup_cost(self, optimize):
        """Proves: Piecewise investment cost and startup cost work together."""


class TestStatusWithMultipleConstraints:
    """Tests combining multiple status parameters on the same flow."""

    @pytest.mark.skip(reason='startup_limit not supported — issue #17')
    def test_startup_limit_with_downtime_max(self, optimize):
        """Proves: startup_limit and downtime_max interact correctly."""

    def test_uptime_min_with_downtime_min(self, optimize):
        """Proves: uptime_min and downtime_min together force a regular on/off pattern.

        Boiler: uptime_min=2, downtime_min=2, prior_rates=[0].
        Demand=[20]*6. Backup at eta=0.5.

        Sensitivity: Without these constraints, boiler could run all 6 hours.
        With constraints, forced into block pattern → backup needed for off blocks.
        """

        result = optimize(
            timesteps=ts(6),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([20, 20, 20, 20, 20, 20])),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            converters=[
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=100,
                        relative_rate_min=0.1,
                        prior_rates=[0],
                        status=Status(uptime_min=2, downtime_min=2),
                    ),
                ),
                Converter.boiler(
                    'Backup',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(carrier='Heat', size=100),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        on = result.solution['flow--on'].sel(flow='CheapBoiler(Heat)').values

        # Verify uptime_min: each on-block is ≥2 hours
        assert_on_blocks(on, min_length=2)

        # Verify downtime_min: each off-block is ≥2 hours (within horizon)
        assert_off_blocks(on, min_length=2)

        # Pattern [off,on,on,on,on,on]: CheapBoiler 5h=100, Backup 1h*20/0.5=40. Total=140.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 140.0, rtol=1e-5)


class TestEffectsWithConversion:
    """Tests for effects interacting with conversion and other constraints."""

    def test_effect_share_with_investment(self, optimize):
        """Proves: contribution_from works when the contribution comes from investment costs.

        cost has contribution_from={'CO2': 20}. Boiler binary invest (size=50) has CO2=10 fixed.
        direct cost = 50 invest + 20 fuel; shared = 20*10 = 200 → total 270 (70 without).
        flixopt's share_from_periodic; equivalent here since CO2 is invest-only.
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost', contribution_from={'CO2': 20}), Effect(id='CO2')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(size_min=50, size_max=50, mandatory=False, effects_fixed={'cost': 50, 'CO2': 10}),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 270.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 10.0, rtol=1e-5)

    def test_effect_maximum_with_status_contribution(self, optimize):
        """Proves: Effect maximum correctly accounts for contributions from
        StatusParameters (effects_per_startup) when constraining.

        CO2 has total_max=20. Boiler startup emits 15 kg CO2.
        Fuel emits 0.1 kg CO2/kWh. Demand=[0,10,0,10].
        2 startups = 30 kg CO2 (exceeds cap). Forced to 1 startup.

        Sensitivity: Without CO2 cap, 2 startups → CO2=30+fuel. With cap=20, ≤1 startup.
        """

        result = optimize(
            timesteps=ts(4),
            effects=[
                Effect(id='cost'),
                Effect(id='CO2', total_max=20),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 10, 0, 10])),
                    ],
                ),
                Port(
                    id='GasSrc',
                    imports=[
                        Flow(carrier='Gas', effects_per_flow_hour={'cost': 1, 'CO2': 0.1}),
                    ],
                ),
                waste('Heat'),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=100,
                        relative_rate_min=0.1,
                        prior_rates=[0],
                        status=Status(effects_per_startup={'CO2': 15}),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 20.0 + 1e-5


class TestInvestWithEffects:
    """Tests combining investment with effect constraints."""

    def test_invest_per_size_on_non_cost_effect(self, optimize):
        """Proves: effects_per_size can feed a non-cost effect whose cap bounds the investment.

        Boiler effects_per_size={'cost': 1, 'CO2': 2}; CO2 periodic_max=50 → size ≤ 25 (< peak 30).
        Backup (eta=0.5) covers the rest: cost = 25 invest + 2*25 fuel + 2*5/0.5 backup = 95.
        flixopt's maximum_periodic; equivalent here since CO2 is invest-only. Without cap: cost=60.
        """
        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost'), Effect(id='CO2', periodic_max=50)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([30, 30]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter.boiler(
                    'InvestBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(size_min=0, size_max=100, mandatory=True, effects_per_size={'cost': 1, 'CO2': 2}),
                    ),
                ),
                Converter.boiler(
                    'Backup',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(carrier='Heat', size=100),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        assert result.effect_totals.sel(effect='CO2').item() <= 50.0 + 1e-5
        assert_allclose(result.sizes.sel(flow='InvestBoiler(Heat)').item(), 25.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 95.0, rtol=1e-4)
