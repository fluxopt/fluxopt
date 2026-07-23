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
        """Proves: PiecewiseConversion and Sizing on the same converter's flow
        work together — the optimizer picks the right piecewise segment AND sizes the flow.

        Converter: fuel→heat, piecewise 2-segment.
        Seg1: fuel 0→30, heat 0→20 (efficiency 0.667).
        Seg2: fuel 30→80, heat 20→70 (efficiency 1.0, better at high load).
        Demand=[40,40]. Falls in segment 2.
        Heat flow has Sizing(size_max=100, effects_per_size={'cost': 1}).

        Sensitivity: If invest sizing were broken, the piecewise constraint couldn't
        interact with size → infeasible or wrong cost. The unique cost (invest + fuel)
        proves both mechanisms cooperate.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([40, 40])),
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
                Converter(
                    id='Converter',
                    inputs=[
                        Flow(carrier='Gas', short_id='fuel', size=Sizing(size_min=0, size_max=100, mandatory=False)),
                    ],
                    outputs=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(
                                size_min=0,
                                size_max=100,
                                mandatory=False,
                                effects_per_size={'cost': 1},
                            ),
                        ),
                    ],
                    conversion=PiecewiseConversion(points={'fuel': [0, 30, 80], 'Heat': [0, 20, 70]}),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # heat=40 in segment 2: fuel = 30 + (40-20)/(70-20) * (80-30) = 30 + 20 = 50
        # invest = 40 * 1 = 40 (size=40, peak demand)
        # fuel cost = 2 * 50 = 100
        # total = 40 + 100 = 140
        assert_allclose(result.sizes.sel(flow='Converter(Heat)').item(), 40.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 140.0, rtol=1e-4)

    @pytest.mark.skip(reason='piecewise investment effects not supported — issue #26')
    def test_piecewise_invest_cost_with_optional_skip(self, optimize):
        """Proves: Piecewise investment cost function works with optional investment."""


class TestPiecewiseWithStatus:
    """Tests combining PiecewiseConversion with StatusParameters."""

    def test_piecewise_nonlinear_conversion_with_startup_cost(self, optimize):
        """Proves: PiecewiseConversion (non-1:1 ratio) and startup costs interact correctly.

        flixopt used an off piece [0,0] + operating piece [30→60 fuel, 30→50 heat];
        fluxopt expresses the gap as the operating curve alone plus a component-level
        Status providing the off-state. The operating piece has ratio 1.5:1 (fuel:heat).
        Startup cost = 100€. Demand=[0, 40, 0, 40]. Two startups.

        heat=40 in operating range: fuel = 30 + (40-30)/(50-30) * (60-30) = 30 + 15 = 45.

        Sensitivity:
        - Without piecewise (1:1 conversion): fuel=80, total=80+200=280.
        - With piecewise (1.5:1 effective ratio): fuel=90, total=90+200=290.
        - Without startup cost: total=90 (fuel only).
        The 290 is unique to BOTH features being correct.
        """

        result = optimize(
            timesteps=ts(4),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 40, 0, 40])),
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
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [30, 60], 'Heat': [30, 50]},
                        status=Status(effects_per_startup={'cost': 100}),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # heat=40: fuel = 30 + (40-30)/(50-30) * (60-30) = 30 + 15 = 45 per active ts
        # fuel = 2 * 45 = 90
        # 2 startups * 100 = 200
        # total = 290 (not 280 as with 1:1, not 90 without startups)
        assert_allclose(result.solution['flow--rate'].sel(flow='Converter(fuel)').values[1], 45.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 290.0, rtol=1e-4)

    def test_piecewise_minimum_load_with_status(self, optimize):
        """Proves: Piecewise gap enforces minimum load, interacting with status on/off.

        flixopt used an off piece [0,0] + operating piece [20→50 fuel, 20→50 heat];
        fluxopt expresses the gap as the operating curve alone plus a component-level
        Status providing the off-state. The gap creates a minimum load of 20.
        Demand=[15, 40]. At t=0, demand=15 < min_load=20 → converter must be OFF.
        Backup covers t=0 at 5€/kWh. Converter covers t=1 at 1€/kWh.

        Sensitivity:
        - Without piecewise gap (continuous 0→50): converter produces 15 at t=0, cost=55.
        - With piecewise gap (min load 20): converter OFF at t=0, backup=75, conv=40, cost=115.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([15, 40])),
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
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 5}),
                    ],
                ),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [20, 50], 'Heat': [20, 50]},
                        status=Status(),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # t=0: demand=15 < min_load=20 → converter OFF, backup: 15*5=75
        # t=1: demand=40 → converter ON: fuel=40
        # total = 75 + 40 = 115 (without gap: 15 + 40 = 55)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 115.0, rtol=1e-4)
        # Verify converter off at t=0
        conv_heat = result.solution['flow--rate'].sel(flow='Converter(Heat)').values[0]
        assert conv_heat < 1e-5, f'Converter should be off at t=0 (demand < min_load), got {conv_heat}'

    def test_piecewise_no_zero_point_with_status(self, optimize):
        """Proves: Piecewise WITHOUT off-state breakpoint interacts with Status correctly.
        The piecewise defines a MANDATORY operating range [20→60], meaning when ON the
        converter must produce ≥20. Status allows OFF.

        Without Status, the piecewise alone would force the converter to always
        operate in [20,60]. With PiecewiseConversion.status, the optimizer can
        turn it OFF (flow=0) despite no zero point in the curve.

        Converter: fuel [20→60], heat [10→40]. Plus Status.
        Demand=[5, 35]. Backup at 5€/kWh.

        t=0: demand=5 < min_heat=10 → converter must be OFF, backup covers: 5*5=25.
        t=1: demand=35 in range → heat=35, fuel = 20 + (35-10)/(40-10)*40 = 20+33.3=53.3.

        Sensitivity:
        - Without status (converter always on): infeasible or forced to produce ≥10 at t=0.
        - With status + no zero point: converter can be OFF at t=0, ON at t=1.
        - If piecewise conversion ignored (1:1): fuel at t=1 would be 35 instead of 53.3.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([5, 35])),
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
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 5}),
                    ],
                ),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [20, 60], 'Heat': [10, 40]},
                        status=Status(),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # t=0: demand=5 < min_heat=10 → OFF, backup=5*5=25
        # t=1: heat=35 → fuel = 20 + (35-10)/(40-10) * (60-20) = 20 + 33.33 = 53.33
        # total = 25 + 53.33 = 78.33
        expected_fuel_t1 = 20 + (25 / 30) * 40
        fuel = result.solution['flow--rate'].sel(flow='Converter(fuel)').values
        assert_allclose(fuel[1], expected_fuel_t1, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 25.0 + expected_fuel_t1, rtol=1e-4)
        # Verify converter OFF at t=0 (status allows it despite no zero point)
        assert fuel[0] < 1e-5

    def test_piecewise_no_zero_point_startup_cost(self, optimize):
        """Proves: Piecewise without zero point + startup cost work together.

        Converter: fuel [30→80], heat [20→60] (no off point). Plus startup cost=200€.
        Demand=[0, 40, 0, 40]. Status allows OFF. Two startups.

        heat=40: fuel = 30 + (40-20)/(60-20) * (80-30) = 30 + 25 = 55.

        Sensitivity:
        - Without startup cost: total = 2*55 = 110.
        - With startup cost: total = 110 + 2*200 = 510.
        - If piecewise ignored (1:1): fuel=40/ts, total = 80 + 400 = 480.
        The 510 is unique to BOTH features.
        """

        result = optimize(
            timesteps=ts(4),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 40, 0, 40])),
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
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 100}),
                    ],
                ),
            ],
            converters=[
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel', size=100)],
                    outputs=[Flow(carrier='Heat', size=100)],
                    conversion=PiecewiseConversion(
                        points={'fuel': [30, 80], 'Heat': [20, 60]},
                        status=Status(effects_per_startup={'cost': 200}),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # heat=40: fuel = 30 + (40-20)/(60-20) * 50 = 30 + 25 = 55
        # fuel = 2 * 55 = 110
        # 2 startups * 200 = 400
        # total = 510 (not 480 as with 1:1, not 110 without startups)
        expected_fuel = 30 + (20 / 40) * 50
        assert_allclose(result.solution['flow--rate'].sel(flow='Converter(fuel)').values[1], expected_fuel, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 2 * expected_fuel + 400, rtol=1e-4)


class TestPiecewiseThreeSegments:
    """Tests for piecewise conversion with 3+ segments."""

    def _run_three_segment(self, optimize, demand: float):
        """Optimize the shared 3-segment setup for a constant demand level.

        Segments:
        Seg1: fuel 0→10, heat 0→10  (efficiency 1.0 — low load)
        Seg2: fuel 10→30, heat 10→25 (efficiency 0.75 — mid load, less efficient)
        Seg3: fuel 30→60, heat 25→55 (efficiency 1.0 — high load)
        """
        return optimize(
            timesteps=ts(2),
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([demand, demand])),
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
                Converter(
                    id='Converter',
                    inputs=[Flow(carrier='Gas', short_id='fuel')],
                    outputs=[Flow(carrier='Heat')],
                    conversion=PiecewiseConversion(
                        points={'fuel': [0, 10, 30, 60], 'Heat': [0, 10, 25, 55]},
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )

    def test_three_segment_piecewise(self, optimize):
        """Proves: 3-segment PiecewiseConversion correctly selects the optimal segment
        for a given demand level.

        Demand=40 falls in segment 3.

        Sensitivity: If segment selection were wrong (e.g. always seg1 ratio),
        fuel would differ. Only correct 3-segment handling gives the right fuel value.
        """

        result = self._run_three_segment(optimize, 40.0)
        # heat=40 in segment 3: fuel = 30 + (40-25)/(55-25) * (60-30) = 30 + 15 = 45
        # cost = 2 * 45 = 90
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 90.0, rtol=1e-4)
        assert_allclose(result.solution['flow--rate'].sel(flow='Converter(fuel)').values[0], 45.0, rtol=1e-4)

    def test_three_segment_low_load_selection(self, optimize):
        """Proves: With 3 segments, low demand correctly uses segment 1.

        Demand=5 falls in segment 1: fuel 0→10, heat 0→10 (1:1 ratio).

        Sensitivity: If segment 2 or 3 were incorrectly selected, fuel would differ.
        """

        result = self._run_three_segment(optimize, 5.0)
        # heat=5 in segment 1: fuel = 0 + (5-0)/(10-0) * (10-0) = 5
        # cost = 2 * 5 = 10
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-4)

    def test_three_segment_mid_load_selection(self, optimize):
        """Proves: With 3 segments, mid demand correctly uses segment 2.

        Demand=18 falls in segment 2: fuel 10→30, heat 10→25.

        Sensitivity: fuel = 10 + (18-10)/(25-10) * (30-10) = 10 + 10.667 ≈ 20.667.
        This value is unique to segment 2.
        """

        result = self._run_three_segment(optimize, 18.0)
        # heat=18 in segment 2: fuel = 10 + (18-10)/(25-10) * (30-10) = 10 + 8/15*20
        expected_fuel = 10 + (8 / 15) * 20
        expected_cost = 2 * expected_fuel
        assert_allclose(result.effect_totals.sel(effect='cost').item(), expected_cost, rtol=1e-4)


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
        """Proves: contribution_from works correctly when the contribution
        comes from investment costs of a converter.

        cost has contribution_from={'CO2': 20}. Boiler invests (binary, size=50)
        with CO2=10 from the fixed investment effects. Direct costs = invest(50)
        + fuel(20). Shared: 20 * 10 = 200. Total cost = 50 + 20 + 200 = 270.

        In flixopt this was share_from_periodic; CO2 only has invest contributions
        here, so the scalar contribution_from factor is mathematically equivalent.

        Sensitivity: Without contribution_from, cost=70. With it, cost=270.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[
                Effect(id='cost', contribution_from={'CO2': 20}),
                Effect(id='CO2'),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10])),
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
                        size=Sizing(
                            size_min=50,
                            size_max=50,
                            mandatory=False,
                            effects_fixed={'cost': 50, 'CO2': 10},
                        ),
                    ),
                ),
            ],
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
        )
        # direct cost = 50 (invest) + 20 (fuel) = 70
        # CO2 = 10 (invest only)
        # cost += 20 * 10 = 200
        # total cost = 270
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
        """Proves: effects_per_size can contribute to a non-cost effect,
        and effect constraints correctly bound the investment.

        Boiler: effects_per_size = {'cost': 1, 'CO2': 2}.
        CO2 has periodic_max=50. This limits the investment size to ≤25 (50/2).
        Demand peak=30. Without CO2 cap, size=30. With cap, size limited to 25.
        Need backup for remaining 5.

        In flixopt this was maximum_periodic; CO2 only has invest contributions
        here, so bounding the whole period's effect is equivalent.

        Sensitivity: Without CO2 cap, size=30, cost=30+30=60.
        With cap, size=25, invest_cost=25, need backup for excess → cost differs.
        """

        result = optimize(
            timesteps=ts(2),
            effects=[
                Effect(id='cost'),
                Effect(id='CO2', periodic_max=50),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([30, 30])),
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
                    'InvestBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas', short_id='fuel'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(
                            size_min=0,
                            size_max=100,
                            mandatory=True,
                            effects_per_size={'cost': 1, 'CO2': 2},
                        ),
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
        # CO2 = size * 2 ≤ 50 → size ≤ 25
        # InvestBoiler: size=25, invest_cost=25, fuel=2*25=50
        # Backup covers remaining: 2*5/0.5 = 20
        # total = 25 + 50 + 20 = 95
        assert result.effect_totals.sel(effect='CO2').item() <= 50.0 + 1e-5
        assert_allclose(result.sizes.sel(flow='InvestBoiler(Heat)').item(), 25.0, rtol=1e-4)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 95.0, rtol=1e-4)
