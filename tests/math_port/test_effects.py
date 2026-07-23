"""Mathematical correctness tests for effects & objective.

flixopt's per-domain (temporal) effect bound tests were dropped: PR #242
deliberately collapsed per-domain bounds into total_max/total_min and
periodic_max/periodic_min.
"""

import numpy as np
from numpy.testing import assert_allclose

from fluxopt import Carrier, Converter, Effect, Flow, Port, Sizing

from .conftest import ts, waste


class TestEffects:
    def test_effects_per_flow_hour(self, optimize):
        """Proves: effects_per_flow_hour correctly accumulates flow * rate for each
        named effect independently.

        Source has costs=2€/kWh and CO2=0.5kg/kWh. Total flow=30.

        Sensitivity: If effects_per_flow_hour were ignored, both effects=0. If only
        one effect were applied, the other would be wrong. Both values (60€, 15kg)
        are uniquely determined by the rates and total flow.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='cost'),
                Effect(id='CO2'),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 20])),
                    ],
                ),
                Port(
                    id='HeatSrc',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 2, 'CO2': 0.5}),
                    ],
                ),
            ],
        )
        # costs = (10+20)*2 = 60, CO2 = (10+20)*0.5 = 15
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 15.0, rtol=1e-5)

    def test_contribution_from_temporal(self, optimize):
        """Proves: contribution_from adds a weighted fraction of one effect's
        temporal sum into another effect's total.

        cost has contribution_from={'CO2': 0.5}. Source: cost=1, CO2=10 per kWh.
        Demand=20. Direct cost=20, CO2=200. Cross: 0.5*200=100. Total cost=120.

        Sensitivity: Without cross-effect, cost=20 (6x less). The 120
        value is impossible without contribution_from working.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='cost', contribution_from={'CO2': 0.5}),
                Effect(id='CO2'),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10]))],
                ),
                Port(
                    id='HeatSrc',
                    imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 10})],
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 120.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 200.0, rtol=1e-5)

    def test_effect_maximum(self, optimize):
        """Proves: maximum on an effect constrains the optimizer to respect an
        upper bound on cumulative effect, forcing suboptimal dispatch.

        CO2 capped at 15kg. Dirty source: 1€+1kgCO2/kWh. Clean source: 10€+0kgCO2/kWh.
        Demand=20. Optimizer must split: 15 from Dirty + 5 from Clean.

        Sensitivity: Without the CO2 cap, all 20 from Dirty → cost=20 instead of 65.
        The 3.25* cost increase proves the constraint is binding.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='cost'),
                Effect(id='CO2', total_max=15),
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
                    id='Dirty',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1}),
                    ],
                ),
                Port(
                    id='Clean',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 10, 'CO2': 0}),
                    ],
                ),
            ],
        )
        # Without CO2 limit: all from Dirty = 20€
        # With CO2 max=15: 15 from Dirty (15€), 5 from Clean (50€) → total 65€
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 65.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 15.0, rtol=1e-5)

    def test_effect_minimum(self, optimize):
        """Proves: minimum on an effect forces cumulative effect to reach at least
        the specified value, even if it means using a dirtier source.

        CO2 floor at 25kg. Dirty source: 1€+1kgCO2/kWh. Clean source: 1€+0kgCO2/kWh.
        Demand=20. Must produce ≥25 CO2 → Dirty ≥ 25 kWh, excess absorbed by dump.

        Sensitivity: Without minimum, optimizer could use all Clean → CO2=0.
        With total_min=25, forced to use ≥25 from Dirty → CO2≥25.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='cost'),
                Effect(id='CO2', total_min=25),
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
                    id='Dirty',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 1}),
                    ],
                ),
                Port(
                    id='Clean',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 1, 'CO2': 0}),
                    ],
                ),
                waste('Heat'),
            ],
        )
        # Must produce ≥25 CO2. Only Dirty emits CO2 at 1kg/kWh → Dirty ≥ 25 kWh.
        # Demand only 20, so 5 excess absorbed by dump. cost = 25
        assert_allclose(result.effect_totals.sel(effect='CO2').item(), 25.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 25.0, rtol=1e-5)

    def test_contribution_from_periodic(self, optimize):
        """Proves: contribution_from adds a weighted fraction of one effect's periodic
        (investment) sum into another effect's total.

        cost has contribution_from={'CO2': 10}. Boiler invest: fixed cost=100, CO2=5.
        Fuel cost=20. Direct cost = 100 + 20 = 120. CO2 periodic = 5.
        Cross: 10 * 5 = 50. Total cost = 170.

        Sensitivity: Without contribution_from, cost=120. With it, cost=170.
        """

        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[
                Effect(id='cost', contribution_from={'CO2': 10}),
                Effect(id='CO2'),
            ],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10]))],
                ),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter.boiler(
                    'Boiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(size_min=50, size_max=50, mandatory=False, effects_fixed={'cost': 100, 'CO2': 5}),
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 170.0, rtol=1e-5)

    def test_effect_maximum_periodic(self, optimize):
        """Proves: periodic_max limits the period's total effect.

        CheapBoiler (invest=10€, CO2=100kg) vs ExpensiveBoiler (invest=50€, CO2=10kg);
        CO2 periodic_max=50 rules out CheapBoiler → cost = 50 + 20 = 70 (30 without the limit).
        flixopt's maximum_periodic; equivalent here since CO2 is invest-only (PR #242).
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', periodic_max=50)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter.boiler(
                    'CheapBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(size_min=50, size_max=50, mandatory=False, effects_fixed={'cost': 10, 'CO2': 100}),
                    ),
                ),
                Converter.boiler(
                    'ExpensiveBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(size_min=50, size_max=50, mandatory=False, effects_fixed={'cost': 50, 'CO2': 10}),
                    ),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 70.0, rtol=1e-5)
        assert result.effect_totals.sel(effect='CO2').item() <= 50.0 + 1e-5

    def test_effect_minimum_periodic(self, optimize):
        """Proves: periodic_min forces a minimum on the period's total effect.

        Optional boiler invest (100€, CO2=50kg); CO2 periodic_min=40 forces the invest
        → cost = 100 + 20 = 120 (40 backup-only without the floor).
        flixopt's minimum_periodic; equivalent here since CO2 is invest-only (PR #242).
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Gas'), Carrier(id='Heat')],
            effects=[Effect(id='cost'), Effect(id='CO2', periodic_min=40)],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10]))]),
                Port(id='GasSrc', imports=[Flow(carrier='Gas', effects_per_flow_hour={'cost': 1})]),
            ],
            converters=[
                Converter.boiler(
                    'InvestBoiler',
                    thermal_efficiency=1.0,
                    fuel_flow=Flow(carrier='Gas'),
                    thermal_flow=Flow(
                        carrier='Heat',
                        size=Sizing(size_min=50, size_max=50, mandatory=False, effects_fixed={'cost': 100, 'CO2': 50}),
                    ),
                ),
                Converter.boiler(
                    'Backup',
                    thermal_efficiency=0.5,
                    fuel_flow=Flow(carrier='Gas'),
                    thermal_flow=Flow(carrier='Heat', size=100),
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 120.0, rtol=1e-5)
        assert result.effect_totals.sel(effect='CO2').item() >= 40.0 - 1e-5
