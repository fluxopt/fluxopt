"""Mathematical correctness tests for multi-period optimization."""

import numpy as np
import pytest
from conftest import ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Investment, Port


class TestMultiPeriod:
    def test_period_weights_affect_objective(self, optimize):
        """Proves: period weights scale per-period costs in the objective.

        3 timesteps, periods=[2020, 2025], period_weights=[5, 5].
        Grid @1 cost/MWh, Demand=[10, 10, 10]. Per-period cost=30.
        Objective = 5*30 + 5*30 = 300.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow('Heat', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        assert_allclose(result.objective, 300.0, rtol=1e-5)

    @pytest.mark.skip(reason='multi-period over-period constraints not yet implemented')
    def test_flow_hours_max_over_periods(self, optimize):
        """Proves: flow_hours_max_over_periods caps the weighted total flow-hours."""

    @pytest.mark.skip(reason='multi-period over-period constraints not yet implemented')
    def test_flow_hours_min_over_periods(self, optimize):
        """Proves: flow_hours_min_over_periods forces a minimum weighted total."""

    @pytest.mark.skip(reason='multi-period over-period constraints not yet implemented')
    def test_effect_maximum_over_periods(self, optimize):
        """Proves: Effect.maximum_over_periods caps weighted total of an effect."""

    @pytest.mark.skip(reason='multi-period over-period constraints not yet implemented')
    def test_effect_minimum_over_periods(self, optimize):
        """Proves: Effect.minimum_over_periods forces minimum weighted total."""

    @pytest.mark.skip(reason='multi-period linked periods not yet implemented')
    def test_invest_linked_periods(self, optimize):
        """Proves: InvestParameters.linked_periods forces equal sizes across periods."""

    def test_effect_period_weights_periodic(self, optimize):
        """Proves: Effect.period_weights_periodic overrides global period weights.

        3 timesteps, periods=[2020, 2025], global weights=[5, 5].
        Per-period cost = 30. With custom periodic weights [1, 2]:
        Objective = 1*30 + 2*30 = 90  (instead of 5*30 + 5*30 = 300).
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[
                Effect('cost', is_objective=True, period_weights_periodic=[1, 2]),
            ],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow('Heat', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        assert_allclose(result.objective, 90.0, rtol=1e-5)

    @pytest.mark.skip(reason='multi-period storage constraints not yet implemented')
    def test_storage_relative_minimum_final_level_scalar(self, optimize):
        """Proves: scalar relative_minimum_final_level works in multi-period."""

    @pytest.mark.skip(reason='multi-period storage constraints not yet implemented')
    def test_storage_relative_maximum_final_level_scalar(self, optimize):
        """Proves: scalar relative_maximum_final_level works in multi-period."""


@pytest.mark.xfail(
    reason='effect_contributions does not yet support investment/period decomposition (#84)',
    condition=True,
    strict=False,  # pass when no investment costs (optional/prior/lifetime-only)
)
class TestInvestment:
    def test_investment_mandatory_builds_once(self, optimize):
        """Proves: mandatory Investment builds exactly once across periods.

        3 timesteps, 3 periods [2020, 2025, 2030], weights=[5, 5, 5].
        Grid with Investment(0, 20, mandatory=True), CAPEX=10/MW once.
        Demand=10 MW constant. Cheapest: build in first period, CAPEX=10*10=100.
        Operational cost per period: 10*3*1=30, weighted=5*30=150.
        Total recurring: 3*150=450. Total once: 100. Objective=450+100=550.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(
                            'Heat',
                            size=Investment(0, 20, effects_per_size={'cost': 10}),
                            effects_per_flow_hour={'cost': 1},
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025, 2030],
            period_weights=[5, 5, 5],
        )
        # CAPEX: 10/MW * 10 MW = 100 (once)
        # Operational: 10 MW * 3h * 1 cost/MWh = 30 per period, weighted 5*30=150 per period
        # Total = 3 * 150 + 100 = 550
        assert_allclose(result.objective, 550.0, rtol=1e-4)

    def test_investment_optional_skips_when_expensive(self, optimize):
        """Proves: optional Investment is not built if cost exceeds benefit.

        Demand=0, so building would only add cost. Optional investment should not build.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([0, 0, 0])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(
                            'Heat',
                            size=Investment(0, 20, mandatory=False, effects_per_size={'cost': 100}),
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        assert_allclose(result.objective, 0.0, atol=1e-4)

    def test_investment_lifetime_limits_active_periods(self, optimize):
        """Proves: lifetime=1 means capacity is active only in the build period.

        3 periods, lifetime=1. Build must happen. Demand=10 in all periods.
        With lifetime=1, only 1 period has capacity → other 2 periods have no supply.
        Grid also has a backup with high cost (100/MWh) to satisfy demand.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Cheap',
                    imports=[
                        Flow(
                            'Heat',
                            short_id='cheap',
                            size=Investment(0, 20, lifetime=1, effects_per_size={'cost': 0}),
                            effects_per_flow_hour={'cost': 1},
                        ),
                    ],
                ),
                Port(
                    'Expensive',
                    imports=[
                        Flow('Heat', short_id='expensive', effects_per_flow_hour={'cost': 100}),
                    ],
                ),
            ],
            periods=[2020, 2025, 2030],
            period_weights=[5, 5, 5],
        )
        # Cheap is active for 1 period only (30 * 1 * 5 = 150)
        # Expensive covers 2 remaining periods (30 * 100 * 5 * 2 = 30000)
        # Total = 150 + 30000 = 30150
        assert_allclose(result.objective, 30150.0, rtol=1e-4)

    def test_investment_prior_size_available_from_start(self, optimize):
        """Proves: prior_size makes capacity available from period 0 without building.

        prior_size=10 means 10 MW is available from the start, no CAPEX.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(
                            'Heat',
                            size=Investment(
                                0,
                                20,
                                mandatory=False,
                                prior_size=10,
                                effects_per_size={'cost': 1000},
                            ),
                            effects_per_flow_hour={'cost': 1},
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        # With prior_size=10, no build needed. flow_size=10 in all periods.
        # Cost = 2 * 5 * 30 = 300 (operational only, no CAPEX)
        assert_allclose(result.objective, 300.0, rtol=1e-4)

    def test_investment_capex_charged_once(self, optimize):
        """Proves: effects_per_size goes to effect_once domain, not periodic.

        CAPEX = 10/MW, size=10 MW → one-time cost = 100 (unweighted).
        No operational costs. 2 periods, weights=[5, 5].
        Objective = 5*0 + 5*0 + 1*100 = 100  (once costs have ω_once=1 by default).
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(
                            'Heat',
                            size=Investment(10, 10, effects_per_size={'cost': 10}),
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        # min_size == max_size == 10, so invest_size = 10.
        # CAPEX: 10 * 10 = 100 (effect_once, ω_once=1 by default)
        # No flow costs. Objective = 100.
        assert_allclose(result.objective, 100.0, rtol=1e-4)

    def test_investment_periodic_costs_weighted(self, optimize):
        """Proves: effects_per_size_periodic goes to effect_periodic, scaled by period weights.

        Recurring O&M = 2/MW/period, size=10 MW. 2 periods, weights=[5, 5].
        Periodic cost per period = 2*10 = 20. Weighted: 5*20 + 5*20 = 200.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Heat')],
            effects=[Effect('cost', is_objective=True)],
            ports=[
                Port(
                    'Demand',
                    exports=[
                        Flow('Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    'Grid',
                    imports=[
                        Flow(
                            'Heat',
                            size=Investment(
                                10,
                                10,
                                effects_per_size_periodic={'cost': 2},
                            ),
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        # Periodic: 2/MW * 10 MW = 20 per period. Weighted: 5*20 + 5*20 = 200.
        assert_allclose(result.objective, 200.0, rtol=1e-4)
