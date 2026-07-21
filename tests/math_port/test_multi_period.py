"""Mathematical correctness tests for multi-period optimization."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from conftest import ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Investment, Port, Sizing, Status, Storage


class TestMultiPeriod:
    def test_period_weights_affect_objective(self, optimize):
        """Proves: period weights scale per-period costs in the objective.

        3 timesteps, periods=[2020, 2025], period_weights=[5, 5].
        Grid @1 cost/MWh, Demand=[10, 10, 10]. Per-period cost=30.
        Objective = 5*30 + 5*30 = 300.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 1}),
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

    def test_period_varying_demand_via_dataframe(self, optimize):
        """Proves: fixed_relative_profile accepts (time, period) DataFrame.

        Demand grows across periods: 10 MW in 2020, 20 MW in 2025.
        Grid cost = 1 [unit/MWh]. Period weights = [1, 1].
        Per-period cost: 2020→10*3=30, 2025→20*3=60. Total = 90.
        """
        timesteps = ts(3)
        time_idx = pd.DatetimeIndex(timesteps, name='time')
        periods = pd.Index([2020, 2025], name='period')
        demand = pd.DataFrame(
            np.array([[10, 20], [10, 20], [10, 20]], dtype=float),
            index=time_idx,
            columns=periods,
        )
        result = optimize(
            timesteps=timesteps,
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[Flow(carrier='Heat', size=1, fixed_relative_profile=demand)],
                ),
                Port(
                    id='Grid',
                    imports=[Flow(carrier='Heat', effects_per_flow_hour={'cost': 1})],
                ),
            ],
            periods=list(periods),
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 90.0, rtol=1e-5)

    def test_period_varying_effects_per_flow_hour(self, optimize):
        """Proves: effects_per_flow_hour with period dim produces period-specific costs.

        3 timesteps, periods=[2020, 2025], weights=[1, 1].
        Demand=10 constant. Grid cost varies: 1 in 2020, 3 in 2025.
        Per-period cost: 2020→10*3*1=30, 2025→10*3*3=90.
        Objective = 1*30 + 1*90 = 120.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([1.0, 3.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': cost_by_period}),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 120.0, rtol=1e-5)

    @pytest.mark.skip(reason='multi-period linked periods not yet implemented')
    def test_invest_linked_periods(self, optimize):
        """Proves: InvestParameters.linked_periods forces equal sizes across periods."""

    def test_effect_period_weights(self, optimize):
        """Proves: Effect.period_weights overrides global period weights.

        3 timesteps, periods=[2020, 2025], global weights=[5, 5].
        Per-period cost = 30. With custom period_weights [1, 2]:
        Objective = 1*30 + 2*30 = 90  (instead of 5*30 + 5*30 = 300).
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='cost', period_weights=[1, 2]),
            ],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'cost': 1}),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        assert_allclose(result.objective, 90.0, rtol=1e-5)

    @pytest.mark.skip(reason='multi-period storage constraints not yet implemented')
    def test_storage_relative_rate_min_final_level_scalar(self, optimize):
        """Proves: scalar relative_rate_min_final_level works in multi-period."""

    @pytest.mark.skip(reason='multi-period storage constraints not yet implemented')
    def test_storage_relative_rate_max_final_level_scalar(self, optimize):
        """Proves: scalar relative_rate_max_final_level works in multi-period."""


class TestInvestment:
    def test_investment_mandatory_builds_once(self, optimize):
        """Proves: mandatory Investment builds exactly once across periods.

        3 timesteps, 3 periods [2020, 2025, 2030], weights=[5, 5, 5].
        Grid with Investment(0, 20, mandatory=True), CAPEX=10/MW at build.
        Demand=10 MW constant. Cheapest: build in first period, CAPEX=10*10=100.
        Operational cost per period: 10*3*1=30.
        total[p=0]=130, total[p=1]=30, total[p=2]=30.
        Objective = 5*130 + 5*30 + 5*30 = 950.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(size_min=0, size_max=20, effects_per_size_at_build={'cost': 10}),
                            effects_per_flow_hour={'cost': 1},
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025, 2030],
            period_weights=[5, 5, 5],
        )
        # CAPEX: 10/MW * 10 MW = 100 in build period (p=0)
        # Operational: 10 MW * 3h * 1 cost/MWh = 30 per period
        # total[p=0]=130, total[p=1]=30, total[p=2]=30
        # Objective = 5*130 + 5*30 + 5*30 = 950
        assert_allclose(result.objective, 950.0, rtol=1e-4)

    def test_investment_optional_skips_when_expensive(self, optimize):
        """Proves: optional Investment is not built if cost exceeds benefit.

        Demand=0, so building would only add cost. Optional investment should not build.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([0, 0, 0])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(
                                size_min=0, size_max=20, mandatory=False, effects_per_size_at_build={'cost': 100}
                            ),
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
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Cheap',
                    imports=[
                        Flow(
                            carrier='Heat',
                            short_id='cheap',
                            size=Investment(size_min=0, size_max=20, lifetime=1, effects_per_size_at_build={'cost': 0}),
                            effects_per_flow_hour={'cost': 1},
                        ),
                    ],
                ),
                Port(
                    id='Expensive',
                    imports=[
                        Flow(carrier='Heat', short_id='expensive', effects_per_flow_hour={'cost': 100}),
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
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(
                                size_min=0,
                                size_max=20,
                                mandatory=False,
                                prior_size=10,
                                effects_per_size_at_build={'cost': 1000},
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

    def test_investment_capex_in_lump(self, optimize):
        """Proves: effects_per_size_at_build goes to lump domain, weighted like all lump costs.

        CAPEX = 10/MW, size=10 MW → lump cost = 100 per build period.
        No operational costs. 2 periods, weights=[5, 5].
        Mandatory build → built in period 0. Lump = 100 in p=0, 0 in p=1.
        Objective = 5*100 + 5*0 = 500.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(size_min=10, size_max=10, effects_per_size_at_build={'cost': 10}),
                        ),
                    ],
                ),
            ],
            periods=[2020, 2025],
            period_weights=[5, 5],
        )
        # size_min == size_max == 10, so invest_size = 10.
        # CAPEX: 10 * 10 = 100 in build period. Weighted: 5 * 100 = 500.
        assert_allclose(result.objective, 500.0, rtol=1e-4)

    def test_investment_periodic_costs_weighted(self, optimize):
        """Proves: effects_per_size_recurring goes to effect_periodic, scaled by period weights.

        Recurring O&M = 2/MW/period, size=10 MW. 2 periods, weights=[5, 5].
        Periodic cost per period = 2*10 = 20. Weighted: 5*20 + 5*20 = 200.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(
                                size_min=10,
                                size_max=10,
                                effects_per_size_recurring={'cost': 2},
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


class TestPeriodVaryingEffects:
    def test_sizing_effects_per_size_vary_by_period(self, optimize):
        """Proves: Sizing.effects_per_size can vary across periods.

        Grid with Sizing(10, 10), effects_per_size varies: 1 in 2020, 3 in 2025.
        Size is fixed at 10. Periodic cost = effects_per_size * size.
        2020: 1*10=10, 2025: 3*10=30. Weights=[1, 1]. Objective = 10 + 30 = 40.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([1.0, 3.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=10, size_max=10, effects_per_size={'cost': cost_by_period}),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 40.0, rtol=1e-4)

    def test_sizing_effects_fixed_vary_by_period(self, optimize):
        """Proves: Sizing.effects_fixed can vary across periods.

        Grid with Sizing(10, 10), effects_fixed varies: 5 in 2020, 15 in 2025.
        Fixed cost is independent of size. Weights=[1, 1].
        Objective = 5 + 15 = 20.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([5.0, 15.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Sizing(size_min=10, size_max=10, effects_fixed={'cost': cost_by_period}),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 20.0, rtol=1e-4)

    def test_investment_periodic_costs_vary_by_period(self, optimize):
        """Proves: Investment.effects_per_size_recurring can vary across periods.

        Investment(10, 10), recurring O&M varies: 1 in 2020, 3 in 2025.
        Active in both periods. Periodic cost = O&M * size.
        2020: 1*10=10, 2025: 3*10=30. Weights=[1, 1]. Objective = 10 + 30 = 40.
        """
        periods = [2020, 2025]
        om_by_period = xr.DataArray([1.0, 3.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(
                                size_min=10, size_max=10, effects_per_size_recurring={'cost': om_by_period}
                            ),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 40.0, rtol=1e-4)

    def test_investment_fixed_periodic_costs_vary_by_period(self, optimize):
        """Proves: Investment.effects_fixed_recurring can vary across periods.

        Investment(10, 10), fixed periodic cost varies: 5 in 2020, 15 in 2025.
        Active in both periods. Weights=[1, 1]. Objective = 5 + 15 = 20.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([5.0, 15.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(size_min=10, size_max=10, effects_fixed_recurring={'cost': cost_by_period}),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 20.0, rtol=1e-4)

    def test_investment_capex_per_size_varies_by_period(self, optimize):
        """Proves: Investment.effects_per_size_at_build (once) can vary across periods.

        Investment(10, 10), CAPEX varies: 10 in 2020, 20 in 2025.
        Mandatory build → builds in cheapest period (2020). Once cost = 10*10 = 100.
        Weights=[1, 1]. Objective = 100.
        """
        periods = [2020, 2025]
        capex_by_period = xr.DataArray([10.0, 20.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(
                                size_min=10, size_max=10, effects_per_size_at_build={'cost': capex_by_period}
                            ),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 100.0, rtol=1e-4)

    def test_investment_capex_fixed_varies_by_period(self, optimize):
        """Proves: Investment.effects_fixed_at_build (once) can vary across periods.

        Investment(10, 10), fixed CAPEX varies: 50 in 2020, 100 in 2025.
        Mandatory build → builds in cheapest period (2020). Once cost = 50.
        Weights=[1, 1]. Objective = 50.
        """
        periods = [2020, 2025]
        capex_by_period = xr.DataArray([50.0, 100.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=Investment(size_min=10, size_max=10, effects_fixed_at_build={'cost': capex_by_period}),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 50.0, rtol=1e-4)

    def test_status_running_cost_varies_by_period(self, optimize):
        """Proves: Status.effects_per_running_hour can vary across periods.

        Flow with status, demand=10 forces flow on for all 3 timesteps.
        Running cost: 1 in 2020, 3 in 2025. dt=1h.
        Running cost: 2020→1*3=3, 2025→3*3=9.
        Weights=[1, 1]. Objective = 3 + 9 = 12.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([1.0, 3.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=10,
                            relative_rate_min=0.5,
                            status=Status(effects_per_running_hour={'cost': cost_by_period}),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 12.0, rtol=1e-4)

    def test_status_startup_cost_varies_by_period(self, optimize):
        """Proves: Status.effects_per_startup can vary across periods.

        Demand profile [10, 0, 10] forces off→on at t=2 (1 startup per period).
        Startup cost: 100 in 2020, 300 in 2025.
        Weights=[1, 1]. Objective = 100 + 300 = 400.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([100.0, 300.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 0, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(
                            carrier='Heat',
                            size=10,
                            relative_rate_min=0.5,
                            status=Status(effects_per_startup={'cost': cost_by_period}),
                        ),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 400.0, rtol=1e-4)

    def test_contribution_from_varies_by_period(self, optimize):
        """Proves: Effect.contribution_from can vary across periods.

        CO2 effect tracks emissions. Cost effect gets contribution_from CO2
        at rate 50 in 2020, 100 in 2025 (rising carbon price).
        Grid emits 1 CO2/MWh, demand=10 for 3 timesteps.
        CO2 per period = 30. Cost from CO2: 2020→50*30=1500, 2025→100*30=3000.
        Weights=[1, 1]. Objective = 1500 + 3000 = 4500.
        """
        periods = [2020, 2025]
        carbon_price = xr.DataArray([50.0, 100.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[
                Effect(id='co2'),
                Effect(id='cost', contribution_from={'co2': carbon_price}),
            ],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([10, 10, 10])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat', effects_per_flow_hour={'co2': 1}),
                    ],
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 4500.0, rtol=1e-4)

    def test_storage_sizing_effects_per_size_vary_by_period(self, optimize):
        """Proves: Storage Sizing.effects_per_size can vary across periods.

        Storage with capacity=Sizing(10, 10), effects_per_size varies: 1 in 2020, 3 in 2025.
        Capacity fixed at 10. Periodic sizing cost: 2020→1*10=10, 2025→3*10=30.
        Weights=[1, 1]. Objective = 10 + 30 = 40.
        """
        periods = [2020, 2025]
        cost_by_period = xr.DataArray([1.0, 3.0], dims=['period'], coords={'period': periods})
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Heat')],
            effects=[Effect(id='cost')],
            objective_effects='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Heat', size=1, fixed_relative_profile=np.array([5, 5, 5])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Heat'),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='store',
                    charging=Flow(carrier='Heat'),
                    discharging=Flow(carrier='Heat'),
                    capacity=Sizing(size_min=10, size_max=10, effects_per_size={'cost': cost_by_period}),
                ),
            ],
            periods=periods,
            period_weights=[1, 1],
        )
        assert_allclose(result.objective, 40.0, rtol=1e-4)
