"""Mathematical correctness tests for multi-period optimization."""

import numpy as np
import pytest
from conftest import ts
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port


class TestMultiPeriod:
    def test_period_weights_affect_objective(self, optimize):
        """Proves: period weights scale per-period costs in the objective.

        3 timesteps, periods=[2020, 2025], weight_of_last_period=5.
        Weights = [5, 5] (2025-2020=5, last=5).
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
            weight_of_last_period=5,
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

    @pytest.mark.skip(reason='multi-period per-effect period weights not yet implemented')
    def test_effect_period_weights(self, optimize):
        """Proves: Effect.period_weights overrides default period weights."""

    @pytest.mark.skip(reason='multi-period storage constraints not yet implemented')
    def test_storage_relative_minimum_final_level_scalar(self, optimize):
        """Proves: scalar relative_minimum_final_level works in multi-period."""

    @pytest.mark.skip(reason='multi-period storage constraints not yet implemented')
    def test_storage_relative_maximum_final_level_scalar(self, optimize):
        """Proves: scalar relative_maximum_final_level works in multi-period."""
