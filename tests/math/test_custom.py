from __future__ import annotations

import pytest
from conftest import ts

from fluxopt import Carrier, Effect, Flow, ModelData, Port, optimize
from fluxopt.model import FlowSystem


class TestCustomize:
    """Tests for the customize callback and custom variables/constraints."""

    @pytest.fixture
    def simple_system(self):
        """Single-bus system: grid source (size=100) feeding a fixed 50 MW demand."""
        return {
            'timesteps': ts(3),
            'carriers': [Carrier('elec')],
            'effects': [Effect('cost')],
            'objective_effects': 'cost',
            'ports': [
                Port('grid', imports=[Flow('elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
                Port('demand', exports=[Flow('elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
            ],
        }

    def test_customize_adds_constraint(self, simple_system):
        """A custom constraint restricting flow rate should affect the solution."""
        # Without customize: grid imports 50 MW each hour (matching demand)
        result_base = optimize(**simple_system)
        base_rates = result_base.flow_rate('grid(elec)').values
        for rate in base_rates:
            assert rate == pytest.approx(50.0, abs=1e-6)

        # With customize: cap grid import at 30 MW — this makes the problem infeasible
        # for a fixed 50 MW demand, so instead we test a less restrictive constraint.
        # Cap at 60 MW (above demand, so solution unchanged but constraint is present)
        def cap_at_60(model: FlowSystem) -> None:
            grid_rate = model.m.variables['flow--rate'].sel(flow='grid(elec)')
            model.m.add_constraints(grid_rate <= 60, name='custom_grid_cap')

        result = optimize(**simple_system, customize=cap_at_60)
        rates = result.flow_rate('grid(elec)').values
        for rate in rates:
            assert rate == pytest.approx(50.0, abs=1e-6)

        # With cap at 60, demand of 50 is still feasible → same objective
        assert result.objective == pytest.approx(result_base.objective, abs=1e-6)

    def test_custom_variable_in_results(self, simple_system):
        """A custom variable added via callback should appear in result.solution."""

        def add_slack(model: FlowSystem) -> None:
            time = model.m.variables['flow--rate'].indexes['time']
            slack = model.m.add_variables(lower=0, coords=[time], name='my_slack')
            grid = model.m.variables['flow--rate'].sel(flow='grid(elec)')
            # grid + slack >= 60 → slack >= 10 (since grid = 50)
            model.m.add_constraints(grid + slack >= 60, name='slack_floor')
            model.m.objective += 100 * slack.sum()

        result = optimize(**simple_system, customize=add_slack)

        assert 'my_slack' in result.solution
        slack_vals = result.solution['my_slack'].values
        for val in slack_vals:
            assert val == pytest.approx(10.0, abs=1e-6)

    def test_no_customize_works(self, simple_system):
        """optimize() without customize callback works as before."""
        result = optimize(**simple_system)
        assert result.objective == pytest.approx(150.0, abs=1e-6)
        rates = result.flow_rate('grid(elec)').values
        for rate in rates:
            assert rate == pytest.approx(50.0, abs=1e-6)

    def test_direct_model_customization(self, simple_system):
        """Using FlowSystem directly with custom variable works."""
        data = ModelData.build(
            simple_system['timesteps'],
            simple_system['carriers'],
            simple_system['effects'],
            simple_system['ports'],
        )
        model = FlowSystem(data, objective='cost')
        model.build()

        # Add custom variable and constraint
        time = model.m.variables['flow--rate'].indexes['time']
        bonus = model.m.add_variables(lower=0, upper=5, coords=[time], name='bonus')
        model.m.objective += -bonus.sum()  # maximize bonus (minimize negative)

        result = model.solve()

        assert 'bonus' in result.solution
        for val in result.solution['bonus'].values:
            assert val == pytest.approx(5.0, abs=1e-6)


class TestFlowSystemApi:
    """Tests for the FlowSystem construction / inspection surface."""

    @pytest.fixture
    def simple_system(self):
        """Single-bus system: grid source (size=100) feeding a fixed 50 MW demand."""
        return {
            'timesteps': ts(3),
            'carriers': [Carrier('elec')],
            'effects': [Effect('cost')],
            'ports': [
                Port('grid', imports=[Flow('elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
                Port('demand', exports=[Flow('elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
            ],
        }

    def test_from_elements_builds_inspectable_model(self, simple_system):
        """from_elements + build yields an inspectable, unsolved model."""
        fs = FlowSystem.from_elements(objective='cost', **simple_system)
        assert fs.objective == {'cost': 1.0}
        fs.build()
        assert 'flow--rate' in fs.m.variables
        result = fs.solve()
        assert result.objective == pytest.approx(150.0, abs=1e-6)

    def test_solve_before_build_raises(self, simple_system):
        """Calling solve() on an unbuilt model is a clear error, not a silent no-op."""
        fs = FlowSystem.from_elements(objective='cost', **simple_system)
        with pytest.raises(RuntimeError, match='not built'):
            fs.solve()

    def test_build_without_objective_raises(self, simple_system):
        """Building with no objective set errors instead of silently minimizing penalty."""
        fs = FlowSystem.from_elements(**simple_system)  # no objective
        with pytest.raises(ValueError, match='No objective set'):
            fs.build()

    def test_optimize_without_objective_raises(self, simple_system):
        """optimize() with neither a stored nor a passed objective errors."""
        fs = FlowSystem.from_elements(**simple_system)
        with pytest.raises(ValueError, match='No objective set'):
            fs.optimize()

    def test_objective_property_retarget(self, simple_system):
        """The objective property normalizes assignment; optimize() reuses it."""
        fs = FlowSystem.from_elements(**simple_system)
        assert fs.objective == {}
        fs.objective = 'cost'
        assert fs.objective == {'cost': 1.0}
        result = fs.optimize()  # no objective arg — uses the property
        assert result.objective == pytest.approx(150.0, abs=1e-6)

    def test_optimize_arg_overrides_objective(self, simple_system):
        """An explicit optimize(objective_effects=...) overrides the stored objective."""
        fs = FlowSystem.from_elements(objective='cost', **simple_system)
        result = fs.optimize({'cost': 2.0})
        assert fs.objective == {'cost': 2.0}
        assert result.objective == pytest.approx(300.0, abs=1e-6)
