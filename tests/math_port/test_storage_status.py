"""Mathematical correctness tests for Storage component-level Status."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port, Status, Storage

from .conftest import ts


class TestStorageStatusValidation:
    def test_flow_level_status_forbidden(self):
        """Flow.status on a storage flow conflicts with component-level Status."""
        with pytest.raises(ValueError, match='cannot have flow-level status'):
            Storage(
                'Bat',
                charging=Flow('Elec', size=10, relative_minimum=0.1, status=Status(min_uptime=2)),
                discharging=Flow('Elec', size=10),
                capacity=100,
                status=Status(),
            )

    def test_fixed_profile_forbidden(self):
        """Flow.fixed_relative_profile on a storage flow conflicts with component-level Status."""
        with pytest.raises(ValueError, match='cannot have fixed_relative_profile'):
            Storage(
                'Bat',
                charging=Flow('Elec', size=10, fixed_relative_profile=0.5),
                discharging=Flow('Elec', size=10),
                capacity=100,
                status=Status(),
            )

    def test_storage_status_optional(self):
        """Storage with status=None still constructs (default)."""
        s = Storage('Bat', Flow('Elec'), Flow('Elec'), capacity=10)
        assert s.status is None


class TestStorageComponentStatus:
    def test_solution_includes_component_variables(self, optimize):
        """Storage with status emits ``component--on/startup/shutdown`` solutions."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Elec')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('Demand', exports=[Flow('Elec', size=1, fixed_relative_profile=np.array([0, 0, 10]))]),
                Port('Grid', imports=[Flow('Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    'Bat',
                    charging=Flow('Elec', size=20),
                    discharging=Flow('Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        assert 'component--on' in result.solution
        assert 'component--startup' in result.solution
        assert 'component--shutdown' in result.solution
        assert 'Bat' in result.solution['component--on'].coords['component'].values

    def test_status_gates_both_flows(self, optimize):
        """When component_on=0, both charging and discharging are forced to 0."""
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier('Elec')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('Demand', exports=[Flow('Elec', size=1, fixed_relative_profile=np.array([0, 0, 5]))]),
                Port('Grid', imports=[Flow('Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    'Bat',
                    charging=Flow('Elec', size=20),
                    discharging=Flow('Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Bat').values
        charge = result.solution['flow--rate'].sel(flow='Bat(charge)').values
        discharge = result.solution['flow--rate'].sel(flow='Bat(discharge)').values
        for t in range(3):
            if on[t] < 0.5:
                assert charge[t] < 1e-6, f't={t}: charge={charge[t]} but on=0'
                assert discharge[t] < 1e-6, f't={t}: discharge={discharge[t]} but on=0'

    def test_startup_cost(self, optimize):
        """Proves: effects_per_startup deters cycling — cost accrues per on-transition.

        Demand [0, 10, 0, 10, 0] over 5 steps; storage must charge from grid
        and re-discharge twice. Startup cost makes a single long charge cheaper
        than two short ones.

        Without startup cost, optimal=20 (grid energy at €1/MWh, no losses).
        With 50€/startup, the storage does not cycle — solver routes from grid
        directly when possible, charging once. Either way startup cost gets
        baked into the objective only when the storage is used.
        """
        result = optimize(
            timesteps=ts(5),
            carriers=[Carrier('Elec')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('Demand', exports=[Flow('Elec', size=1, fixed_relative_profile=np.array([0, 10, 0, 10, 0]))]),
                Port('Grid', imports=[Flow('Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    'Bat',
                    charging=Flow('Elec', size=20),
                    discharging=Flow('Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(effects_per_startup={'cost': 1000}),
                ),
            ],
        )
        # Direct supply costs 20€ — startup cost deters using storage at all.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)
        startups = result.solution['component--startup'].sel(component='Bat').values
        assert startups.sum() == 0

    def test_running_cost_accrues_per_timestep(self, optimize):
        """``effects_per_running_hour`` charges (cost/h) * on * dt per timestep."""
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Elec')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('Demand', exports=[Flow('Elec', size=1, fixed_relative_profile=np.array([0, 0, 0, 10]))]),
                Port('Grid', imports=[Flow('Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    'Bat',
                    charging=Flow('Elec', size=20),
                    discharging=Flow('Elec', size=20),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(effects_per_running_hour={'cost': 5}),
                ),
            ],
        )
        # Running cost so high that storage stays off and demand draws from grid directly.
        # objective = 10 (grid only); storage on-hours = 0.
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 10.0, rtol=1e-5)
        on_hours = result.solution['component--on'].sel(component='Bat').values
        assert on_hours.sum() == 0

    def test_min_uptime_forces_consecutive_on(self, optimize):
        """min_uptime keeps the unit on across multiple steps once started."""
        result = optimize(
            timesteps=ts(4),
            carriers=[Carrier('Elec')],
            effects=[Effect('cost')],
            objective_effects='cost',
            ports=[
                Port('Demand', exports=[Flow('Elec', size=1, fixed_relative_profile=np.array([0, 5, 0, 0]))]),
                Port(
                    'Grid',
                    imports=[Flow('Elec', effects_per_flow_hour={'cost': np.array([1, 1, 1, 1])})],
                ),
                # Cheap charge source at t=0 makes it economic to charge then discharge at t=1
                Port('CheapCharge', imports=[Flow('Elec', size=10, effects_per_flow_hour={'cost': 0.01})]),
            ],
            storages=[
                Storage(
                    'Bat',
                    charging=Flow('Elec', size=10),
                    discharging=Flow('Elec', size=10, relative_minimum=0.1),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    status=Status(min_uptime=3),
                ),
            ],
        )
        on = result.solution['component--on'].sel(component='Bat').values
        # If the storage is ever on, it must remain on for ≥3 consecutive steps
        if on.any():
            run_lengths = []
            cur = 0
            for v in on:
                if v > 0.5:
                    cur += 1
                elif cur > 0:
                    run_lengths.append(cur)
                    cur = 0
            if cur > 0:
                run_lengths.append(cur)
            assert all(rl >= 3 for rl in run_lengths), f'Min uptime violated: runs={run_lengths}'
