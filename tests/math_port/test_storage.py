"""Mathematical correctness tests for storage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fluxopt import Carrier, Effect, Flow, Port, Sizing, Storage

from .conftest import ts


class TestStorage:
    def test_storage_shift_saves_money(self, optimize):
        """Proves: Storage enables temporal arbitrage — charge cheap, discharge when expensive.

        Sensitivity: Without storage, demand at t=2 must be bought at 10€/kWh → cost=200.
        With working storage, buy at t=1 for 1€/kWh → cost=20. A 10x difference.
        """
        result = optimize(
            timesteps=ts(3),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 0, 20])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Elec', effects_per_flow_hour={'cost': np.array([10, 1, 10])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=100),
                    discharging=Flow(carrier='Elec', size=100),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)

    def test_storage_losses(self, optimize):
        """Proves: relative_loss_per_hour correctly reduces stored energy over time.

        Sensitivity: If losses were ignored (0%), only 90 would be charged → cost=90.
        With 10% loss, must charge 100 to have 90 after 1h → cost=100.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 90])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Elec', effects_per_flow_hour={'cost': np.array([1, 1000])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=200,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0.1,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_storage_eta_charge_discharge(self, optimize):
        """Proves: eta_charge and eta_discharge are both applied to the energy flow.

        Sensitivity: If eta_charge broken (1.0), cost=90. If eta_discharge broken (1.0),
        cost=80. If both broken, cost=72. Only both correct yields cost=100.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 72])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Elec', effects_per_flow_hour={'cost': np.array([1, 1000])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=200,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=0.9,
                    eta_discharge=0.8,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    def test_storage_soc_bounds(self, optimize):
        """Proves: relative_level_max caps how much energy can be stored.

        Sensitivity: If level bound were ignored, all 60 stored cheaply → cost=60.
        With the bound enforced, cost=1050 (50*1 + 10*100).
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 60])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Elec', effects_per_flow_hour={'cost': np.array([1, 100])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    relative_level_max=0.5,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 1050.0, rtol=1e-5)

    def test_storage_invest_capacity(self, optimize):
        """Proves: Sizing on capacity correctly sizes the storage.

        Sensitivity: If invest cost were 100€/kWh (>9 saving), no storage built → cost=500.
        At 1€/kWh, storage built → cost=50*1 (buy) + 50*1 (invest) = 100.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(
                    id='Demand',
                    exports=[
                        Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([0, 50])),
                    ],
                ),
                Port(
                    id='Grid',
                    imports=[
                        Flow(carrier='Elec', effects_per_flow_hour={'cost': np.array([1, 10])}),
                    ],
                ),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=200),
                    discharging=Flow(carrier='Elec', size=200),
                    capacity=Sizing(size_min=0, size_max=200, mandatory=False, effects_per_size={'cost': 1}),
                    prior_level=0,
                    cyclic=False,
                    eta_charge=1,
                    eta_discharge=1,
                    relative_loss_per_hour=0,
                ),
            ],
        )
        assert_allclose(result.storage_capacities.sel(storage='Battery').item(), 50.0, rtol=1e-5)
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 100.0, rtol=1e-5)

    @pytest.mark.skip(reason='prior_level is absolute in fluxopt, not relative')
    def test_storage_relative_level_min(self, optimize):
        """Proves: relative_level_min enforces a minimum SOC at all times.

        Sensitivity: Without min SOC, discharge all 100 → no grid → cost=50.
        With min SOC=0.3, max discharge=70 → grid covers 10 @100€ → cost=1050.
        """
        raise NotImplementedError  # TODO: rewrite for fluxopt API (prior_level is absolute)

    @pytest.mark.skip(reason='flixopt-specific legacy test')
    def test_storage_cyclic_level(self, optimize):
        """Proves: cyclic=True forces the storage to end at the
        same level it started, preventing free energy extraction."""
        raise NotImplementedError  # TODO: rewrite for fluxopt API (cyclic is a bool)

    def test_storage_minimal_final_level(self, optimize):
        """Proves: final_level_min forces the storage to retain at least the
        specified absolute energy at the end.

        Battery (prior=0, cyclic=False, final_level_min=50). Demand=[10,0].
        Grid must supply demand (10) plus the mandated final stock (50).

        Sensitivity: Without final_level_min, only demand is bought → cost=10.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([10, 0]))]),
                Port(id='Grid', imports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=100),
                    discharging=Flow(carrier='Elec', size=100),
                    capacity=100,
                    prior_level=0,
                    cyclic=False,
                    final_level_min=50,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 60.0, rtol=1e-5)

    def test_prevent_simultaneous_charge_and_discharge(self, optimize):
        """Proves: prevent_simultaneous=True forbids charging and discharging
        in the same timestep.

        Must-run production=[30,30], no free sink; Battery capacity=20 with
        eta_charge=0.5 destroys energy when cycled. Dump costs 1€/kWh.

        Without prevention the battery burns the t1 surplus for free by
        charging and discharging at once (charge 50 = production 30 +
        discharge 20; stored 15+25-20=20) → dump=0, cost=0.
        With prevention, t1 can only charge up to 10 (level 15+5=20)
        → 20 must be dumped → cost=20.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='MustRun', imports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([30, 30]))]),
                Port(id='Dump', exports=[Flow(carrier='Elec', effects_per_flow_hour={'cost': 1})]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=100),
                    discharging=Flow(carrier='Elec', size=100),
                    capacity=20,
                    prior_level=0,
                    cyclic=False,
                    eta_charge=0.5,
                    prevent_simultaneous=True,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 20.0, rtol=1e-5)
        charge = result.flow_rate('Battery(charge)').values
        discharge = result.flow_rate('Battery(discharge)').values
        simul = (charge > 1e-5) & (discharge > 1e-5)
        assert not simul.any(), f'Simultaneous charge/discharge: charge={charge}, discharge={discharge}'

    def test_storage_maximal_final_level(self, optimize):
        """Proves: final_level_max caps the storage level at the end.

        Battery starts at prior_level=50 (free energy), discharge costs
        1€/kWh. Demand=[10,10]; final_level_max=10 forces the level from
        50 down to <=10, i.e. discharge >= 40 (20 beyond demand, wasted).

        Sensitivity: Without final_level_max, discharge=20 → cost=20.
        """
        result = optimize(
            timesteps=ts(2),
            carriers=[Carrier(id='Elec')],
            effects=[Effect(id='cost')],
            objective='cost',
            ports=[
                Port(id='Demand', exports=[Flow(carrier='Elec', size=1, fixed_relative_profile=np.array([10, 10]))]),
                Port(id='Dump', exports=[Flow(carrier='Elec')]),
            ],
            storages=[
                Storage(
                    id='Battery',
                    charging=Flow(carrier='Elec', size=100),
                    discharging=Flow(carrier='Elec', size=100, effects_per_flow_hour={'cost': 1}),
                    capacity=100,
                    prior_level=50,
                    cyclic=False,
                    final_level_max=10,
                ),
            ],
        )
        assert_allclose(result.effect_totals.sel(effect='cost').item(), 40.0, rtol=1e-5)

    @pytest.mark.skip(reason='relative final level not supported in fluxopt')
    def test_storage_relative_rate_min_final_level(self, optimize):
        """Proves: relative_rate_min_final_level forces a minimum final SOC
        as a fraction of capacity."""
        raise NotImplementedError  # TODO: implement relative final level constraint

    @pytest.mark.skip(reason='relative final level + imbalance penalty not supported in fluxopt')
    def test_storage_relative_rate_max_final_level(self, optimize):
        """Proves: relative_rate_max_final_level caps the storage at end
        as a fraction of capacity."""
        raise NotImplementedError  # TODO: implement relative final level constraint

    @pytest.mark.skip(reason='relative final level not supported in fluxopt')
    def test_storage_relative_rate_min_final_level_scalar(self, optimize):
        """Proves: relative_rate_min_final_level works when relative_level_min
        is a scalar (default=0, no time dimension)."""
        raise NotImplementedError  # TODO: implement relative final level constraint

    @pytest.mark.skip(reason='relative final level + imbalance penalty not supported in fluxopt')
    def test_storage_relative_rate_max_final_level_scalar(self, optimize):
        """Proves: relative_rate_max_final_level works when relative_level_max
        is a scalar (default=1, no time dimension)."""
        raise NotImplementedError  # TODO: implement relative final level constraint

    @pytest.mark.skip(reason='balanced invest not supported in fluxopt')
    def test_storage_balanced_invest(self, optimize):
        """Proves: balanced=True forces charge and discharge invest sizes to be equal."""
        raise NotImplementedError  # TODO: implement balanced invest on Storage
