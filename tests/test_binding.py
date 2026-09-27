"""What binding a system checks, on systems that exercise every feature.

The engine comparisons that used to live here ran the program on lpspec's
eager lane as an oracle; that lane is not part of specsolve, so the math
suites (`tests/math`, `tests/math_port`) are the gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import read, solve_data

from fluxopt import (
    Carrier,
    Converter,
    Effect,
    Flow,
    Investment,
    ModelData,
    PiecewiseConversion,
    Port,
    Sizing,
    Status,
    Storage,
)
from fluxopt.math import UnsupportedFeatureError, build_sources

OBJECTIVE = {'cost': 1.0}
#: Generous: the short horizons used here have no co2 slack, so a tighter cap
#: would be infeasible rather than binding.
CO2_CAP = 1.0e6


def _system(n: int, periods: list[int] | None = None, co2_cap: float | None = CO2_CAP) -> dict:
    """A system exercising every feature the program expresses."""
    rng = np.random.default_rng(0)
    hours = np.arange(n)
    demand = np.clip(0.5 + 0.3 * np.cos(2 * np.pi * hours / 24) + 0.05 * rng.standard_normal(n), 0.05, 1.0)
    gas_price = 30 + 5 * np.sin(2 * np.pi * hours / 168)
    elec_price = 60 + 25 * np.sin(2 * np.pi * hours / 24) + 3 * rng.standard_normal(n)
    return {
        **({'periods': periods} if periods else {}),
        'timesteps': pd.date_range('2025-01-01', periods=n, freq='h'),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='heat'), Carrier(id='ambient')],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': 45.0}),
            Effect(id='co2', unit='kg', total_max=co2_cap),
        ],
        'ports': [
            Port(
                id='gas_grid',
                imports=[
                    Flow(carrier='gas', size=60.0, effects_per_flow_hour={'cost': gas_price.tolist(), 'co2': 0.2})
                ],
            ),
            Port(
                id='power_exchange',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=30.0,
                        effects_per_flow_hour={'cost': elec_price.tolist(), 'co2': 0.35},
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=30.0,
                        effects_per_flow_hour={'cost': (-0.9 * elec_price).tolist()},
                    )
                ],
            ),
            Port(id='ambient_air', imports=[Flow(carrier='ambient', size=1e6)]),
            Port(
                id='heat_network',
                exports=[Flow(carrier='heat', size=20.0, fixed_relative_profile=demand.tolist())],
            ),
        ],
        'converters': [
            Converter.boiler(
                'gas_boiler',
                0.92,
                Flow(carrier='gas'),
                Flow(carrier='heat', size=Sizing(size_min=2.0, size_max=15.0, effects_per_size={'cost': 240.0})),
            ),
            Converter.heat_pump(
                'heat_pump',
                3.2,
                Flow(carrier='elec'),
                Flow(carrier='ambient', size=1e6),
                Flow(
                    carrier='heat',
                    size=Sizing(
                        size_min=1.0,
                        size_max=8.0,
                        mandatory=False,
                        effects_per_size={'cost': 400.0},
                        effects_fixed={'cost': 5000.0},
                    ),
                ),
            ),
            Converter.chp(
                'chp',
                0.38,
                0.45,
                Flow(
                    carrier='gas',
                    size=Sizing(
                        size_min=8.0,
                        size_max=25.0,
                        mandatory=False,
                        effects_per_size={'cost': 180.0},
                        effects_fixed={'cost': 3000.0},
                    ),
                    relative_rate_min=0.4,
                    ramp_up_per_hour=0.3,
                    ramp_down_per_hour=0.25,
                    status=Status(
                        uptime_min=24.0,
                        downtime_min=6.0,
                        effects_per_startup={'cost': 900.0},
                        effects_per_running_hour={'cost': 12.0},
                    ),
                    prior_rates=[18.0, 18.0, 18.0],
                ),
                Flow(carrier='elec'),
                Flow(carrier='heat'),
            ),
        ],
        'storages': [
            Storage(
                id='tank',
                charging=Flow(carrier='heat', size=10.0),
                discharging=Flow(carrier='heat', size=10.0),
                capacity=Sizing(
                    size_min=10.0,
                    size_max=200.0,
                    mandatory=False,
                    effects_per_size={'cost': 55.0},
                    effects_fixed={'cost': 1200.0},
                ),
                relative_loss_per_hour=0.003,
                final_level_min=20.0,
                prevent_simultaneous=True,
            ),
        ],
    }


def test_mandatory_storage_sizing_binds_the_right_dims() -> None:
    """An absent lump term must still be keyed by its own entity dim.

    A storage sized with `mandatory=True` has no build indicator, so the
    `cap_ind_coeff` table is empty — and an empty table keyed by `flow`
    instead of `storage` fails to bind. Regression for a real benchmark
    system (`green_city`).
    """
    elements = _system(24)
    elements['storages'] = [
        Storage(
            id='tank',
            charging=Flow(carrier='heat', size=10.0),
            discharging=Flow(carrier='heat', size=10.0),
            capacity=Sizing(size_min=10.0, size_max=200.0, effects_per_size={'cost': 55.0}),
            relative_loss_per_hour=0.003,
        )
    ]
    data = ModelData.build(**elements)
    result = solve_data(data, OBJECTIVE)

    # Binding is the assertion: a mis-keyed empty table fails before a number
    # is ever produced, so an answer at all is the regression check.
    assert result.objective > 0
    assert float(read(result, 'chosen_capacity').sel(storage='tank')) >= 10.0


def test_sparse_coefficients_are_not_materialised() -> None:
    """`effect_coeff` is declared dense but only live rows are emitted."""
    data = ModelData.build(**_system(48))
    sources, coords = build_sources(data, OBJECTIVE)

    dense = len(coords['flow']) * len(coords['effect']) * len(coords['time']) * len(coords['period'])
    assert len(sources['effects_per_flow_hour']) < dense / 2


def test_piecewise_lp_method_raises_rather_than_answering_differently() -> None:
    """`method='lp'` is a relaxation this lane has no formulation for."""
    elements = _system(24)
    elements['converters'] = [
        Converter(
            id='pw_boiler',
            inputs=[Flow(carrier='gas', size=100.0)],
            outputs=[Flow(carrier='heat', size=70.0)],
            conversion=PiecewiseConversion(points=[('gas', [0, 50, 100]), ('heat', [0, 45, 70], '<=')], method='lp'),
        )
    ]
    data = ModelData.build(**elements)

    with pytest.raises(UnsupportedFeatureError, match='lp'):
        build_sources(data, OBJECTIVE)


def test_investment_requires_periods() -> None:
    """Investment is period-timed; without periods it must not build silently."""
    elements = _system(24)
    elements['converters'] = [
        Converter.boiler(
            'gas_boiler',
            0.92,
            Flow(carrier='gas'),
            Flow(carrier='heat', size=Investment(size_min=1.0, size_max=10.0, lifetime=20)),
        )
    ]
    data = ModelData.build(**elements)

    with pytest.raises(UnsupportedFeatureError, match='multi-period'):
        build_sources(data, OBJECTIVE)
