"""User-runnable benchmark: build a few realistic energy systems, report speed and memory.

Run it against your installation to see how fast fluxopt's build pipeline
(Elements → ModelData → linopy model) is on your hardware::

    python -m fluxopt.benchmark                        # all systems, one hourly year
    python -m fluxopt.benchmark district_heating       # a single system
    python -m fluxopt.benchmark --timesteps 720        # one month instead of a year
    python -m fluxopt.benchmark --solve                # also time the HiGHS solve
    python -m fluxopt.benchmark --json                 # machine-readable output

The reference systems are realistic, readable models — constant and
time-varying data, several effects and cross-effect couplings — so the numbers
reflect real workloads and the builders double as examples:

- ``district_heating`` — municipal utility: gas boiler, ramp-limited CHP and
  a heat pump with a weather-driven COP feed a heat network backed by a
  hot-water tank; seasonal gas tariff, day-ahead electricity prices, CO2
  priced into cost.
- ``industry_park`` — factory site: a steam boiler fleet with on/off unit
  commitment, a gas-engine CHP with a piecewise part-load curve, investment
  decisions for an electrode boiler and a steam accumulator, and an annual
  CO2 cap; three-shift steam demand.
- ``green_city`` — sector-coupled city: wind (PPA with a contracted energy
  cap), rooftop PV and a grid connection supply a battery (sized by the
  optimizer) and two district-heating networks; cost, CO2 and primary-energy
  accounting.
- ``energy_transition`` — ``green_city`` planned over eight five-year
  investment periods: growing demand, a decarbonizing grid, a rising carbon
  price, and the battery as a multi-period ``Investment`` (15-year lifetime,
  capex learning curve, recurring O&M); ~2 million variables at the default
  horizon.

All data is deterministic (no randomness), and each system is built in a
fresh subprocess so peak memory is attributed per model. Memory is
whole-process peak RSS — the number that has to fit in your RAM; for
allocator-level profiles use pytest-benchmem on ``benchmark/test_reference.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta
from importlib.metadata import version
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

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
from fluxopt.model import FlowSystemModel

if TYPE_CHECKING:
    from collections.abc import Callable

Elements = dict[str, Any]

HOURS_PER_YEAR = 8760
CARBON_PRICE = 0.045
"""EUR per kg CO2 (45 EUR/t), fed into ``cost`` via ``Effect.contribution_from``."""

GAS_PRICE = 35.0
"""EUR per MWh of natural gas (flat supply tariff)."""

GAS_CO2 = 202.0
"""kg CO2 per MWh of natural gas burned."""


def _hourly_index(n: int) -> list[datetime]:
    """``n`` hourly timesteps starting Monday, 2024-01-01."""
    start = datetime(2024, 1, 1)
    return [start + timedelta(hours=i) for i in range(n)]


def _clock(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hour-of-day, day index and weekend mask for the hourly index."""
    t = np.arange(n)
    hour = t % 24
    day = t // 24
    weekend = (day % 7) >= 5
    return hour, day, weekend


def _winter(day: np.ndarray) -> np.ndarray:
    """Seasonal factor: 1 at mid-winter, 0 at mid-summer."""
    return 0.5 + 0.5 * np.cos(2 * np.pi * day / 365.0)


def _gas_price(n: int) -> np.ndarray:
    """Indexed gas tariff [EUR/MWh]: winter premium on a 30 EUR base."""
    _, day, _ = _clock(n)
    return 30.0 + 11.0 * _winter(day)


def _heat_demand(n: int) -> np.ndarray:
    """Relative heat load: seasonal base plus morning and evening peaks."""
    hour, day, _ = _clock(n)
    peaks = 0.12 * np.exp(-((hour - 7.0) ** 2) / 8.0) + 0.10 * np.exp(-((hour - 19.0) ** 2) / 12.0)
    return np.clip(0.12 + 0.6 * _winter(day) + peaks, 0.05, 1.0)


def _elec_price(n: int) -> np.ndarray:
    """Day-ahead electricity price [EUR/MWh]: peaks at 8 h and 19 h, a solar dip at midday, cheaper weekends."""
    hour, day, weekend = _clock(n)
    peaks = 18.0 * np.exp(-((hour - 8.0) ** 2) / 6.0) + 22.0 * np.exp(-((hour - 19.0) ** 2) / 8.0)
    solar_dip = 14.0 * np.exp(-((hour - 13.0) ** 2) / 10.0) * (1.0 - _winter(day))
    return 52.0 + 14.0 * _winter(day) - 8.0 * weekend + peaks - solar_dip


def _grid_co2(n: int) -> np.ndarray:
    """Grid CO2 intensity [kg/MWh]: higher in winter, dips at midday with solar."""
    hour, day, _ = _clock(n)
    solar_dip = 130.0 * np.exp(-((hour - 13.0) ** 2) / 12.0) * (1.0 - 0.7 * _winter(day))
    return np.clip(340.0 + 70.0 * _winter(day) - solar_dip, 90.0, 600.0)


def _heat_pump_cop(n: int) -> np.ndarray:
    """Air-source heat-pump COP from a seasonal + daily ambient-temperature curve."""
    hour, day, _ = _clock(n)
    temp = 11.0 - 11.0 * np.cos(2 * np.pi * (day - 15.0) / 365.0) + 3.0 * np.sin(2 * np.pi * (hour - 15.0) / 24.0)
    return np.clip(2.9 + 0.09 * temp, 1.6, 5.0)


def _solar(n: int) -> np.ndarray:
    """Relative PV availability: daylight bell, stronger in summer."""
    hour, day, _ = _clock(n)
    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.5) / 13.0))
    return daylight * (0.9 - 0.55 * _winter(day))


def _wind(n: int) -> np.ndarray:
    """Relative wind availability: overlapping weather fronts, windier in winter."""
    t = np.arange(n)
    fronts = 0.38 + 0.25 * np.sin(t / 9.3) + 0.2 * np.sin(t / 37.0 + 2.0) + 0.15 * np.sin(t / 171.0 + 1.0)
    return np.clip(fronts + 0.15 * _winter(t // 24), 0.0, 1.0)


def _steam_demand(n: int) -> np.ndarray:
    """Relative steam load: three-shift weekdays, reduced weekend crew."""
    hour, _, weekend = _clock(n)
    weekday_shift = np.where((hour >= 6) & (hour < 22), 0.85, 0.6)
    return np.where(weekend, 0.35, weekday_shift)


def _city_elec_demand(n: int) -> np.ndarray:
    """Relative city electricity load: business-hours peak, evening shoulder, weekend reduction."""
    hour, day, weekend = _clock(n)
    business = 0.25 * np.exp(-((hour - 11.0) ** 2) / 24.0) + 0.15 * np.exp(-((hour - 19.0) ** 2) / 10.0)
    return np.clip(0.45 + 0.08 * _winter(day) + business - 0.1 * weekend, 0.2, 1.0)


def district_heating(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Municipal district-heating utility.

    A 15 MW gas boiler, a gas CHP and an 8 MW air-source heat pump with a
    weather-driven COP feed a 20 MW-peak heat network backed by an 80 MWh
    hot-water tank. Gas is bought at an indexed tariff with a winter premium;
    electricity is bought and sold at a day-ahead price profile; every kg of
    CO2 — burned on site or embodied in grid power — is priced into cost at
    45 EUR/t.
    """
    n = timesteps
    price = _elec_price(n)
    return {
        'timesteps': _hourly_index(n),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='heat'), Carrier(id='ambient')],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': CARBON_PRICE}),
            Effect(id='co2', unit='kg'),
        ],
        'ports': [
            Port(
                id='gas_grid',
                imports=[
                    Flow(
                        carrier='gas', size=60.0, effects_per_flow_hour={'cost': _gas_price(n).tolist(), 'co2': GAS_CO2}
                    )
                ],
            ),
            Port(
                id='power_exchange',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=30.0,
                        effects_per_flow_hour={'cost': price.tolist(), 'co2': _grid_co2(n).tolist()},
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=30.0,
                        effects_per_flow_hour={'cost': (-0.95 * price).tolist()},
                    )
                ],
            ),
            Port(id='ambient_air', imports=[Flow(carrier='ambient', size=1e6)]),
            Port(
                id='heat_network',
                exports=[Flow(carrier='heat', size=20.0, fixed_relative_profile=_heat_demand(n).tolist())],
            ),
        ],
        'converters': [
            Converter.boiler('gas_boiler', 0.92, Flow(carrier='gas'), Flow(carrier='heat', size=15.0)),
            Converter.chp(
                'chp',
                0.38,
                0.45,
                Flow(carrier='gas', size=25.0, ramp_up_per_hour=0.4, ramp_down_per_hour=0.4),
                Flow(carrier='elec'),
                Flow(carrier='heat'),
            ),
            Converter.heat_pump(
                'heat_pump',
                _heat_pump_cop(n).tolist(),
                Flow(carrier='elec'),
                Flow(carrier='ambient', size=1e6),
                Flow(carrier='heat', size=8.0),
            ),
        ],
        'storages': [
            Storage(
                id='hot_water_tank',
                charging=Flow(carrier='heat', size=10.0),
                discharging=Flow(carrier='heat', size=10.0),
                capacity=80.0,
                relative_loss_per_hour=0.003,
            ),
        ],
    }


def industry_park(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Industrial steam-and-power site with unit commitment and investment.

    Two 20 MW steam boilers with minimum load, minimum up/down times and
    startup costs cover a three-shift steam demand alongside a gas-engine CHP
    whose part-load efficiency follows a piecewise-linear curve. The optimizer
    may additionally invest in an electrode boiler (0-20 MW) and a steam
    accumulator (0-60 MWh), both carrying annualized capital cost and embodied
    CO2, and site emissions are capped at 80 kt CO2 per year.
    """
    n = timesteps
    price = _elec_price(n)
    steam_boilers = [
        Converter.boiler(
            f'steam_boiler_{i}',
            0.90,
            Flow(carrier='gas'),
            Flow(
                carrier='steam',
                size=20.0,
                relative_rate_min=0.35,
                status=Status(
                    uptime_min=4,
                    downtime_min=2,
                    effects_per_startup={'cost': 400.0},
                    effects_per_running_hour={'cost': 18.0},
                ),
            ),
        )
        for i in (1, 2)
    ]
    electrode_boiler = Converter.boiler(
        'electrode_boiler',
        0.99,
        Flow(carrier='elec'),
        Flow(
            carrier='steam',
            size=Sizing(size_min=0.0, size_max=20.0, effects_per_size={'cost': 9000.0, 'co2': 1800.0}),
        ),
    )
    site_chp = Converter(
        id='site_chp',
        inputs=[Flow(carrier='gas', size=30.0, ramp_up_per_hour=0.3, ramp_down_per_hour=0.3)],
        outputs=[Flow(carrier='elec'), Flow(carrier='steam')],
        conversion=PiecewiseConversion(
            points={
                'gas': [0.0, 12.0, 20.0, 30.0],
                'elec': [0.0, 3.6, 7.4, 12.0],
                'steam': [0.0, 6.0, 9.2, 12.6],
            }
        ),
    )
    return {
        'timesteps': _hourly_index(n),
        'carriers': [Carrier(id='gas'), Carrier(id='elec'), Carrier(id='steam')],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': CARBON_PRICE}),
            Effect(id='co2', unit='kg', total_max=8.0e7),
        ],
        'ports': [
            Port(
                id='gas_grid',
                imports=[Flow(carrier='gas', size=90.0, effects_per_flow_hour={'cost': GAS_PRICE, 'co2': GAS_CO2})],
            ),
            Port(
                id='power_grid',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=40.0,
                        effects_per_flow_hour={'cost': price.tolist(), 'co2': _grid_co2(n).tolist()},
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=15.0,
                        effects_per_flow_hour={'cost': (-0.9 * price).tolist()},
                    )
                ],
            ),
            Port(
                id='process_steam',
                exports=[Flow(carrier='steam', size=45.0, fixed_relative_profile=_steam_demand(n).tolist())],
            ),
            Port(
                id='machinery',
                exports=[Flow(carrier='elec', size=12.0, fixed_relative_profile=_city_elec_demand(n).tolist())],
            ),
        ],
        'converters': [*steam_boilers, electrode_boiler, site_chp],
        'storages': [
            Storage(
                id='steam_accumulator',
                charging=Flow(carrier='steam', size=15.0),
                discharging=Flow(carrier='steam', size=15.0),
                capacity=Sizing(size_min=0.0, size_max=60.0, effects_per_size={'cost': 1200.0, 'co2': 300.0}),
                relative_loss_per_hour=0.01,
            ),
        ],
    }


def green_city(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """Sector-coupled city energy system.

    A wind PPA, rooftop PV and a grid connection (hourly prices and CO2
    intensity) supply the city load, a battery sized by the optimizer, and two
    district-heating networks — each served by a heat pump with weather-driven
    COP, a gas peak boiler and a hot-water tank. Tracks cost, CO2 and primary
    energy; CO2 is priced into cost at 45 EUR/t.
    """
    n = timesteps
    price = _elec_price(n)
    cop = _heat_pump_cop(n).tolist()
    demand_north = _heat_demand(n)
    demand_south = np.roll(demand_north, 1)
    districts = [('north', demand_north, 25.0, 120.0), ('south', demand_south, 15.0, 60.0)]
    heat_ports = [
        Port(
            id=f'heat_network_{name}',
            exports=[Flow(carrier=f'heat_{name}', size=peak, fixed_relative_profile=demand.tolist())],
        )
        for name, demand, peak, _ in districts
    ]
    heat_plants = [
        converter
        for name, _, peak, _ in districts
        for converter in (
            Converter.heat_pump(
                f'heat_pump_{name}',
                cop,
                Flow(carrier='elec'),
                Flow(carrier='ambient', size=1e6),
                Flow(carrier=f'heat_{name}', size=0.6 * peak),
            ),
            Converter.boiler(
                f'peak_boiler_{name}', 0.93, Flow(carrier='gas'), Flow(carrier=f'heat_{name}', size=0.8 * peak)
            ),
        )
    ]
    tanks = [
        Storage(
            id=f'tank_{name}',
            charging=Flow(carrier=f'heat_{name}', size=0.5 * peak),
            discharging=Flow(carrier=f'heat_{name}', size=0.5 * peak),
            capacity=capacity,
            relative_loss_per_hour=0.003,
        )
        for name, _, peak, capacity in districts
    ]
    return {
        'timesteps': _hourly_index(n),
        'carriers': [
            Carrier(id='elec'),
            Carrier(id='gas'),
            Carrier(id='ambient'),
            Carrier(id='heat_north'),
            Carrier(id='heat_south'),
        ],
        'effects': [
            Effect(id='cost', unit='EUR', contribution_from={'co2': CARBON_PRICE}),
            Effect(id='co2', unit='kg'),
            Effect(id='primary_energy', unit='MWh'),
        ],
        'ports': [
            Port(
                id='wind_farm',
                imports=[
                    Flow(
                        carrier='elec',
                        size=60.0,
                        relative_rate_max=_wind(n).tolist(),
                        flow_hours_max=150_000.0,
                        effects_per_flow_hour={'cost': 58.0, 'primary_energy': 0.03},
                    )
                ],
            ),
            Port(
                id='rooftop_pv',
                imports=[
                    Flow(
                        carrier='elec',
                        size=35.0,
                        relative_rate_max=_solar(n).tolist(),
                        effects_per_flow_hour={'cost': 21.0, 'primary_energy': 0.03},
                    )
                ],
            ),
            Port(
                id='transmission_grid',
                imports=[
                    Flow(
                        carrier='elec',
                        short_id='buy',
                        size=80.0,
                        effects_per_flow_hour={
                            'cost': price.tolist(),
                            'co2': _grid_co2(n).tolist(),
                            'primary_energy': 1.9,
                        },
                    )
                ],
                exports=[
                    Flow(
                        carrier='elec',
                        short_id='sell',
                        size=40.0,
                        effects_per_flow_hour={'cost': (-0.9 * price).tolist()},
                    )
                ],
            ),
            Port(
                id='gas_grid',
                imports=[
                    Flow(
                        carrier='gas',
                        size=50.0,
                        effects_per_flow_hour={'cost': GAS_PRICE, 'co2': GAS_CO2, 'primary_energy': 1.1},
                    )
                ],
            ),
            Port(id='ambient_air', imports=[Flow(carrier='ambient', size=1e6)]),
            Port(
                id='city_load',
                exports=[Flow(carrier='elec', size=45.0, fixed_relative_profile=_city_elec_demand(n).tolist())],
            ),
            *heat_ports,
        ],
        'converters': heat_plants,
        'storages': [
            Storage(
                id='battery',
                charging=Flow(carrier='elec', size=25.0),
                discharging=Flow(carrier='elec', size=25.0),
                capacity=Sizing(size_min=0.0, size_max=200.0, effects_per_size={'cost': 14000.0, 'co2': 65000.0}),
                eta_charge=0.97,
                eta_discharge=0.97,
                relative_level_min=0.1,
            ),
            *tanks,
        ],
    }


def energy_transition(timesteps: int = HOURS_PER_YEAR) -> Elements:
    """The ``green_city`` system planned over eight five-year investment periods.

    Each period 2025-2060 is represented by a full hourly year (the ``period``
    dimension multiplies every temporal variable): electricity and heat demand
    grow with electrification, grid CO2 intensity falls as the surrounding
    power system decarbonizes, and the carbon price rises from 45 to 130 EUR/t.
    The battery becomes a proper multi-period ``Investment``: 15-year lifetime
    (three periods), overnight capex falling along a learning curve, fixed O&M
    recurring over each build's life. At the default horizon this is a
    ~2 million variable model.
    """
    n = timesteps
    periods = list(range(2025, 2065, 5))
    demand_growth = np.linspace(0.55, 1.0, len(periods))
    grid_decarbonization = np.linspace(1.0, 0.25, len(periods))
    elements = green_city(n)
    time_index = pd.DatetimeIndex(elements['timesteps'], name='time')
    period_index = pd.Index(periods, name='period')

    def by_period(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=['period'], coords={'period': periods})

    def spread(profile: Any, per_period: np.ndarray) -> pd.DataFrame:
        """Hourly profile times per-period factors → a (time, period) DataFrame."""
        return pd.DataFrame(np.outer(np.asarray(profile), per_period), index=time_index, columns=period_index)

    ports = {port.id: port for port in elements['ports']}
    for port_id in ('city_load', 'heat_network_north', 'heat_network_south'):
        flow = ports[port_id].exports[0]
        grown = flow.model_copy(update={'fixed_relative_profile': spread(flow.fixed_relative_profile, demand_growth)})
        ports[port_id] = ports[port_id].model_copy(update={'exports': [grown]})
    grid = ports['transmission_grid']
    buy = grid.imports[0]
    grid_effects = dict(buy.effects_per_flow_hour)
    grid_effects['co2'] = spread(grid_effects['co2'], grid_decarbonization)
    cleaner_buy = buy.model_copy(update={'effects_per_flow_hour': grid_effects})
    ports['transmission_grid'] = grid.model_copy(update={'imports': [cleaner_buy]})
    elements['ports'] = list(ports.values())
    rising_carbon_price = by_period(np.linspace(CARBON_PRICE, 0.13, len(periods)))
    elements['effects'] = [
        Effect(id='cost', unit='EUR', contribution_from={'co2': rising_carbon_price}),
        Effect(id='co2', unit='kg'),
        Effect(id='primary_energy', unit='MWh'),
    ]
    storages = {storage.id: storage for storage in elements['storages']}
    learning_curve = Investment(
        size_min=0.0,
        size_max=200.0,
        lifetime=3,
        effects_per_size_at_build={'cost': by_period(np.linspace(220_000.0, 80_000.0, len(periods))), 'co2': 65_000.0},
        effects_per_size_recurring={'cost': 3_000.0},
    )
    storages['battery'] = storages['battery'].model_copy(update={'capacity': learning_curve})
    elements['storages'] = list(storages.values())
    elements['periods'] = periods
    elements['period_weights'] = [5.0] * len(periods)
    return elements


SYSTEMS: dict[str, Callable[[int], Elements]] = {
    'district_heating': district_heating,
    'industry_park': industry_park,
    'green_city': green_city,
    'energy_transition': energy_transition,
}


def measure(model: str, timesteps: int = HOURS_PER_YEAR, solve: bool = False) -> dict[str, Any]:
    """Build one reference system and return stage timings, model size and peak memory."""
    builder = SYSTEMS[model]
    start = perf_counter()
    elements = builder(timesteps)
    elements_s = perf_counter() - start
    start = perf_counter()
    data = ModelData.build(**elements)
    data_s = perf_counter() - start
    start = perf_counter()
    fsm = FlowSystemModel(data, objective='cost')
    fsm.build()
    build_s = perf_counter() - start
    row: dict[str, Any] = {
        'model': model,
        'timesteps': timesteps,
        'variables': fsm.m.nvars,
        'constraints': fsm.m.ncons,
        'elements_s': elements_s,
        'data_s': data_s,
        'build_s': build_s,
    }
    if solve:
        start = perf_counter()
        fsm.solve(solver_name='highs', output_flag=False)
        row['solve_s'] = perf_counter() - start
    row['peak_mib'] = _peak_rss_mib()
    return row


def _peak_rss_mib() -> float | None:
    """Peak resident memory of this process in MiB (None where unsupported, e.g. Windows).

    Whole-process, OS-level high-water: catches every allocation (numpy, solver
    C libraries, ...) but includes the interpreter + import footprint and
    allocator slack — the number that has to fit in RAM, not the build's own
    appetite.
    """
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == 'darwin' else 1024
    return peak * scale / 2**20


def _measure_in_subprocess(model: str, timesteps: int, solve: bool) -> dict[str, Any]:
    """Measure one system in a fresh interpreter so peak memory is attributed per model."""
    cmd = [sys.executable, '-m', 'fluxopt.benchmark', '--worker', model, '--timesteps', str(timesteps)]
    if solve:
        cmd.append('--solve')
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f'exit code {proc.returncode}'
        return {'model': model, 'error': detail}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f'{n / 1e6:.2f}M'
    if n >= 10_000:
        return f'{n / 1e3:.0f}k'
    return str(n)


def _fmt_seconds(s: float) -> str:
    return f'{s * 1000:.0f} ms' if s < 1.0 else f'{s:.1f} s'


def _fmt_mem(mib: float | None) -> str:
    if mib is None:
        return 'n/a'
    return f'{mib / 1024:.1f} GiB' if mib >= 1024 else f'{mib:.0f} MiB'


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Plain-text table; first column left-aligned, the rest right-aligned."""
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = []
    for cells in [headers, *rows]:
        first = cells[0].ljust(widths[0])
        rest = (cell.rjust(width) for cell, width in zip(cells[1:], widths[1:], strict=True))
        lines.append('  '.join([first, *rest]).rstrip())
    lines.insert(1, '-' * len(lines[0]))
    return '\n'.join(lines)


def _print_report(rows: list[dict[str, Any]], timesteps: int, solve: bool) -> None:
    print(f'fluxopt {version("fluxopt")} — build-pipeline benchmark')
    print(f'Python {platform.python_version()} · {platform.system()} {platform.machine()} · {os.cpu_count()} CPUs')
    print(f'{timesteps} hourly timesteps ({timesteps / HOURS_PER_YEAR:.1f} years)')
    print()
    headers = [
        'model',
        'variables',
        'constraints',
        'elements',
        'data',
        'build',
        *(['solve'] if solve else []),
        'peak rss',
    ]
    table_rows = [
        [
            row['model'],
            _fmt_count(row['variables']),
            _fmt_count(row['constraints']),
            _fmt_seconds(row['elements_s']),
            _fmt_seconds(row['data_s']),
            _fmt_seconds(row['build_s']),
            *([_fmt_seconds(row['solve_s'])] if solve else []),
            _fmt_mem(row['peak_mib']),
        ]
        for row in rows
        if 'error' not in row
    ]
    if table_rows:
        print(_render_table(headers, table_rows))
    for row in (r for r in rows if 'error' in r):
        print(f'{row["model"]}: FAILED — {row["error"]}')


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m fluxopt.benchmark',
        description='Build a few realistic reference energy systems and report speed and memory.',
    )
    parser.add_argument(
        'models',
        nargs='*',
        choices=sorted(SYSTEMS),
        metavar='model',
        help=f'reference systems to run (default: all — {", ".join(SYSTEMS)})',
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=HOURS_PER_YEAR,
        help='number of hourly timesteps (default: 8760, one year)',
    )
    parser.add_argument('--solve', action='store_true', help='also solve each model with HiGHS and time it')
    parser.add_argument('--json', action='store_true', help='print JSON instead of the table')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    if args.worker:
        json.dump(measure(args.worker, args.timesteps, args.solve), sys.stdout)
        return 0
    models = args.models or list(SYSTEMS)
    rows = []
    for name in models:
        print(f'building {name} ({args.timesteps} timesteps) ...', file=sys.stderr, flush=True)
        rows.append(_measure_in_subprocess(name, args.timesteps, args.solve))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        _print_report(rows, args.timesteps, args.solve)
    return 1 if any('error' in row for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
