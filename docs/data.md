# Data in and out

fluxopt reads numbers as scalars, arrays or polars tables, and a solve
returns polars tables. It does not read pandas or xarray objects.

## The time axis

| Argument         | Takes                                                                        |
| ---------------- | ---------------------------------------------------------------------------- |
| `timesteps`      | A list of `datetime`, or a `pl.Series` of datetimes. Strictly increasing.     |
| `dt`             | Hours per step: one number, or one per step. Derived from the steps if left out. |
| `periods`        | Integer labels, such as years, for a multi-period system.                     |
| `period_weights` | One weight per period. Derived from the gaps between periods if left out.     |

Numbered steps are refused, because a number is not a time. Write hourly
steps as `pl.datetime_range(datetime(2024, 1, 1), datetime(2024, 1, 1, 23), '1h', eager=True)`.

## A value that varies

A field such as `Flow.fixed_relative_profile` or `effects_per_flow_hour`
takes a `Variate`, in one of these forms:

| Form                                    | Read as                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| A number                                | The same value everywhere.                                                     |
| A list, a 1-D `np.ndarray`, a `pl.Series` | One value per label of the dimension its length matches.                      |
| A tidy `pl.DataFrame`                   | One column per dimension it varies over, named after it, and a `value` column. |
| A `ProfileRef`                          | A column of a table supplied at solve time, see [below](#profiles).            |

A 1-D value is matched to a dimension by its length. When `time` and
another dimension have the same length, it is read as time. Give a
per-period value as a table with a `period` column to remove the doubt:

```python
import polars as pl

from fluxopt import Effect

carbon_price = pl.DataFrame({'period': [2030, 2040], 'value': [80.0, 120.0]})
cost = Effect(id='cost', contribution_from={'co2': carbon_price})
```

A tidy table must fill the dimensions it names. Each of these is refused,
with a message that names it:

- a column that is not a dimension of the field,
- a label the system does not have,
- a combination that appears twice,
- a combination that is missing.

A value that varies over time and period is a table with both columns:

```python
from datetime import datetime

import polars as pl

from fluxopt import Flow

times = [datetime(2024, 1, 1, h) for h in range(2)]
demand = pl.DataFrame(
    {
        'time': [t for t in times for _ in range(2)],
        'period': [2030, 2040] * 2,
        'value': [0.4, 0.5, 0.7, 0.8],
    }
)
load = Flow(carrier='heat', size=100, fixed_relative_profile=demand)
```

## Profiles

A system you save to YAML holds no long series inline. Name the series with a
`ProfileRef` instead, and supply the table when you solve:

```python
from datetime import datetime

import polars as pl

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port, ProfileRef

times = [datetime(2024, 1, 1, h) for h in range(3)]
system = FlowSystem(
    timesteps=times,
    carriers=[Carrier(id='elec')],
    effects=[Effect(id='cost')],
    objective='cost',
    ports=[
        Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
        Port(
            id='demand',
            exports=[Flow(carrier='elec', size=100, fixed_relative_profile=ProfileRef(table='load', column='demand'))],
        ),
    ],
)

assert system.required_profiles() == {'load': {'demand'}}
load = pl.DataFrame({'time': times, 'demand': [0.4, 0.7, 0.5]})
result = system.optimize(profiles={'load': load})
```

A profile table is keyed by its `time` column, and a `period` column where it
varies by period. Every other column is a profile. A table with no key
columns is read by length, like a list.

## Reading a result

`optimize` returns specsolve's `Result`. Every reader returns a tidy polars
table, with one column per dimension and a `value` column:

| Reader                    | Returns                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `result.objective`        | The optimal objective, a number.                                              |
| `result.primal(name)`     | A variable, such as `rate`, `level`, `chosen_size` or `effect_total`.          |
| `result.evaluate(name)`   | A named expression, such as `flow_hours`, `carrier_balance`, `capacity_factor` or `priced_flow_hour`. |
| `result.dual(name)`       | The shadow prices of a constraint, such as `carrier_balance`.                  |

```python
rates = result.primal('rate')  # columns: flow, time, period, value
grid = rates.filter(pl.col('flow') == 'grid(elec)')
```

A system that declares no periods is solved on one period labelled `0`; drop
that column with `.drop('period')`. The names a result holds are the names in
the [program pages](math/program/flows.md).
