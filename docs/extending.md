# Extending the math

This guide adds a constraint of your own to a fluxopt system, edits one of
the tables fluxopt builds, charges a cost of your own, and asks the result for
a quantity of your own. It uses
[mathspec](https://github.com/energy-models/mathspec) to change the spec and
[specsolve](https://github.com/fluxopt/specsolve) to solve it. Read
[How fluxopt works](how-it-works.md) first if the spec and the tables are new
to you.

## The system

A grid and a dearer backup source supply a fixed demand. Unconstrained, the
cheap grid carries all of it: 50 in each hour.

```python
from datetime import datetime

import mathspec
import polars as pl
import specsolve

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port

times = [datetime(2024, 1, 1, h) for h in range(3)]
system = FlowSystem(
    timesteps=times,
    carriers=[Carrier(id='elec')],
    effects=[Effect(id='cost')],
    objective='cost',
    ports=[
        Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
        Port(id='backup', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 5.0})]),
        Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
    ],
)
spec = system.spec()
sources = system.sources()
```

## Add a constraint

Write the constraint in mathspec's language. It declares the data it needs,
`grid_cap` and `is_grid`, and reads `rate` from fluxopt's math by name:

```python
grid_cap = """
parameters:
  grid_cap: {dims: [time]}
  is_grid: {dims: [flow], dtype: bool}
constraints:
  grid_cap_row:
    dims: [flow, time, period]
    where: is_grid
    expression: rate <= grid_cap
"""
capped = mathspec.override(spec, [grid_cap])
```

Give each new parameter a table, keyed by the labels fluxopt uses, and solve
the pair:

```python
capped_sources = sources | {
    'grid_cap': pl.DataFrame({'time': times, 'value': [30.0, 30.0, 30.0]}),
    'is_grid': pl.DataFrame({'flow': ['grid(elec)'], 'value': [True]}),
}
result = specsolve.solve(capped, capped_sources)

grid = result.primal('rate').filter(pl.col('flow') == 'grid(elec)')
assert grid.get_column('value').to_list() == [30.0, 30.0, 30.0]
```

The cap binds: the grid carries 30 in each hour, and the backup carries the
other 20. `mathspec.to_markdown(capped)` prints the new constraint beside
fluxopt's own.

## Edit a table

The tables are yours to change before the solve. A grid three times as
expensive costs three times as much:

```python
rates = sources['effects_per_flow_hour']
dearer = sources | {'effects_per_flow_hour': rates.with_columns(pl.col('value') * 3)}

assert specsolve.solve(spec, dearer).objective == 3 * system.optimize().objective
```

## Charge a new cost

What each effect is charged in a step is the sum `direct_step`, and each
feature of fluxopt adds its own term to it. A file of your own adds a term the
same way: it reads the sum under `given:` and names it with `adds_to:`. Here a
fee is charged on every unit a flow carries:

```python
grid_fee = """
dimensions:
  time: {dtype: datetime}
  period: {dtype: int}
  effect: {dtype: str}
  flow: {dtype: str}
parameters:
  fee: {dims: [flow, effect]}
given:
  parameters:
    dt: {dims: [time]}
  variables:
    rate: {dims: [flow, time, period]}
  expressions:
    direct_step: {dims: [effect, time, period]}
expressions:
  grid_fees:
    expression: sum(rate * dt * fee, over=flow)
    adds_to: direct_step
"""
charged = mathspec.merge([spec, grid_fee])
fee = pl.DataFrame({'flow': ['grid(elec)'], 'effect': ['cost'], 'value': [2.0]})

assert specsolve.solve(charged, sources | {'fee': fee}).objective == 3 * system.optimize().objective
```

`mathspec.merge` appends `grid_fees` to the sum, and no file of fluxopt
changes. The fee file loads and prints on its own, so `mathspec.to_markdown`
shows the new cost before any merge.

## Name a quantity to read it

A result holds the names the spec declares. Declare an expression to get a
quantity fluxopt does not report:

```python
grid_energy = """
expressions:
  grid_energy:
    expression: sum(rate * dt, over=flow)
    description: every flow the system moved, per step
"""
result = specsolve.solve(mathspec.override(spec, [grid_energy]), sources)
print(result.evaluate('grid_energy'))
```

Pass `archive='run.zip'` to `specsolve.solve` to keep the spec, the tables and
the answer together. `specsolve.load_archive('run.zip')` reads them back.
