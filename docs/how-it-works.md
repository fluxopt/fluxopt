# How fluxopt works

fluxopt turns an energy system you describe with components into the tables
of a math program, and hands both to a solver. Three packages share the work:

| Package                                                 | Owns                                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **fluxopt**                                             | The components (`Carrier`, `Flow`, `Port`, `Converter`, `Storage`, `Effect`), their checks, and the tables built from them. It also owns the math, as YAML files. |
| [mathspec](https://github.com/energy-models/mathspec)   | The language the math is written in. It composes fluxopt's YAML files into one spec, checks it with no data, and prints it as equations. |
| [specsolve](https://github.com/fluxopt/specsolve)          | Binding the tables to the spec, building the model, solving it, and returning the answer as tables.    |

## The path of one solve

```text
components ──► FlowSystem ─┬─► spec()     a mathspec Spec: the math, no data
                           └─► sources()  a dict of polars tables: the data
                                    │
                                    ▼
                     specsolve.solve(spec, sources) ──► Result: polars tables
```

`optimize(...)` and `FlowSystem.optimize()` run this path in one call. Each
step is also available on its own, and the result is the same:

```python
from datetime import datetime

import specsolve

from fluxopt import Carrier, Effect, Flow, FlowSystem, Port

system = FlowSystem(
    timesteps=[datetime(2024, 1, 1, h) for h in range(3)],
    carriers=[Carrier(id='elec')],
    effects=[Effect(id='cost')],
    objective='cost',
    ports=[
        Port(id='grid', imports=[Flow(carrier='elec', size=100, effects_per_flow_hour={'cost': 1.0})]),
        Port(id='demand', exports=[Flow(carrier='elec', size=100, fixed_relative_profile=[0.5, 0.5, 0.5])]),
    ],
)

spec = system.spec()  # the math: read, typeset or extend it
sources = system.sources()  # the data: one polars table per name the spec declares
result = specsolve.solve(spec, sources)

assert result.objective == system.optimize().objective
```

## The math is a file

fluxopt's math is the YAML files under
[`src/fluxopt/math/program/`](https://github.com/fluxopt/fluxopt/tree/main/src/fluxopt/math/program),
one per feature: flows, converters, storage, status, sizing and so on. There is
no second implementation in Python. The same files are:

- **checked** by mathspec before any data is bound, so a wrong name or
  dimension fails when the spec loads.
- **printed** as the [program pages](math/program/flows.md) of this site, in
  the notation of the [Math](math/notation.md) pages.
- **solved** by specsolve, from the same spec that is printed.

The Math pages explain each feature. The program pages show exactly what the
solver reads.

## The data is tables

`system.sources()` returns every table the spec declares, keyed by the labels
you wrote: the timestamps, the period years, and qualified ids such as
`grid(elec)`. A table holds only the rows that exist. For example, a flow with
no cost has no row in `effects_per_flow_hour`.

You can read any table, edit it, or add a table for math of your own, and
pass the dict to `specsolve.solve`. The spec's `assumptions:` check whatever
arrives. [Extending the math](extending.md) shows how.

## What fluxopt does not do

fluxopt holds no model object and no result type of its own. The model lives
inside specsolve for the length of a solve. The answer is specsolve's
`Result`, read as polars tables; [Data in and out](data.md) lists the
readers.
