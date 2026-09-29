# fluxopt

Energy system optimization with [specsolve](https://github.com/fluxopt/lpspec) — detailed dispatch, scaled to multi period planning.

[![PyPI](https://img.shields.io/pypi/v/fluxopt)](https://pypi.org/project/fluxopt/)
[![Downloads](https://img.shields.io/pypi/dm/fluxopt)](https://pypi.org/project/fluxopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Early development** — the API may change between releases.
> Planned features and progress are tracked in [Issues](https://github.com/FBumann/fluxopt/issues).

## Installation

```bash
pip install fluxopt
```

Includes the [HiGHS](https://highs.dev/) solver out of the box.

## Quick Start

<!-- --8<-- [start:quickstart] -->

```python
# A gas boiler covers a heat demand, minimizing fuel cost
from datetime import datetime
from fluxopt import Carrier, Converter, Effect, Flow, Port, optimize

result = optimize(
    timesteps=[datetime(2024, 1, 1, h) for h in range(4)],
    carriers=[Carrier(id='gas'), Carrier(id='heat')],
    effects=[Effect(id='cost')],
    ports=[
        Port(id='grid', imports=[Flow(carrier='gas', size=500, effects_per_flow_hour={'cost': 0.04})]),
        Port(id='demand', exports=[Flow(carrier='heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])]),
    ],
    converters=[
        Converter.boiler(
            'boiler',
            thermal_efficiency=0.9,
            fuel_flow=Flow(carrier='gas', size=300),
            thermal_flow=Flow(carrier='heat', size=200),
        )
    ],
    objective='cost',
)

print(f'Total cost: {result.objective:.2f}')
print(result.to_dataarray('rate').squeeze('period', drop=True))
```

<!-- --8<-- [end:quickstart] -->

## One API, three levels of control

Every level answers with specsolve's `Result`; each one only adds control —
pick the lowest rung that does the job.

**1. One-shot** — `optimize(...)` as above. Elements in, a solved answer out,
with fail-fast validation of ids and references.

**2. Declarative** — gather the same arguments into a reusable, serializable
system. Time series can stay out of the structure as `ProfileRef`s and be
supplied at solve time via `profiles`:

```python
system = fx.FlowSystem.from_yaml('system.yaml')  # or FlowSystem(...) in Python
result = system.optimize(profiles={'load': demand_ds}, archive='run.zip')
system.to_yaml('system.yaml')  # round-trips
```

**3. Spec and sources** — the system is a mathspec spec and the tables bound to
it. Read, typeset or extend the spec, edit any table, and solve with specsolve:

```python
import mathspec, specsolve

spec = mathspec.override(system.spec(), {'my cap': 'my_cap.yaml'})
sources = system.sources(profiles={'load': demand_ds}) | {'grid_cap': caps}
result = specsolve.solve(spec, sources)
```

The spec's `assumptions:` check whatever tables arrive.

Read an answer with `result.to_dataarray("rate")` for a variable, or
`result.to_dataarray("flow_hours", kind="expression")` for a reported quantity:
flow hours, carrier balance, capacity factor, storage mean level, and each
contribution with its cross-effects charged (`priced_*`). `archive=` writes
the spec, its sources and the answer; `specsolve.load_archive` reads them back.

## Roadmap

fluxopt is evolving into a family of packages with a lean core and optional companions:

```
                          ┌──────────────┐
                          │   fluxopt    │  core: model building, solving, results, IO
                          └──────┬───────┘
        ┌──────────────┬─────────┼──────────────┬──────────────┐
        │              │         │              │              │
 fluxopt-plot   fluxopt-yaml  fluxopt-tsam  fluxopt-marimo  (examples)
   plotting      YAML+CSV    time series    interactive     cross-package
   (plotly)       loader     aggregation       apps          notebooks
```

Companion packages depend on core — core has no knowledge of companions.

### Companion packages

| Package          | Role                                                                           | Versioning · Tier                                            | `fluxopt` pin                                                                                                           | Status                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `fluxopt-plot`   | Result visualization (Plotly)                                                  | Semver · Experimental — method signatures may change         | Tight (`>=A.B,<A.C`), validated per release                                                                             | Scaffolded — [docs](https://fbumann.github.io/fluxopt-plot/latest/) · [#51](https://github.com/FBumann/fluxopt/issues/51) |
| `fluxopt-yaml`   | Declarative model loader (YAML + CSV → `Element`s)                             | Semver · Experimental — YAML schema may change               | Tight (`>=A.B,<A.C`), validated per release                                                                             | Scaffolded — [docs](https://fbumann.github.io/fluxopt-yaml/latest/) · [#52](https://github.com/FBumann/fluxopt/issues/52) |
| `fluxopt-tsam`   | Time series aggregation — input pre-processing, possibly result disaggregation | Semver · Experimental — round-trip schema may evolve         | **Undecided** — depends on whether representative-period primitives live in core (→ loose) or in this package (→ tight) | Planned                                                                                                                   |
| `fluxopt-marimo` | Interactive exploration & dashboards (marimo apps)                             | CalVer (`YYYY.MM.PATCH`) · Experimental — apps are templates | Tight (`>=A.B,<A.C`), validated per release                                                                             | Planned                                                                                                                   |

Tight-pinned companions release on every `fluxopt` minor; validation is
automated via scheduled CI. `fluxopt-tsam`'s pin policy is blocked on an
architectural decision — if representative-period primitives live in core, tsam stays
a thin adapter (loose pin); if they live in tsam, the package owns deep
round-trip behavior (tight pin).

### Milestones

Cross-cutting work not tied to a single companion package:

| Milestone               | Description                                           | Status  | Issue                                               |
| ----------------------- | ----------------------------------------------------- | ------- | --------------------------------------------------- |
| ReadTheDocs migration   | Automatic versioned docs from git tags                | Planned | [#53](https://github.com/FBumann/fluxopt/issues/53) |
| Remove plotly from core | Keep core lean — plotting deps in `fluxopt-plot` only | Planned | [#54](https://github.com/FBumann/fluxopt/issues/54) |

### Stability Tiers

| Component         | Tier            | Policy                                                                |
| ----------------- | --------------- | --------------------------------------------------------------------- |
| Core modeling API | **Stable**      | Semver. Deprecation warnings before removal.                          |
| Stats accessor    | **Semi-stable** | Breaking changes allowed between minor versions with changelog entry. |

Companion packages have their own stability policies — see the table above.

See [#47](https://github.com/FBumann/fluxopt/issues/47) for the full architecture discussion.

## Development

Requires [pixi](https://pixi.sh). It installs Python and every tool from
`pixi.lock`.

```bash
pixi run pre-commit-install  # Lint every commit
pixi run test                # Run tests
pixi run lint                # Lint, format and type-check
pixi run ci                  # Everything CI checks
```

See [CONTRIBUTING](.github/CONTRIBUTING.md) and [RELEASING](RELEASING.md).

## License

MIT
