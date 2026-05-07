# fluxopt

Energy system optimization with [linopy](https://github.com/PyPSA/linopy) — detailed dispatch, scaled to multi period planning.

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

<!--quickstart-start-->
```python
# A gas boiler covers a heat demand, minimizing fuel cost
from datetime import datetime
from fluxopt import Carrier, Converter, Effect, Flow, Port, optimize

result = optimize(
    timesteps=[datetime(2024, 1, 1, h) for h in range(4)],
    carriers=[Carrier('gas'), Carrier('heat')],
    effects=[Effect('cost')],
    ports=[
        Port('grid', imports=[
            Flow('gas', size=500, effects_per_flow_hour={'cost': 0.04})
        ]),
        Port('demand', exports=[
            Flow('heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
        ])
    ],
    converters=[
        Converter.boiler(
            'boiler',
            thermal_efficiency=0.9,
            fuel_flow=Flow('gas', size=300),
            thermal_flow=Flow('heat', size=200)
        )
    ],
    objective_effects='cost',
)

print(f"Total cost: {result.objective:.2f}")
print(result.flow_rates)
```
<!--quickstart-end-->

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

### Milestones

| Milestone | Description | Status | Issue |
|-----------|-------------|--------|-------|
| `Result.stats` accessor | Cached xarray properties for post-processing | Planned | [#49](https://github.com/FBumann/fluxopt/issues/49) |
| `.plot` stub on `Result` | Discoverable property, helpful error if plot package absent | Planned | [#50](https://github.com/FBumann/fluxopt/issues/50) |
| `fluxopt-plot` package | Interactive plotly visualization as companion package | [Scaffolded](https://fbumann.github.io/fluxopt-plot/latest/) | [#51](https://github.com/FBumann/fluxopt/issues/51) |
| `fluxopt-yaml` package | Declarative model definition via YAML + CSV | [Scaffolded](https://fbumann.github.io/fluxopt-yaml/latest/) | [#52](https://github.com/FBumann/fluxopt/issues/52) |
| `fluxopt-tsam` package | Time series aggregation preprocessing | Planned | — |
| `fluxopt-marimo` package | Interactive marimo apps for result exploration & dashboards | Planned | — |
| ReadTheDocs migration | Automatic versioned docs from git tags | Planned | [#53](https://github.com/FBumann/fluxopt/issues/53) |
| Remove plotly from core | Keep core lean — plotting deps in `fluxopt-plot` only | Planned | [#54](https://github.com/FBumann/fluxopt/issues/54) |

### Stability Tiers

| Component | Tier | Policy |
|-----------|------|--------|
| Core modeling API | **Stable** | Semver. Deprecation warnings before removal. |
| Stats accessor | **Semi-stable** | Breaking changes allowed between minor versions with changelog entry. |
| `fluxopt-yaml` | **Experimental** | Own versioning. YAML schema may change. |
| `fluxopt-plot` | **Experimental** | Own versioning. Method signatures may change. |
| `fluxopt-marimo` | **Experimental** | Own versioning. App templates, not a stable API. |
| `fluxopt-tsam` | **Independent** | Fully independent semver. |

See [#47](https://github.com/FBumann/fluxopt/issues/47) for the full architecture discussion.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python >= 3.12.

```bash
uv sync --group dev      # Install deps
uv run pytest -v         # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
```

## License

MIT
