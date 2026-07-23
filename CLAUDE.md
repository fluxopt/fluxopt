# CLAUDE.md — Project Guide for fluxopt

Contribution rules, development workflow and project conventions live in
[`AGENTS.md`](AGENTS.md), imported below so it loads automatically. Read it
before making changes, opening pull requests or issues, or writing comments.

@AGENTS.md

## What is fluxopt?

Energy system optimization library. Models energy flows between components
(converters, storages, ports) on buses, then solves with linopy/HiGHS.

## Architecture

Three-layer design: **Elements** (dataclasses) → **ModelData** (xr.Datasets) → **Model** (linopy)

```
src/fluxopt/
├── elements.py        # User-facing dataclasses (Carrier, Flow, Effect, Storage, Status, Sizing, Investment, PiecewiseConversion)
├── components.py      # Port, Converter — group flows into components
├── types.py           # IdList[T], Variate, Identified protocol
├── model_data.py      # ModelData + builder functions → 5 xr.Datasets (flows, buses, converters, effects, storages); netcdf IO
├── model.py           # Builds linopy Model from ModelData (variables, constraints, objective)
├── constraints/       # Modular constraint builders (status, storage, sparse)
├── results.py         # Extract results from solved model; netcdf IO
├── stats.py           # StatsAccessor — derived KPIs on Result
└── contributions.py   # Effect contribution bookkeeping
```

Field naming follows a fixed grammar (`<quantity>_min/max` suffix style,
rate/periodic/total scopes) — see `docs/design/naming-grammar.md`.

Key runtime deps: xarray, linopy, numpy, pandas.

## Philosophy

- **uv is the single entry point** — no pip, no setuptools CLI, no tox
- **pyproject.toml is the single source of truth** — no setup.py/cfg, no tox.ini
- **src layout** — `src/fluxopt/`, enforcing proper installation
- **hatchling + hatch-vcs** — version from git tags
- **ruff** replaces flake8, isort, pyupgrade, black
- **pyrefly** — Meta's type checker, enforced from day one
- **No lock file** — `uv.lock` is gitignored

## Common Commands

```bash
uv sync --group dev      # Install runtime + dev deps
uv run pytest -v         # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run pyrefly check src/ # Type check
```

## Code Style

- **Docstrings**: Google style, brief, on public functions
  - No types in docstrings (types live in signatures only)
  - Always include `Args` section when there are parameters
  - `Returns` / `Raises` only when non-obvious
  - **Exception — pydantic model fields**: document each field with an inline
    attribute docstring (a `"""..."""` under the field), not an `Args:`/`Attributes:`
    section. This is what makes griffe-pydantic emit per-field anchors that
    `docs/math/*.md` cross-links to (`Class.field`). `Args:` still applies to
    functions, methods, and non-model classes.
- Python >= 3.12 — use modern syntax (PEP 604 unions `X | Y`, etc.)
- **linopy**: use concise, vectorized syntax — no loops over coordinates
- **xr.DataArray** is the primary data container; prefer broadcasting over iteration

## Commit & PR Conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) for **all** commit messages and PR titles:

```
<type>: <short summary>
```

Types: `feat`, `fix`, `refac`, `test`, `docs`, `chore`, `ci`, `perf`.
Optional scope: `feat(storage): add cyclic constraint`.

- **Commit messages**: `feat: add TimeIndex alias` (imperative, lowercase after colon)
- **PR titles**: same format — `refac: clarify timestep input vs internal types`
- **Issue titles**: same format — `fix: wrong cost on cyclic storage`
- No period at end, max ~70 chars

## Labels

- **`type:*`** — Conventional Commits type. Auto-applied from issue/PR title by
  `.github/workflows/label-from-title.yaml`. Only `feat`, `fix`, `docs`, `perf`
  get a label; other prefixes (`refac`, `test`, `ci`, `build`, `chore`) no-op.
- **`area:*`** — subsystem. Applied manually during triage.
  - `area:api` — cross-cutting naming, user-facing types, ergonomics
  - `area:flow` — Flow, Port
  - `area:converter` — Converter, part-load behavior
  - `area:storage` — Storage elements
  - `area:status` — on/off behavior, startup, min run time
  - `area:sizing` — Sizing / Investment, capacity optimization
  - `area:effects` — Effect system, objectives, contributions
  - `area:multi-period` — Periods, rolling horizon, scenarios, TSA
  - `area:io` — serialization, loading, saving
  - `area:new` — novel concept not yet fitting an existing area (temporary
    flag; recategorize when a new area emerges)
- **Meta**: `good first issue`, `help wanted` (unprefixed — GitHub's
  contributors page recognizes these exact strings).

## Math Documentation

Hybrid approach — plain-text formulas in code, full LaTeX in docs.

- **Docstrings**: one-line formulas in Unicode notation (P⁺, P⁻, η, δ), plus `See: docs/math/...` link
- **`docs/math/`**: full LaTeX derivations, variable tables, explanations (rendered by mkdocs-material)
- **Notation**: uppercase Latin for variables (P, E, S), Greek for properties (η, δ),
  superscript +/− for bounds (P⁺ upper, P⁻ lower), subscripts for indexing (f, t, s, b, k),
  superscripts for qualification (η^c, η^d)
