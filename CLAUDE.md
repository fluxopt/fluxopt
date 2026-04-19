# CLAUDE.md — Project Guide for fluxopt

## What is fluxopt?

Energy system optimization library. Models energy flows between components
(converters, storages, ports) on buses, then solves with linopy/HiGHS.

## Architecture

Three-layer design: **Elements** (dataclasses) → **ModelData** (xr.Datasets) → **Model** (linopy)

```
src/fluxopt/
├── elements.py        # User-facing dataclasses (Flow, Bus, Effect, Storage, Status, Sizing, Bounds)
├── components.py      # Port, Converter — group flows into components
├── types.py           # IdList[T], TimeSeries, Identified protocol
├── model_data.py      # ModelData + builder functions → 5 xr.Datasets (flows, buses, converters, effects, storages)
├── model.py           # Builds linopy Model from ModelData (variables, constraints, objective)
├── constraints/       # Modular constraint builders (status, accumulation, etc.)
├── results.py         # Extract results from solved model
└── io.py              # Serialization
```

Key runtime deps: xarray, linopy, numpy, pandas.

## Philosophy

- **uv is the single entry point** — no pip, no setuptools CLI, no tox
- **pyproject.toml is the single source of truth** — no setup.py/cfg, no tox.ini
- **src layout** — `src/fluxopt/`, enforcing proper installation
- **hatchling + hatch-vcs** — version from git tags
- **ruff** replaces flake8, isort, pyupgrade, black
- **mypy strict** — enforced from day one
- **No lock file** — `uv.lock` is gitignored

## Common Commands

```bash
uv sync --group dev      # Install runtime + dev deps
uv run pytest -v         # Run tests
uv run ruff check .      # Lint
uv run ruff format .     # Format
uv run mypy src/         # Type check
```

## Code Style

- **Docstrings**: Google style, brief, on public functions
  - No types in docstrings (types live in signatures only)
  - Always include `Args` section when there are parameters
  - `Returns` / `Raises` only when non-obvious
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
  `area:api` (user-facing types, ergonomics), `area:model` (optimization math
  and data model), `area:io` (serialization, loading, saving),
  `area:viz` (plots, stats).
- **Meta**: `good first issue`, `help wanted` (unprefixed — GitHub's
  contributors page recognizes these exact strings).

## Math Documentation

Hybrid approach — plain-text formulas in code, full LaTeX in docs.

- **Docstrings**: one-line formulas in Unicode notation (P⁺, P⁻, η, δ), plus `See: docs/math/...` link
- **`docs/math/`**: full LaTeX derivations, variable tables, explanations (rendered by mkdocs-material)
- **Notation**: uppercase Latin for variables (P, E, S), Greek for properties (η, δ),
  superscript +/− for bounds (P⁺ upper, P⁻ lower), subscripts for indexing (f, t, s, b, k),
  superscripts for qualification (η^c, η^d)
