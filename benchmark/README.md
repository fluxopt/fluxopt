# Benchmarks

Time + memory benchmarks for the build pipeline —
`Elements → ModelData (xarray) → FlowSystemModel (linopy)`. The HiGHS solve is
excluded (non-deterministic, not ours to profile).

One `benchmark()` suite, served by [CodSpeed] in CI and [pytest-benchmem] locally.

- `systems.py` — feature archetypes (`multi_node`, `status`, `piecewise`,
  `effects`, `sizing`) + `(n, timesteps)` scale tiers. Deterministic.
- `test_build.py` — the feature matrix at one scale, and a multi_node scaling curve.

## Pinned, standalone env

This directory is its own small uv project (`pyproject.toml` + committed
`uv.lock`) — the **one** pinned environment in the repo; the root keeps regular,
unpinned resolution. A pinned bench env keeps the CodSpeed dashboard tracking
fluxopt's code, not upstream dependency releases. Dependabot bumps `uv.lock`
monthly (`.github/dependabot.yml`).

```bash
cd benchmark
uv sync                          # or --frozen to enforce the lock exactly
uv run pytest . --codspeed                        # CodSpeed (walltime)
uv run pytest . --codspeed --codspeed-mode memory # CodSpeed (memory / heap)
```

## CI — CodSpeed

`.github/workflows/benchmarks.yaml` runs the suite under CodSpeed (OIDC, no
token), tracking history and annotating PRs on the dashboard:

- **Memory** (`mode: memory`) — heap tracking on a free GitHub runner, every PR + main.
- **Walltime** (`mode: walltime`) — bare-metal macro runner, main + PRs labelled
  `trigger:benchmark` only.

Both are `continue-on-error` (informational, never block a merge). The walltime
job needs a CodSpeed `codspeed-macro` runner provisioned for the org.

## Local memory profiling — pytest-benchmem

Peak-memory number next to the timings, plus a flamegraph of where it goes:

```bash
uv run pytest . --benchmark-only --benchmark-memory
uv run pytest . --benchmark-only --benchmark-memory --benchmark-memory-profile profiles/
uv run benchmem flamegraph profiles/ --worst peak --open
```

[CodSpeed]: https://codspeed.io
[pytest-benchmem]: https://github.com/fluxopt/pytest-benchmem
