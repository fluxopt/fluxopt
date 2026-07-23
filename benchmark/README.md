# Benchmarks

Time + memory benchmarks for the build pipeline —
`Elements → ModelData (xarray) → FlowSystemModel (linopy)` — and for NetCDF IO
(`ModelData ⇄ .nc`). The HiGHS solve is excluded (non-deterministic, not ours to
profile); the solved-`Result` round-trip is excluded for the same reason (it
needs a solve to produce).

One `benchmark()` suite, served by [CodSpeed] in CI and [pytest-benchmem] locally.

- `systems.py` — feature archetypes (`multi_node`, `status`, `piecewise`,
  `effects`, `sizing`) + `(n, timesteps)` scale tiers. Deterministic.
- `test_build.py` — the feature matrix at one scale, and a multi_node scaling curve.
- `test_io.py` — the same matrix + scaling curve for `ModelData.to_netcdf`
  (write) and `ModelData.from_netcdf` (read). Solve-free, so no solved `Result`.
- `test_reference.py` — the realistic reference systems bundled in the package
  (`fluxopt.benchmark`, also behind `python -m fluxopt.benchmark`) at a quarter
  year. Skips on fluxopt versions that predate the module.

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

## Compare two refs — benchmem sweep

To compare performance between any two fluxopt versions or git refs (e.g. a PR
branch against `main`) with one fresh venv per ref, from the repo root:

```bash
uvx --from 'pytest-benchmem[plot]' benchmem sweep fluxopt \
    git+https://github.com/fluxopt/fluxopt@main \
    git+https://github.com/fluxopt/fluxopt@my-branch \
    --suite benchmark --copy-dir benchmark --memory --pin pytest-benchmem \
    --out "$PWD/.benchmarks/sweep"
uvx --from 'pytest-benchmem[plot]' benchmem compare .benchmarks/sweep/*.json
```

The `--pin pytest-benchmem` installs the memory harness into each ref's venv
(the suite's `--memory` pass needs it there, not just in the CLI venv).
`--copy-dir` stages the suite into sweep's isolated working directory, and
`--out` must be absolute because relative paths resolve inside that
directory. All three flags are workarounds for
[pytest-benchmem#169–171](https://github.com/fluxopt/pytest-benchmem/issues/169)
and can be dropped once fixed upstream. Heads-up: `--copy-dir benchmark` also
copies `benchmark/.venv` if you have synced it locally.

Sweep resolves one fresh venv per ref (no lockfile — it can't, the dependency
set differs per ref); add `--as-of YYYY-MM-DD` for a date-pinned resolve or
`--pin <spec>` to hold individual dependencies still.

This runs the whole suite — archetypes, IO, and the realistic reference
systems — against each ref. On PRs, the `benchmark-hint` workflow runs
`test_reference.py` the same way and posts the numbers to its job summary.

## Local memory profiling — pytest-benchmem

Peak-memory number next to the timings, plus a flamegraph of where it goes:

```bash
uv run pytest . --benchmark-only --benchmark-memory
uv run pytest . --benchmark-only --benchmark-memory --benchmark-memory-profile profiles/
uv run benchmem flamegraph profiles/ --worst peak --open
```

[CodSpeed]: https://codspeed.io
[pytest-benchmem]: https://github.com/fluxopt/pytest-benchmem
