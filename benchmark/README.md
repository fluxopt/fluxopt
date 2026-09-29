# Benchmarks

Time + memory benchmarks for the build pipeline —
`Elements → sources → specsolve model`. The HiGHS solve is excluded
(non-deterministic, not ours to profile). Persisting a spec and its sources is
specsolve's archive, and benchmarked there.

One `benchmark()` suite, served by [CodSpeed] in CI and [pytest-benchmem] locally.

- `systems.py` — feature archetypes (`multi_node`, `status`, `piecewise`,
  `effects`, `sizing`) + `(n, timesteps)` scale tiers. Deterministic.
- `test_build.py` — the feature matrix at one scale, and a multi_node scaling curve.
- `reference.py` — the realistic reference systems, and the command that times
  them at a full year (`pixi run -e bench python reference.py`, see `docs/benchmark.md`).
- `test_reference.py` — the reference systems at a quarter year.
- `test_reference_cli.py` — smoke tests for `reference.py`; not benchmarks, so
  CodSpeed deselects them and only the CI smoke run executes them.

## Pinned env

The suite runs in pixi's `bench` environment, pinned by `pixi.lock` like every
other environment in the repo, so the CodSpeed dashboard tracks fluxopt's code,
not upstream dependency releases. `.github/workflows/update-lockfiles.yaml`
refreshes the lock monthly. `pytest.ini` makes this directory the suite's
rootdir.

```bash
cd benchmark
pixi run -e bench pytest . --codspeed                        # CodSpeed (walltime)
pixi run -e bench pytest . --codspeed --codspeed-mode memory # CodSpeed (memory / heap)
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

Compare any two fluxopt refs or released versions with one fresh venv per
ref — without touching your checkout (a dirty tree is fine). From the repo
root:

```bash
uvx --from 'pytest-benchmem[plot]' benchmem sweep fluxopt \
    git+https://github.com/fluxopt/fluxopt@main \
    git+https://github.com/fluxopt/fluxopt@my-branch \
    --suite benchmark/ --memory
uvx --from 'pytest-benchmem[plot]' benchmem compare .benchmarks/sweep/*.json
```

`compare` can also show the model-size labels the reference benchmarks record
in `extra_info` — the temporal grid, element stats and the measured
solver-model size — via `--columns`, e.g.
`--columns time,peak,extra:flows,extra:variables`; the full set is
`timesteps/periods/components/flows/effects/series/variables/nonzeros/constraints`
(pytest-benchmem >= 0.4.12).

The suite, reference systems included, comes from your checkout and only
fluxopt changes per ref. A ref whose element API differs from the checkout's
therefore fails rather than compares; for such a pair, switch branches instead.

Sweep resolves one fresh venv per ref (no lockfile — it can't, the dependency
set differs per ref); add `--as-of YYYY-MM-DD` for a date-pinned resolve or
`--pin <spec>` to hold individual dependencies still.

## Or: switch branches

Zero extra installs — run the suite on each branch in the pinned env
(fluxopt is installed editable, so each switch is picked up; needs a clean
tree):

```bash
cd benchmark
pixi run -e bench pytest . --benchmark-only --benchmark-memory --benchmark-json head.json
git switch main
pixi run -e bench pytest . --benchmark-only --benchmark-memory --benchmark-json base.json
git switch -
pixi run -e bench benchmem compare base.json head.json
```

Both flows run the whole suite — archetypes and the realistic reference
systems. On PRs, the `benchmark-hint` workflow runs `test_reference.py` the
same way and posts the numbers as a sticky comment.

## Comparison with flixopt

`flixopt_stress.py` is a 1:1 port of the packaged `stress` reference system
to [flixopt](https://github.com/flixOpt/flixopt) — same fixed-seed
parameters, identical binary counts. It is standalone (flixopt is not a
dependency of this env); run it via
`uv run --no-project --with 'flixopt==7.2.3' python benchmark/flixopt_stress.py`.
Measured numbers and methodology: `docs/benchmark.md`.

## Local memory profiling — pytest-benchmem

Peak-memory number next to the timings, plus a flamegraph of where it goes:

```bash
pixi run -e bench pytest . --benchmark-only --benchmark-memory
pixi run -e bench pytest . --benchmark-only --benchmark-memory --benchmark-memory-profile profiles/
pixi run -e bench benchmem flamegraph profiles/ --worst peak --open
```

[CodSpeed]: https://codspeed.io
[pytest-benchmem]: https://github.com/fluxopt/pytest-benchmem
