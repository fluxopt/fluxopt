# Benchmark

fluxopt ships a user-runnable benchmark that builds a few realistic energy
systems and reports how fast the build pipeline
(`Elements → ModelData → linopy model`) runs on *your* hardware — and how much
memory it peaks at:

```console
$ python -m fluxopt.benchmark
fluxopt 0.9.0 — build-pipeline benchmark
Python 3.13.2 · Darwin arm64 · 8 CPUs
8760 hourly timesteps (1.0 years)

model              variables  constraints  elements    data   build  peak rss
-----------------------------------------------------------------------------
district_heating        140k         298k      6 ms   59 ms  182 ms   208 MiB
industry_park           289k         534k      5 ms   71 ms  481 ms   255 MiB
green_city              245k         499k      7 ms  212 ms  511 ms   264 MiB
energy_transition      1.96M        3.85M     53 ms   51 ms  603 ms   1.2 GiB
```

**peak rss** is the whole build subprocess's OS-level high-water mark — it
catches every allocation (numpy buffers, solver C libraries) but includes the
~140 MiB interpreter-and-imports footprint and allocator slack: the number
that has to fit in your RAM. For allocator-level numbers (net of the
interpreter, attributable to code), run the same systems under
[pytest-benchmem](https://github.com/fluxopt/pytest-benchmem) via the
repository's `benchmark/test_reference.py` — that is what the CodSpeed
dashboard and the PR benchmark hint report.

Each system is built in a fresh subprocess, so peak memory is attributed per
model, and all input data is deterministic — two runs of the same version on
the same machine measure the same workload.

## The reference systems

The models are realistic and readable — constant and time-varying data,
several effects, and cross-effect couplings (CO₂ priced into cost at
45 €/t via `Effect.contribution_from`). Their builders in
`fluxopt/benchmark.py` double as worked examples:

- **`district_heating`** — a municipal utility: gas boiler, ramp-limited CHP
  and an air-source heat pump with a weather-driven COP feed a 20 MW-peak
  heat network backed by a hot-water tank. Seasonal gas tariff, day-ahead
  electricity prices, hourly grid CO₂ intensity.
- **`industry_park`** — a factory site: two steam boilers with minimum load,
  minimum up/down times and startup costs (unit commitment), a gas-engine CHP
  with a piecewise-linear part-load curve, and investment decisions for an
  electrode boiler and a steam accumulator with annualized capital cost and
  embodied CO₂. Site emissions carry an annual CO₂ cap.
- **`green_city`** — a sector-coupled city: a wind PPA with a contracted
  annual energy cap, rooftop PV and a grid connection supply the city load, a
  battery sized by the optimizer (with a 10 % reserve level), and two
  district-heating networks. Tracks cost, CO₂ and primary energy.
- **`energy_transition`** — `green_city` planned over eight five-year
  investment periods (2025–2060), each represented by a full hourly year:
  demand grows with electrification, grid CO₂ intensity falls, the carbon
  price rises from 45 to 130 €/t, and the battery becomes a multi-period
  `Investment` — 15-year lifetime, overnight capex falling along a learning
  curve, recurring O&M. About 2 million variables at the default horizon.

## Options

```console
$ python -m fluxopt.benchmark district_heating   # a single system
$ python -m fluxopt.benchmark --timesteps 720    # one month instead of a year
$ python -m fluxopt.benchmark --solve            # also time the HiGHS solve
$ python -m fluxopt.benchmark --json             # machine-readable output
```

The solve is excluded by default: solver time depends on HiGHS, not on
fluxopt, and is much less deterministic than the build.

!!! note "Regression benchmarks"
    This command answers "how fast is fluxopt *here*". Tracking performance
    *between versions* is done by the CodSpeed suite in the repository's
    `benchmark/` directory.
