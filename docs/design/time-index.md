# Time index: flat `time` dim with per-period coordinates

Design decision for how fluxopt represents time in multi-period models.
Supersedes the orthogonal `time × period` grid for operational data.

**Status: implemented** (branch `feat/flat-time-index`). See
"Implementation notes" at the end for the shipped API and deliberate
deviations.

## Motivation

The current design uses two orthogonal dims: `time` (within-period
timesteps, shared by all periods) and an optional `period`. This forces
every period onto the **same** timestep grid:

- **No ragged periods.** 2030 cannot have 8760 hourly steps while 2050
  has 2190 4-hourly steps. Coarsening far-future periods is standard
  practice in pathway models — near-term decisions deserve hourly
  detail, far-future uncertainty doesn't, and MILP size shrinks where
  it hurts least.
- **No per-period resolution.** `Dims.dt` is `(time,)`, shared across
  periods, so even equal-count-different-resolution fails.
- **No real calendars.** `time` labels are generic within-period time;
  a 2030 timestep cannot carry a 2030 timestamp.
- **No datetime features for custom constraints.** Users extending the
  model should be able to write constraints against real timestamps —
  `sel(time=slice('2030-06', '2030-08'))`, `groupby('time.dt.month')`
  for monthly budgets, season/weekday masks. This requires meaningful
  datetime labels on the dimension the variables live on.

## Considered alternatives

1. **Dense orthogonal grid** (status quo) — free period isolation for
   shift-based constraints, clean `.sum('time')` reductions, but
   structurally cannot express raggedness. Rejected as the multi-period
   representation; survives unchanged as the single-period shape.
2. **pandas MultiIndex snapshots** (PyPSA style) — expresses raggedness,
   but xarray MultiIndex support is second-class: it cannot serialize to
   NetCDF (fluxopt's IO layer is netcdf groups), and broadcasting/
   selection fight the dimension model. Rejected.
3. **Masked dense grid** — keep orthogonality, size `time` to the finest
   period, mask the tail of coarser periods via linopy
   `add_variables(mask=)` / `add_constraints(mask=)`; `dt` becomes
   `(time, period)`. Keeps period isolation free and the migration
   small, and solver size matches the flat design. Rejected because the
   shared `time` axis degrades to a *positional* index in ragged mode:
   real timestamps survive only as a 2D `timestamp(time, period)` coord,
   which xarray cannot `sel` on and which kills datetime features
   (`time.dt.month`, date-range slicing) on the model variables —
   exactly what user-written custom constraints need. Complexity moves
   from the constraint layer into the public data/API surface, which is
   the wrong direction. Also limited to trailing-truncation raggedness.
4. **Flat `time` dim + ride-along coordinates** — one dimension spanning
   the full horizon, with a plain coordinate mapping each timestep to
   its period. Ragged-capable, serializes cleanly, xarray-idiomatic,
   and every timestep carries its true timestamp as the dim label.
   **Chosen.**

## The design

One temporal dimension, named `time`, spanning the whole horizon:

```
time          dim     all timesteps across all periods (real timestamps or ints)
time_period   coord   (time,) — investment period each timestep belongs to
dt            data    (time,) — timestep duration [h], may differ per period
weights       data    (time,) — operational weights (representative-period
                      occurrence counts; 1.0 by default)
period        dim     unchanged — investment-period axis (build/active
                      variables, periodic effect bounds)
period_weights data   (period,) — unchanged, period duration/discount weights
```

Key points:

- **The dim keeps the name `time`.** The change is additive: same dim,
  now with an optional `time_period` coord. Renaming to `snapshot` would
  touch every array signature, constraint, and doc for PyPSA familiarity
  only; a `rename({'time': 'snapshot'})` at the interop boundary covers
  that.
- **The coord is `time_period`, not `period`.** Coordinate names are
  global within a Dataset; `period` must stay reserved for the
  investment dim (`invest_build`, `invest_active`, `period_weights`).
  Vectorized indexing does not care about the indexer's name:
  `invest_active.sel(period=time_period)` maps `(flow, period)` →
  `(flow, time)` regardless.
- **Single code path.** Single-period models are the flat dim with no
  `time_period` coord — identical to today. Multi-period attaches the
  coord. Period-boundary masks are no-ops with one period, so every
  constraint is written once.
- **`Dims.episodes` owns boundary handling**: an `Episodes` partition
  whose boolean `starts` mask is `True` where `time_period` differs
  from its predecessor (and at t=0). Single point of truth — see
  "Episodes vs periods" below.

## Episodes vs periods

Two distinct temporal concepts share the flat axis; keeping them apart
is the load-bearing distinction of this design.

Some constraints link a timestep to the one before it: state
accumulation (`E[t] = E[t-1] + …`), duration windows, switch
transitions, ramps. These links only make physical sense if `t-1` and
`t` are **actually adjacent in modeled time**. An **episode** is a
maximal run of genuinely consecutive timesteps — a stretch where those
`t-1 → t` chains may link. An episode *boundary* is a place where the
model's time axis jumps, so every chain must be cut and restarted.

A **period** is a different kind of thing: a bookkeeping and investment
unit — "the year 2030", with its own capacity decisions, cost totals,
and objective weight. Period answers *which ledger does this timestep's
cost land in*; episode answers *did the clock jump right before this
timestep*.

Today the only place the clock jumps is between investment periods, so
episodes and periods coincide 1:1:

```
time:      |––––– 8760 h of 2030 –––––|––– 2190 steps of 2040 –––|
period:    |========== 2030 =========|========== 2040 ==========|
episodes:  |—————— episode 1 ————————|—————— episode 2 —————————|
```

They split the day representative periods (TSA) land: one investment
period aggregated to three typical days is *one* ledger containing
*three* uninterrupted runs of the clock —

```
time:      |– Jan 15 –|– Apr 20 –|– Jul 3 –|––– 2040 … –––|
period:    |========== 2030 (one ledger) ==|=== 2040 =====|
episodes:  |— ep 1 ———|— ep 2 ———|— ep 3 ——|—— ep 4 … ————|
```

— hour 23 of Jan 15 and hour 0 of Apr 20 sit next to each other in the
array but not in reality, so the storage recursion must cut there while
all 72 timesteps still account into 2030's totals. Accordingly the two
concepts own disjoint vocabulary: aggregation, weighting, and element
fields speak *period*; coupling resets, `cyclic`, and `prior_level`
speak *episode* (see Terminology below).

The `Episodes` value class (`fluxopt.constraints.episodes`) is the
single home for the boundary machinery: canonical boolean `starts`,
derived `chain_mask` / `start_positions` / `last_positions` /
`episode_ids` / `max_duration` (the duration-tracking Big-M).
`Dims.episodes` builds it from `time_period`; TSA later extends it via
`Episodes.from_changes(time_period, time_cluster)` with no constraint
changes. The constraint helpers **require** an explicit `Episodes` —
there is deliberately no single-episode default, so a multi-period
model cannot silently chain across period boundaries by omission;
custom single-episode axes pass `Episodes.single(...)`.

## Constraint semantics

Each period remains an **independent operational episode** (unchanged
semantics — today the orthogonal `period` dim gives this implicitly):

- **Temporal coupling resets at episode starts.** The SOC recursion
  (`add_accumulation_constraints`), uptime/downtime windows, and startup
  detection must not link the last timestep of one period to the first
  of the next. The constraint helpers take a required `episodes`
  parameter; status windows reset at its boundaries.
- **Cyclic storage is per-period**: level at each period's last timestep
  equals the level at that period's first (groupby first/last).
  `prior_level` applies at each period start.
- **Per-period aggregates** (`flow_hours`, `load_factor`, startups,
  Effect `periodic_min/max`) become `groupby('time_period').sum()` on
  linopy expressions instead of `.sum('time')`.
- **Effect scopes** map cleanly onto the naming grammar: `rate` = each
  timestep, `periodic` = groupby over `time_period`, `total` =
  `period_weights`-weighted sum of the periodic values.
- **Period-scoped parameters broadcast via vectorized indexing**:
  `arr.sel(period=time_period)` expands `(…, period)` data onto
  `(…, time)`.

## User API

- **Single-period: unchanged.** `timesteps=<DatetimeIndex | ints>`.
- **Multi-period, uniform grid: unchanged.** `timesteps=idx,
  periods=[2030, 2040]` replicates `idx` per period internally (labels
  shifted to each period's calendar when datetimes are used — open
  question below).
- **Multi-period, ragged: dict keyed by period label.**

  ```python
  timesteps={2030: hourly_2030_idx, 2040: four_hourly_2040_idx}
  ```

  Periods are the sorted dict keys; `dt` is derived per period by the
  existing `compute_dt` logic applied piecewise.

### Input profiles — no silent resampling

Once resolutions differ per period, "one hourly profile" is ambiguous:
resampling an intensive quantity (price, availability, relative bound)
needs `mean`; an extensive one needs `sum`. Guessing produces quietly
wrong costs. Policy:

- Time-varying inputs must align to the flat time index — either a full
  flat series, or a dict `{period: per_period_series}` with each entry
  matching that period's grid.
- An explicit helper covers the convenience case:
  `resample_profile(profile, dims, how='mean' | 'sum')`.
- `as_dataarray` length-matching stays; a series matching the *full*
  flat length is unambiguous.

## Datetime semantics on coarse grids

Datetime features work mechanically at any resolution — accessors and
slicing don't care about spacing. Three semantic rules for mixed
resolutions (all inherent to aggregation, not to this index design):

1. **Weight by `dt`, always.** `groupby('time.dt.month').sum()` counts
   timesteps and silently diverges across periods (~720/month hourly vs
   ~180 at 4-hourly). The dt-weighted form `(rate * dt).groupby(...)` is
   exact on any mix of resolutions. House rule for built-in aggregates
   already; for custom constraints it is a correctness requirement.
2. **Labels are interval starts.** Selections finer than a period's
   resolution silently miss: `time.dt.hour == 3` matches nothing on a
   4-hourly grid (hours are 0, 4, 8, …) — the constraint becomes
   vacuous in coarsened periods with no error. Sub-resolution windows
   (7–9am peak) are unrepresentable there. Document prominently.
3. **Boundary smearing.** A timestep belongs to a slice/group iff its
   start does; a block overhanging a window boundary is counted whole
   or not at all.

Mitigations, strongest first:

- **Interval-aware selection helper.** Raw label-based masks are the
  root cause: users select on point labels while timesteps are
  intervals. Provide `dims.window_weights(...)` returning per-timestep
  *fractional overlap weights* computed from `time` + `dt` (CF-bounds
  style): an hourly step inside a 7–9am window gets 1.0, a 4-hour block
  overlapping it gets 0.5, and dead selections (`hour == 3` on a
  4-hourly grid) cannot occur because overlap is interval math. Raises
  if a window has zero overlap in a period that has timesteps —
  the silent no-op becomes a loud error. Custom constraints then read
  ``(rate * dims.window_weights(...) * dt).sum()``.
- Restrict per-period frequencies to divisors of 24 h
  (1/2/3/4/6/8/12/24 h) so day-, week-, and month-boundary groupbys
  stay exact (see open question on enforcement).
- Document rule 2 prominently for users who bypass the helper.

**Aggregation-method caveat.** Uniform mean-downsampling flattens
spikes (prices, peak loads), understating the value of flexibility and
peak capacity. The same goal — less detail in far periods — is often
better served by **representative periods with weights** (TSA), which
preserve extremes at equal problem size. This index design supports
both: `weights(time)` carries occurrence counts, and per-period rep
selections are just another ragged grid. Uniform coarsening is the
first capability, not the endpoint.

## TSA outlook (representative periods)

TSA extends the **coords, not the dims** — no `(cluster, intra_time)`
grid. Rep timesteps concatenate on the flat `time` dim with a
`time_cluster(time)` coord; `weights(time)` carries occurrence counts
(the reason it stays separate from `dt`). The name follows the
ride-along rule — coords on the time dim are `time_<what>` — keeping
bare `cluster` free for a future per-cluster dim (occurrence counts,
the seasonal-storage `chronology_cluster` mapping), exactly as
`period` stays reserved for the investment dim. Values are
period-local cluster ids (tsam's per-slice `clusterOrder`).
Consequences:

- The period-boundary machinery generalizes to **episode starts**:
  cluster boundaries break SOC recursion and status windows with the
  same mask infrastructure (period starts → episode starts).
- Composition with investment periods is automatic: different cluster
  counts per period are just more raggedness; `time_period(time)` and
  `time_cluster(time)` coexist, aggregates groupby either. Episode
  starts are where `time_period` *or* `time_cluster` changes — still
  explicit input, no calendar inference.
- An orthogonal `(cluster, hour)` grid was considered and rejected on
  the established grounds: second constraint code path, and ragged
  cluster counts across investment periods reintroduce masks.
- **Seasonal storage** (Kotzur-style superposition) adds one new dim —
  the original chronology (real day sequence with a `chronology_cluster`
  mapping) — carrying only inter-period SOC linking variables.
  Structurally analogous to the `period` dim for investment; never on
  operational data.
- **Datetime caveat**: with real selected days as representatives,
  within-day features (`dt.hour`, windows) stay valid, but
  calendar-scale groupbys (`dt.month`) are silently wrong — a rep day
  stands for days across months. Calendar-scoped constraints must
  route through the occurrence mapping. Same documentation treatment
  as the sub-resolution footgun.

### Package boundary (tsam / tsam-xarray)

Aggregation tooling owns the clustering lifecycle — mappings,
disaggregation, accuracy analytics; **fluxopt never learns what a
cluster is** (beyond episode boundaries and weights). The contract is
plain data, no cross-imports:

- **Inbound** (aggregator → fluxopt): ``{period: timesteps}``,
  ``{period: profiles}``, ``{period: weights}`` (occurrence counts),
  and — once cluster episodes land — ``{period: cluster_labels}``.
  fluxopt must never *infer* rep-period boundaries from calendar-day
  changes in timestamps (breaks under segmentation / weekly clusters);
  episode boundaries are explicit inputs.
- **Outbound** (fluxopt → aggregator): solution arrays keep ``time``
  labels (the representatives' real timestamps) and the ``time_period``
  coord — enough for the aggregator's stored mapping (cluster
  assignments, occurrences) to disaggregate results back to the full
  calendar and run post-solve analytics.
- The clustering mapping is a **study artifact** (serialized
  separately, e.g. ``clustering.json``), not model data; fluxopt's
  netcdf stays clustering-agnostic. A run reproduces from
  ``model.nc + clustering.json``.

fluxopt-side prerequisites, in order: a public ``weights`` input;
per-segment ``dt`` derivation for non-contiguous rep timesteps
(consecutive-diff derivation produces garbage across day gaps — until
then, ``dt`` must be passed explicitly); cluster-episode boundary
resets for storage/status (until then, TSA feeds are only correct
without intra-period temporal coupling).

## Terminology

The temporal vocabulary sorts into three registers; names are generated
by five rules, recorded here so they survive contributors:

1. **Bare singular noun = a dim** (an axis variables can live on):
   `time`, `period`, `build_period`. Reserved for future dims:
   `cluster`, `chronology`, `scenario`, `episode` (if per-episode
   parameters ever need an axis, e.g. episode-indexed prior state).
2. **Ride-along coord on a dim = `<dim>_<what>`**: `time_period`,
   later `time_cluster`; on the chronology dim, `chronology_cluster`
   (not `cluster_of`). The prefix keeps the bare noun free for the dim.
3. **Unprefixed temporal data vars live on `time`** (`dt`, `weights`);
   anything on another dim carries the dim as prefix
   (`period_weights`, a future `cluster_weights`).
4. **`period` is accounting, `episode` is coupling topology** — neither
   register borrows the other's word. Element fields may say
   period-things (`periodic_min`), never episode-things; constraint
   helpers say episode-things, never period-things. `Storage.cyclic`
   and `prior_level` are episode semantics and documented as such.
5. **Audience split for borrowed words**: prose says "representative
   period" ↔ code says `cluster`; prose "investment period" ↔ code
   `period`; `snapshot` exists only at the PyPSA boundary. "Horizon" /
   "chunk" is reserved for the rolling-horizon driver's solver
   decomposition (state deliberately carries across chunk boundaries)
   and must never be called an episode.

## Serialization & interop

- Plain coords round-trip through NetCDF unchanged — no MultiIndex
  workarounds.
- PyPSA interop is a boundary concern: `rename({'time': 'snapshot'})`
  plus assembling a `(period, timestep)` MultiIndex from `time_period`
  and `time` reproduces PyPSA's snapshot index exactly; the inverse
  ingests one.

## Accepted trade-offs

- **Boundary masks are a permanent tax** on every time-coupled
  constraint. Mitigated by centralizing in `Dims.episodes` and the
  accumulation helper; but "impossible to get wrong" becomes "correct
  if the mask is applied".
- **Per-period reductions are groupbys**, heavier than dim sums.
- **Identical-across-periods profiles materialize** at full horizon
  length instead of broadcasting lazily from `(time,)`. Data memory
  only; variable counts are unchanged for uniform grids and *smaller*
  for ragged ones.

## Implementation notes (as shipped)

- The boundary machinery lives in the frozen `Episodes` value class
  (`fluxopt.constraints.episodes`): canonical boolean `starts`, derived
  `chain_mask`, `start_positions`/`last_positions`, `episode_ids`,
  `n_episodes`, and `max_duration(dt)` (the duration-tracking Big-M).
  `Dims.episodes` builds it from `time_period`; `Dims` additionally
  keeps `map_to_time(obj)` (period → flat time via vectorized `sel`,
  works on linopy objects) and `sum_time(obj)` / `mean_time(obj)`
  (per-period groupby, plain `.sum('time')` in single-period mode).
- `add_accumulation_constraints`, `add_duration_tracking`, and
  `add_switch_transitions` **require** an `episodes` argument — no
  single-episode default, so multi-period models cannot silently chain
  across period boundaries by omission. Custom single-episode axes pass
  `Episodes.single(...)`. Per-period `initial` values are guarded: they
  demand exactly one episode per period, so future cluster episodes
  fail loudly instead of misaligning.
- Operational input conversion lives in `_TimeMapper.to_flat`. Beyond
  the designed shapes ({period: series} dict, flat series, scalars), two
  conveniences were kept deliberately: a within-period-length series
  **tiles** across uniform periods (pre-existing behavior, unambiguous
  since flat length = n_periods × base length), and a `(period,)` array
  expands each period's value over its timesteps.
- Uniform replication shifts datetime labels by the year gap to the
  first period; integer labels get a running index offset by the span.
- `flow_hours`, `load_factor`, and `total_duration` are now **dt-weighted**
  (Σ P·Δt·w), matching docs/math/flows.md. The previous code summed
  occurrence weights only — identical on hourly grids, wrong on others;
  dt-weighting is load-bearing for ragged resolutions.
- The `window_weights` interval-overlap helper is **not yet implemented** —
  it should ship before coarse ragged grids are promoted in user docs.

## Migration touch list

- `Dims` — flat build from dict-of-indexes, `time_period` coord,
  `episodes` partition; `coords()` signature keeps working.
- `constraints/storage.py` — boundary-aware accumulation; per-period
  cyclic/initial.
- `constraints/status.py` + status blocks in `model.py` — window resets
  at episode starts.
- `model.py` — per-period aggregates (`flow_hours`, `load_factor`) via
  groupby; effect periodic/total aggregation; investment↔operation
  coupling via `sel(period=time_period)`.
- `model_data.py` builders — drop the dense `(time, period)` broadcast
  of envelopes/effect coeffs; accept dict-of-profiles input.
- `results.py` / `stats.py` — per-period KPIs via groupby; results keep
  the flat dim (`time_period` coord makes period slicing trivial).

## Resolved questions

1. **Uniform-grid label replication**: when `timesteps` + `periods` is
   given with datetime labels, replicated timesteps are **shifted into
   each period's calendar year**. Decided by the custom-constraints
   requirement: datetime features (`time.dt.month`, date slicing) only
   work if labels are real and unique across the horizon; identical
   labels per period would also collide on a flat dim.
2. **Integer time labels in ragged mode**: real timestamps are
   **required** for ragged multi-period; integer timesteps stay
   supported for single-period (and uniform multi-period, as a global
   running index).
3. **`weights` stays separate from `dt`** (no folding into a single
   per-timestep objective weighting à la PyPSA's
   `snapshot_weightings.objective`). Occurrence count and duration are
   different physical quantities that compose — a 4-hour segment
   occurring 12 times needs both — and folding them is what makes
   disaggregation and per-quantity weighting painful downstream.

## Open questions

1. **Enforce divisor-of-24h frequencies** for ragged per-period grids
   (validation error) or document-and-warn only? Leaning: warn — hard
   enforcement blocks legitimate irregular grids (e.g. rolling-horizon
   stubs), and `dt`-weighting keeps the math correct regardless.
