# Field naming grammar

Every bounded parameter in fluxopt is described by three orthogonal axes.
The field name encodes all three, so future fields name themselves.

## The three axes

1. **Quantity** — what is bounded: `rate`, `level`, `size`, `uptime`,
   `downtime`, `flow_hours`, `startups`, …
2. **Basis** — absolute (`[MW]`, `[MWh]`) or relative to size/capacity.
   Relative fields carry a `relative_` prefix (`relative_rate_min`) or are
   inherently relative (`load_factor`).

   The prefix comes *before* the quantity (not `rate_relative_min`)
   because `relative_rate` is itself a quantity, not a modifier: the
   math notation gives it its own dimensionless symbol (p̲, p̄ ∈ [0, 1])
   distinct from the rate P [MW], just as `relative_level` (e) is
   distinct from the level E [MWh]. Absolute flow bounds deliberately do
   not exist — absolute is always `size × relative`, so no
   `rate_min`/`relative_rate_min` adjacency problem can arise.
3. **Scope** — where the bound is evaluated:
   - **rate** — each timestep (scaled by Δt where applicable)
   - **periodic** — aggregate within each period, independently
   - **total** — weighted aggregate across all periods

## Grammar

```
[relative_]<quantity>_<min|max>
```

Suffix style: quantity first, bound last. This matches PyPSA
(`p_nom_min`), sorts and autocompletes by topic, and mirrors the math
notation where +/− superscripts qualify the symbol (P⁺, P⁻).

| Class | Fields |
|---|---|
| Flow | `relative_rate_min` / `relative_rate_max` |
| Storage | `relative_level_min` / `relative_level_max` |
| Status | `uptime_min` / `uptime_max`, `downtime_min` / `downtime_max` |
| Sizing / Investment | `size_min` / `size_max` |
| Effect | `total_min` / `total_max`, `periodic_min` / `periodic_max` |

For Effect, the quantity (the effect value itself) is implicit, so the
scope takes its place: `total_max` is the weighted total across periods and
`periodic_max` binds each period independently.

## Scope rules

- **Element-level aggregates are always per-period.** Operational limits
  (full-load hours, startups, active hours) recur each period; a
  weighted total of startups across a 20-year horizon is not a thing.
- **Cross-period budgets live on Effects** (`total_min`/`total_max` with
  `period_weights`). Absolute aggregate bounds on a flow are expressible
  by routing through an auxiliary effect; dedicated element fields are
  ergonomic sugar and must match the effect-route semantics.
- **Aggregates are weight-aware from day one**: Σₜ Pₜ·Δtₜ per period, so
  representative-period weighting slots in without changing semantics.

## Other conventions

- `prior_*` — values before the optimization horizon (`prior_rates`,
  `prior_level`, `prior_size`).
- `effects_per_*` / `effects_fixed*` — effect contribution dicts.
- Flat min/max field pairs on the element; a grouped dataclass only when
  parameters form a feature bundle that triggers a modeling regime
  (`Status` → binaries, `Sizing`/`Investment` → capacity variables,
  `PiecewiseConversion` → piecewise formulation).
