# Effects

An `Effect` tracks a quantity (cost, CO2, primary energy, ...) across the
optimization horizon. One effect is the objective to minimize; others can be
bounded.

See [Effects (Math)](../math/effects.md) for the formulation.

## Defining Effects

```python
from fluxopt import Effect

# Cost effect (will be used as objective in optimize())
cost = Effect('cost')

# Tracked effect with a unit
co2 = Effect('co2', unit='kg')
```

The objective is specified in the `optimize()` call via the mandatory
`objective_effects` argument, e.g. `objective_effects='cost'`. Pass a list to
minimize a sum of effect totals, e.g. `objective_effects=['opex', 'capex']`.

## Linking Flows to Effects

Flows contribute to effects via `effects_per_flow_hour`. The value is in
effect-units per flow-hour (e.g., €/MWh):

```python
from fluxopt import Flow

# Single effect
gas_flow = Flow('gas', size=500, effects_per_flow_hour={'cost': 0.04})

# Multiple effects
gas_flow = Flow('gas', size=500, effects_per_flow_hour={'cost': 0.04, 'co2': 0.2})
```

At each timestep, the contribution is `coefficient * flow_rate * dt`.

## Bounding Effects

### Total Bounds

Limit the total effect over the entire horizon:

```python
# CO2 budget: max 1000 kg total
co2 = Effect('co2', unit='kg', maximum=1000)

# Cost floor (e.g., minimum revenue)
revenue = Effect('revenue', minimum=500)
```

### Per-Hour Bounds

Limit the effect **rate** at each timestep. The bound is specified per hour and
scales with the timestep duration (`dt`), so the constraint is
resolution-independent:

```python
# Max 50 kg CO2 per hour — allows 200 kg in a 4h timestep
co2 = Effect('co2', unit='kg', maximum_per_hour=50)

# Time-varying per-hour bound
co2 = Effect('co2', unit='kg', maximum_per_hour=[50, 40, 60, 50])
```

## Cross-Effect Contributions

An effect can include a weighted contribution from another effect using
`contribution_from`. This is useful for carbon pricing, primary energy factors,
or any chain where one tracked quantity feeds into another.

### Scalar (both domains)

A scalar factor applies to **both** domains — temporal (per-timestep) and
lump (sizing, yearly costs):

```python
effects = [
    Effect('cost', contribution_from={'co2': 50}),
    Effect('co2', unit='kg'),
]
```

Here, every kg of CO2 adds 50 to cost — both for temporal emissions
(from flow operation) and lump emissions (e.g., from `Sizing.effects_per_size`).

### Time-Varying

Use `TimeSeries` values in `contribution_from` for time-varying factors:

```python
effects = [
    Effect(
        'cost',
        contribution_from={'co2': [40, 50, 60]},  # time-varying
    ),
    Effect('co2', unit='kg'),
]
```

### Transitive Chains

Contributions chain transitively. A PE -> CO2 -> cost chain is modeled as:

```python
effects = [
    Effect('cost', contribution_from={'co2': 50}),
    Effect('co2', unit='kg', contribution_from={'pe': 0.3}),
    Effect('pe', unit='kWh'),
]
```

### Restrictions

- **No self-references**: an effect cannot reference itself
- **No cycles**: `cost -> co2 -> cost` is rejected at build time

See [Effects (Math)](../math/effects.md#cross-effect-contributions) for the
formulation.

## Accessing Results

After solving, the `Result` provides several views into effect values:

```python
result = optimize(...)

# Objective value (shortcut for the objective effect's total)
print(result.objective)

# Total effect values as (effect,) DataArray
print(result.effect_totals)

# Temporal: per-timestep effect values as (effect, time) DataArray
print(result.effects_temporal)

# Lump: sizing and fixed-cost effect values as (effect,) DataArray
print(result.effects_lump)
```

### Per-Contributor Breakdown

`stats.effect_contributions` decomposes effect totals into per-contributor parts
on a unified `contributor` dimension (flow IDs + storage IDs), matching the
model's temporal/lump domain structure:

```python
contrib = result.stats.effect_contributions

# Per-timestep contributions (contributor, effect, time) — flows only
contrib['temporal']

# Lump contributions (contributor, effect) — flows + storages
contrib['lump']

# Total per contributor: temporal summed over time + lump (contributor, effect)
contrib['total']
```

The contributions are validated against the solver totals — if they don't sum
to `effect_totals`, a `ValueError` is raised.

Cross-effects (e.g., CO2 -> cost) are attributed to the originating contributor.
If a gas flow emits CO2 priced at 50 EUR/kg, its cost contribution includes both
the direct cost and the carbon tax portion.

## Full Example

Two sources with different cost/CO2 tradeoffs, subject to an emission cap:

```python
from datetime import datetime
from fluxopt import Carrier, Effect, Flow, Port, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(3)]

demand = Flow('elec', size=100, fixed_relative_profile=[0.5, 0.8, 0.6])
cheap_dirty = Flow('elec', size=200, effects_per_flow_hour={'cost': 0.02, 'co2': 1.0})
expensive_clean = Flow('elec', size=200, effects_per_flow_hour={'cost': 0.10, 'co2': 0.0})

elec = Carrier('elec')

result = optimize(
    timesteps=timesteps,
    carriers=[elec],
    effects=[
        Effect('cost'),
        Effect('co2', maximum=100),
    ],
    ports=[
        Port('cheap', imports=[cheap_dirty]),
        Port('clean', imports=[expensive_clean]),
        Port('demand', exports=[demand]),
    ],
    objective_effects='cost',
)

print(f"Total cost: {result.objective:.2f}")
print(result.effect_totals)
```

## Parameters Summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | required | Effect identifier |
| `unit` | `str` | `''` | Unit label |
| `maximum` | `float \| None` | `None` | Upper bound on weighted total across all periods |
| `minimum` | `float \| None` | `None` | Lower bound on weighted total across all periods |
| `maximum_per_period` | `float \| None` | `None` | Upper bound on each period independently |
| `minimum_per_period` | `float \| None` | `None` | Lower bound on each period independently |
| `maximum_per_hour` | `TimeSeries \| None` | `None` | Upper bound rate (per hour), scaled by `dt` |
| `minimum_per_hour` | `TimeSeries \| None` | `None` | Lower bound rate (per hour), scaled by `dt` |
| `contribution_from` | `dict[str, TimeSeries]` | `{}` | Cross-effect factor (scalar or time-varying; lump uses `mean('time')`) |
| `period_weights` | `list[float] \| None` | `None` | Per-period weights for total aggregation (overrides global `period_weights`) |
