# Getting Started

This walkthrough builds a simple heat system end to end: define components,
optimize, and inspect results.

## The System

A gas boiler supplies heat to meet a demand profile. We minimize fuel cost.

```
gas ──▶ [boiler η=0.9] ──▶ heat ──▶ demand
 ▲
 │
grid (gas source, 0.04 €/MWh)
```

## Step by Step

### 1. Imports and Timesteps

```python
from datetime import datetime
from fluxopt import Carrier, Converter, Effect, Flow, Port, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]
```

Timesteps can be `datetime` objects or plain integers. The duration `dt` is
inferred from consecutive timestamps (here 1 h each).

### 2. Define Carriers

Carriers are energy types — nodes where flows must balance.

```python
gas = Carrier('gas')
heat = Carrier('heat')
```

### 3. Define Effects

Effects track quantities across the horizon. The objective is specified
in the `optimize()` call (defaults to `'cost'`).

```python
effects = [Effect('cost')]
```

### 4. Define Flows

Flows carry energy on a carrier. Each flow has a `size` (nominal capacity) and
optional parameters like `fixed_relative_profile` or `effects_per_flow_hour`.

```python
# Gas source: up to 500 MW, costs 0.04 €/MWh
gas_source = Flow('gas', size=500, effects_per_flow_hour={'cost': 0.04})

# Boiler fuel input and heat output
fuel = Flow('gas', size=300)
heat_out = Flow('heat', size=200)

# Heat demand: 100 MW capacity, profile sets actual demand per timestep
demand = Flow('heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
```

### 5. Define Ports and Converters

**Ports** connect flows to the outside world (sources and sinks).
**Converters** couple input and output flows with conversion equations.

```python
ports = [
    Port('grid', imports=[gas_source]),
    Port('demand', exports=[demand]),
]

converters = [
    Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat_out),
]
```

### 6. Optimize

```python
result = optimize(
    timesteps=timesteps,
    carriers=[gas, heat],
    effects=effects,
    ports=ports,
    converters=converters,
)
```

### 7. Inspect Results

```python
# Objective value (total cost)
print(result.objective)

# Flow rates for a specific flow
print(result.flow_rate('boiler(gas)'))

# All flow rates
print(result.flow_rates)

# Effect totals
print(result.effect_totals)

# Per-timestep effects
print(result.effects_temporal)
```

Flow ids are qualified as `{component}({carrier_or_id})` — e.g., `boiler(gas)`,
`grid(gas)`, `demand(heat)`.

## Next Steps

- [Flows](flows.md) — sizing, bounds, profiles, effect coefficients
- [Converters](converters.md) — boiler, heat pump, CHP, custom conversion
- [Storage](storage.md) — batteries, thermal storage
- [Effects](effects.md) — multi-effect tracking, bounds, contributions
