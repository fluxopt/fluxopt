# Flows

A `Flow` represents energy transfer on a carrier. Flows are the building blocks —
every port, converter, and storage is defined through its flows.

See [Flows (Math)](../math/flows.md) for the formulation.

## Basic Construction

```python
from fluxopt import Carrier, Flow

heat = Carrier('heat')

# Minimal: just a carrier
f = Flow(heat)

# With capacity
f = Flow(heat, size=100)  # 100 MW nominal capacity
```

## Sizing

`size` sets the nominal capacity \(\bar{P}_f\) in MW. When set, all relative
parameters are scaled by this value:

```python
# Sized: flow rate bounded to [0, 100] MW
f = Flow(heat, size=100)

# Unsized: flow rate bounded to [0, ∞)
f = Flow(heat)
```

## Bounds

`relative_minimum` and `relative_maximum` set per-timestep bounds as fractions
of `size`:

```python
# Minimum load 30%, maximum 100% → [30, 100] MW
f = Flow(heat, size=100, relative_minimum=0.3)

# Time-varying maximum
f = Flow(heat, size=100, relative_maximum=[1.0, 0.8, 0.6, 1.0])
```

## Fixed Profiles

`fixed_relative_profile` pins the flow to an exact profile scaled by `size`:

```python
# Demand: 40, 70, 50, 60 MW
f = Flow(heat, size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])
```

This sets both lower and upper bounds equal to the profile value.

## Effect Coefficients

`effects_per_flow_hour` assigns cost or emission coefficients to a flow.
Values are in units per flow-hour (e.g., €/MWh):

```python
gas = Carrier('gas')

# Constant cost
f = Flow(gas, size=500, effects_per_flow_hour={'cost': 0.04})

# Multiple effects
f = Flow(gas, size=500, effects_per_flow_hour={'cost': 0.04, 'co2': 0.2})

# Time-varying cost
f = Flow(gas, size=500, effects_per_flow_hour={'cost': [0.02, 0.08, 0.04]})
```

See [Effects](effects.md) for how these coefficients feed into the objective
and constraints.

## Id Qualification

Flow ids are auto-qualified by the parent component. You rarely need to set
`id` explicitly:

```python
from fluxopt import Carrier, Port

elec = Carrier('elec')

f = Flow(elec)
Port('grid', imports=[f])
# f.id is now 'grid(elec)'
```

Set `id` to disambiguate when a component has multiple flows on the same carrier:

```python
f1 = Flow(elec, id='base')
f2 = Flow(elec, id='peak')
Port('plant', imports=[f1, f2])
# f1.id = 'plant(base)', f2.id = 'plant(peak)'
```

## Parameters Summary

| Parameter | Type | Default | Description |
|---|---|---|---|
| `carrier` | `Carrier` | required | Carrier this flow connects to |
| `id` | `str` | `''` | Optional id (auto-qualified by parent) |
| `size` | `float \| Sizing \| None` | `None` | Nominal capacity [MW] or [investment](sizing.md) |
| `relative_minimum` | `TimeSeries` | `0.0` | Lower bound as fraction of size |
| `relative_maximum` | `TimeSeries` | `1.0` | Upper bound as fraction of size |
| `fixed_relative_profile` | `TimeSeries \| None` | `None` | Fixed profile as fraction of size |
| `effects_per_flow_hour` | `dict[str, TimeSeries]` | `{}` | Effect coefficients per flow-hour |
| `status` | `Status \| None` | `None` | [On/off behavior](status.md) (semi-continuous, startup costs, durations) |
| `prior_rates` | `list[float] \| None` | `None` | Flow rates [MW] before the horizon (for [status](status.md#prior-historical-state) initial conditions) |
