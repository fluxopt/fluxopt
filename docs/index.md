# fluxopt

Energy system optimization with [linopy](https://github.com/PyPSA/linopy) — progressive modeling, from simple to complex.

## Installation

```bash
pip install fluxopt
```

## Quick Example

A gas boiler covers a heat demand, minimizing fuel cost:

```python
from datetime import datetime
from fluxopt import Carrier, Converter, Effect, Flow, Port, optimize

timesteps = [datetime(2024, 1, 1, h) for h in range(4)]

gas = Carrier('gas')
heat = Carrier('heat')

# Flows
gas_source = Flow('gas', size=500, effects_per_flow_hour={'cost': 0.04})
fuel = Flow('gas', size=300)
heat_out = Flow('heat', size=200)
demand = Flow('heat', size=100, fixed_relative_profile=[0.4, 0.7, 0.5, 0.6])

result = optimize(
    timesteps=timesteps,
    effects=[Effect('cost', is_objective=True)],
    ports=[Port('grid', imports=[gas_source]), Port('demand', exports=[demand])],
    carriers=[gas, heat],
    converters=[Converter.boiler('boiler', thermal_efficiency=0.9, fuel_flow=fuel, thermal_flow=heat_out)],
)

print(f"Total cost: {result.objective:.2f}")
print(result.flow_rates)
```

## Next Steps

- **[Guide](guide/getting-started.md)** — walkthrough of the full API, from flows to storage
- **[Math](math/notation.md)** — formulation reference with notation, constraints, and examples
