---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# fluxopt

Energy system optimization with [linopy](https://github.com/PyPSA/linopy) — detailed dispatch, scaled to multi-period planning.

[![PyPI](https://img.shields.io/pypi/v/fluxopt)](https://pypi.org/project/fluxopt/)
[![Downloads](https://img.shields.io/pypi/dm/fluxopt)](https://pypi.org/project/fluxopt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Get Started](notebooks/01-quickstart.ipynb){ .md-button .md-button--primary }
[GitHub](https://github.com/FBumann/fluxopt){ .md-button }

</div>

---

<div class="landing" markdown>

<div class="grid cards" markdown>

-   :material-cube-outline: __Composable elements__

    ---

    Build models from `Flow`, `Bus`, `Converter`, `Storage`, and `Effect` — clear separation of physics, costs, and topology.

-   :material-chart-line: __xarray-native__

    ---

    Time series, parameters, and results as `xr.Dataset` — vectorized constraints via [linopy](https://github.com/PyPSA/linopy).

-   :material-tune: __Sizing & status__

    ---

    Capacity optimization and on/off behavior as first-class concerns, not bolt-ons.

-   :material-rocket-launch: __HiGHS out of the box__

    ---

    Open-source MIP solver bundled. Swap in Gurobi, CPLEX, or any linopy-supported backend.

-   :material-book-open-page-variant: __Math, documented__

    ---

    Every constraint has a formulation page with notation, derivation, and the line of code that emits it.

-   :material-puzzle: __Companion ecosystem__

    ---

    Lean core, optional companions for [plotting](https://fbumann.github.io/fluxopt-plot/latest/), [YAML loading](https://fbumann.github.io/fluxopt-yaml/latest/), and (planned) interactive marimo apps.

</div>



{%
   include-markdown "../README.md"
   start="<!--quickstart-start-->"
   end="<!--quickstart-end-->"
%}

## Where to next

<div class="grid cards" markdown>

-   :material-school: __Learn by example__

    ---

    Seven executable notebooks from quickstart through investment and piecewise conversion.

    [:octicons-arrow-right-24: Notebooks](notebooks/01-quickstart.ipynb)

-   :material-function-variant: __Math reference__

    ---

    Notation, objective, and per-element formulations.

    [:octicons-arrow-right-24: Math](math/notation.md)

-   :material-api: __API reference__

    ---

    Auto-generated from source — every public class, every parameter.

    [:octicons-arrow-right-24: API](api/fluxopt/index.md)

</div>

</div>
