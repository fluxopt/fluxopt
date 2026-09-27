# Which layer owns which rule

fluxopt refuses a bad model in four places. That is one more than most
projects need and fewer than it looks, because each answers a question the
others cannot — but only if a rule is written in the layer that can answer it.

This page exists because two of them had drifted. A rule about a single
field's value was being enforced on a materialised array three layers away,
and seven checks sat in a layer that could never reach them. Neither was
visible without going looking, so the point of writing the ownership down is
that the next check has an obvious home and the next reader can tell when one
is in the wrong one.

## The four

| | layer | answers | fires when |
|---|---|---|---|
| 1 | the **element** — pydantic on `elements.py` / `components.py` | is this one element internally coherent? | the user constructs it |
| 2 | the **system** — `validation.validate_system` | do these elements refer to each other resolvably? | `FlowSystem(...)`, and `ModelData.build` |
| 3 | the **data** — `ModelData.__post_init__` and each container's | is this table self-consistent? | building it, *and* reloading it |
| 4 | the **bind** — specsolve, with the program's `assumptions:` | does this data fit the program, and hold what it assumes? | every bind: `solve`, `build`, a sweep |

### 1. The element

A rule decidable from one element's own fields. `size_max >= size_min`,
`uptime_max >= uptime_min`, `PiecewiseConversion needs >= 2 flows`,
`method` being one of four literals.

This is the layer with the best error, and it is not close. It fires on the
value the user typed, names the field, and quotes what was passed:

```
Sizing(size_min=-5, size_max=9)
  -> size_min: Input should be greater than or equal to 0 [input_value=-5]
```

**If a rule can be stated here, state it here.** The same rule enforced at
layer 3 reads `Sizing.size_min < 0 on [np.str_('f')]` — the entity
reconstructed from an array coordinate, several hundred lines from the
mistake.

### 2. The system

A rule needing more than one element: duplicate ids, a flow naming a carrier
nobody declared, an effect referenced but never defined, a node not in its
carrier's node list, an objective naming an effect that does not exist.

Layer 1 cannot see any of these, because an element does not know what else
exists. `validate_system` runs on **every** path into the data layer, which
is what makes it the place to put such a rule *once*.

### 3. The data

A rule about a materialised table's internal consistency: `pair_converter`
naming a converter the table does not carry, `governed_by` naming a component
without a Status, a ramp limit on a flow with no size.

Two things reach this layer that the earlier ones could not see, and both
are real reasons for it to exist.

**Reload.** `ModelData` round-trips
through a directory of parquet tables, and a file that was hand-edited — or
written by an older version — never passed through layers 1 and 2. Every check here is answering
"could this table have arrived broken?", and the honest test for whether one
belongs is:

> Could a caller reach this through the public API without layer 1 or 2
> having already refused it?

If not, the check is dead. Seven were: `Unknown effect {k!r} in ...` in five
container builders, all of them behind `validate_system`'s sweep of the same
element models. They were reachable only through the private container API.

**A value the element could not see** is no longer this layer's. A
`Variate` may be a `ProfileRef` — a name pointing at numbers supplied later —
so `Storage` refuses `eta_charge=1.5` at construction and cannot refuse
`eta_charge=ProfileRef(...)`. That range, and every other range on a value
the program reads, is an `assumptions:` entry in the fragment that declares
the parameter (layer 4). It fires on the numbers that actually reach the
program, whether they came from a resolved reference, a reloaded file or a
table the caller edited.

What stays here is structure a table can get wrong on reload: a reference to
an entity the table does not carry, a literal outside its set, a cycle in
`contribution_from`, a ramp on a flow with no size. A check that duplicates
layer 1 for that reason — `PiecewiseData.method` is a `Literal` on the
element and re-checked here — earns its place, and its docstring should say
so. Two value rules stay because the program cannot state them: a status
flow's lower bound above zero (the program cannot tell a flow's own status
from its component's), and `Investment.lifetime` (bound only as the window
derived from it).

### 4. The bind

Whether the data fits the program: unknown labels for a declared dimension,
a column typed `float` where the file says `bool`, a missing lookup column, a
constant side the parameters do not cover, a null bound, a duplicate
coordinate.

**Do not write these.** specsolve already does, against the program's own
declarations, and its messages name the parameter, the dimension, the
offending values *and* the rewrite. Anything fluxopt writes here is a second
implementation of a check the binder is going to run anyway — and one that
cannot see the program, so it will be the weaker of the two.

**A range on a value the program reads is written here, as an assumption.**
`size_min <= size_max` is `size_bounds_are_ordered` in `sizing.yaml`; the
typeset program prints it, and specsolve refuses a table that breaks it with
the entry's name and description. `tests/math/test_assumptions.py` breaks
each one.

## Deciding

```
Can one element answer it alone?                  -> 1, the element
Does it need to see other elements?               -> 2, the system
A range on a value the program reads?             -> 4, an assumptions: entry
Table structure only a reload can break?          -> 3, the data
Is it about shape, dtype, or coverage?            -> 4, leave it to specsolve
```

Two smells worth naming, both of which had occurred:

- **A rule that has to reconstruct which entity failed** is in too deep a
  layer. It had the entity when the user wrote it.
- **A check nothing can reach** is a rule that moved up a layer and left a
  copy behind. Delete the copy; the earlier one is the one with the better
  message.

## Where this leaves layer 3

The reload half of it stays only as long as `ModelData` serializes itself
field by field. If a reloaded file were rebuilt through `ModelData.build`
rather than reconstructed around it, layers 1 and 2 would run on the way back
in and there would be nothing left for reload to catch. That is a consequence
of the tidy-frames re-cut rather than a reason for it, but it is a second
reason.

The `ProfileRef` half has moved to the program's assumptions, which check
values that arrive later wherever they arrive. So layer 3 is down to the
structure of its own tables.
