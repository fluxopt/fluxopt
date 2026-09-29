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
| 2 | the **system** — `validation.validate_system` | do these elements refer to each other resolvably? | `FlowSystem(...)`, and `build_sources` |
| 3 | the **sources** — `math.sources.build_sources` | what only the whole built system can answer | building them |
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
exists. `validate_system` runs on **every** path into the sources, which
is what makes it the place to put such a rule *once*.

### 3. The sources

Two rules, both about the built system as a whole, and both beyond what the
program can state:

- **A cycle in `contribution_from`.** One `Effect` sees only its own
  sources; `build_sources` walks the whole graph.
- **A status flow's floor above zero.** `Flow` refuses a zero floor under a
  `Status` it can see; a `ProfileRef` supplies its numbers at build. The
  program cannot state it, because a flow's own status and its component's
  are one relation to it.

This layer used to re-check tables for **reload**, since the built data saved
and loaded itself as parquet and a hand-edited file never passed layers 1
and 2. The archive replaced that: specsolve saves the spec and its sources,
and a caller who edits a table is checked by the spec's assumptions (layer
4). With no reload, those re-checks guarded nothing a caller could reach,
which is the test for a dead check:

> Could a caller reach this through the public API without layer 1 or 2
> having already refused it?

A range on a value the program reads is an `assumptions:` entry in the
fragment that declares the parameter (layer 4). It fires on the numbers that
actually reach the program, whether they came from a resolved `ProfileRef`
or a table the caller edited.

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
A rule on the built data the program can't state? -> 3, the data
Is it about shape, dtype, or coverage?            -> 4, leave it to specsolve
```

Two smells worth naming, both of which had occurred:

- **A rule that has to reconstruct which entity failed** is in too deep a
  layer. It had the entity when the user wrote it.
- **A check nothing can reach** is a rule that moved up a layer and left a
  copy behind. Delete the copy; the earlier one is the one with the better
  message.

## Where this leaves layer 3

Two rules. Both belong to the build rather than to a stored table, so they
move wherever the build moves.
