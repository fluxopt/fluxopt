"""fluxopt's math, and the engine that builds it.

Two layers: Elements -> this. The math is
:data:`~fluxopt.math.sources.PROGRAM`, a directory of YAML fragments that
mathspec composes into one spec, built and solved by
`specsolve <https://github.com/fluxopt/lpspec>`_;
:func:`~fluxopt.math.sources.build_sources` builds the tables bound to it straight
from the elements, and the answer is specsolve's own ``Result``.

There is no second implementation: the math is a file, so it is reviewed,
diffed, typeset and extended as one.

specsolve and mathspec are pinned to commits — their language surface is
pre-1.0 and still moves, so an unpinned ref would let a ``uv sync`` change
what the program means.

A feature the program does not express yet raises
:class:`UnsupportedFeatureError` rather than being dropped silently.
"""

from fluxopt.math.sources import (
    PROGRAM,
    UnsupportedFeatureError,
    build_sources,
    objective_weights,
    program,
)

__all__ = [
    'PROGRAM',
    'UnsupportedFeatureError',
    'build_sources',
    'objective_weights',
    'program',
]
