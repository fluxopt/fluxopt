"""Test helpers re-exported for tests in this directory.

The ``optimize`` fixture itself lives in ``tests/conftest.py`` and is
auto-discovered by pytest via the conftest tree. ``ts`` and ``waste`` are
re-exported here so tests in this directory can use ``from .conftest import``.
"""

from conftest import ts, waste  # noqa: F401 — re-exported for test imports
