"""Enforce that every ``<!-- verified-by: ... -->`` anchor in docs/math resolves.

Any LaTeX equation in ``docs/math/*.md`` that is backed by a structural test
should sit next to an anchor of the form::

    <!-- verified-by: tests/math/test_doc_formulas.py::<test_name_or_class::method> -->

This test collects all such anchors and fails if the target file or test
function is missing — so renaming a test without updating the doc pointer
(or vice versa) fails CI loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_MATH = REPO_ROOT / 'docs' / 'math'
ANCHOR_RE = re.compile(r'<!--\s*verified-by:\s*(\S+?)::(\S+?)\s*-->')


def _collect_anchors() -> list[tuple[str, str, str]]:
    """Scan docs/math/*.md for `verified-by` anchors. Returns (md_file, target_path, nodeid)."""
    anchors: list[tuple[str, str, str]] = []
    for md in sorted(DOCS_MATH.rglob('*.md')):
        anchors.extend(
            (str(md.relative_to(REPO_ROOT)), m.group(1), m.group(2)) for m in ANCHOR_RE.finditer(md.read_text())
        )
    return anchors


ANCHORS = _collect_anchors()


def test_at_least_one_anchor_exists() -> None:
    """Guardrail so an accidental regex change does not silently disable the check."""
    assert ANCHORS, 'No verified-by anchors found in docs/math/ — did the regex break?'


@pytest.mark.parametrize(('md_file', 'target_path', 'nodeid'), ANCHORS)
def test_anchor_resolves(md_file: str, target_path: str, nodeid: str) -> None:
    """Target file exists and defines the referenced test function."""
    target = REPO_ROOT / target_path
    assert target.exists(), f'{md_file}: anchor points to missing file {target_path!r}'

    # nodeid may be "ClassName::method_name" or just "function_name"
    leaf = nodeid.rsplit('::', 1)[-1]
    content = target.read_text()
    assert re.search(rf'\bdef {re.escape(leaf)}\s*\(', content), (
        f'{md_file}: anchor {target_path}::{nodeid} — no def {leaf}(...) in target file'
    )
