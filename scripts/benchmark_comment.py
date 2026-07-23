#!/usr/bin/env python3
"""Render the sticky PR comment for the benchmark-hint workflow.

Usage: ``benchmark_comment.py HEAD_JSON [BASE_JSON]``

Reads ``python -m fluxopt.benchmark --json`` output for the PR head and (if the
file exists) for the base branch, and prints a markdown comment with per-model
build time and peak memory, plus deltas against the base. Stdlib-only, so it
runs regardless of which revision is checked out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MARKER = '<!-- benchmark-hint -->'


def fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f'{n / 1e6:.2f}M'
    if n >= 10_000:
        return f'{n / 1e3:.0f}k'
    return str(n)


def fmt_seconds(s: float) -> str:
    return f'{s * 1000:.0f} ms' if s < 1.0 else f'{s:.1f} s'


def fmt_mem(mib: float | None) -> str:
    if mib is None:
        return 'n/a'
    return f'{mib / 1024:.1f} GiB' if mib >= 1024 else f'{mib:.0f} MiB'


def fmt_delta(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return '—'
    return f'{(new - old) / old:+.0%}'


def build_seconds(row: dict) -> float:
    """Whole build pipeline: Elements -> ModelData -> linopy model."""
    return row['elements_s'] + row['data_s'] + row['build_s']


def main() -> None:
    head = json.loads(Path(sys.argv[1]).read_text())
    base_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    base = json.loads(base_path.read_text()) if base_path and base_path.exists() else []
    baseline = {row['model']: row for row in base if 'error' not in row}

    timesteps = next((row['timesteps'] for row in head if 'timesteps' in row), '?')
    print(MARKER)
    print('### Benchmark hint — build pipeline')
    print()
    print(
        f'`python -m fluxopt.benchmark` ({timesteps} hourly timesteps) on this PR vs its base. '
        'Shared-runner walltime is noisy — treat this as a hint, not a gate.'
    )
    print()
    print('| model | variables | constraints | build | Δ build | peak mem | Δ peak mem |')
    print('|:--|--:|--:|--:|--:|--:|--:|')
    for row in head:
        if 'error' in row:
            print(f'| {row["model"]} | failed: `{row["error"]}` | | | | | |')
            continue
        ref = baseline.get(row['model'])
        cells = [
            row['model'],
            fmt_count(row['variables']),
            fmt_count(row['constraints']),
            fmt_seconds(build_seconds(row)),
            fmt_delta(build_seconds(row), build_seconds(ref) if ref else None),
            fmt_mem(row['peak_mib']),
            fmt_delta(row['peak_mib'], ref.get('peak_mib') if ref else None),
        ]
        print('| ' + ' | '.join(cells) + ' |')
    if not baseline:
        print()
        print('_No baseline: `fluxopt.benchmark` does not exist on the base branch._')


if __name__ == '__main__':
    main()
