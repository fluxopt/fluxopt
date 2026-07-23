#!/usr/bin/env python3
"""Render the sticky PR comment for the benchmark-hint workflow.

Usage: ``benchmark_comment.py HEAD_JSON [BASE_JSON]``

Reads ``python -m fluxopt.benchmark --json`` output for the PR head and (if the
file exists) for the base branch, and prints a markdown comment with per-model
build time and peak memory, plus deltas against the base. Stdlib-only, so it
runs regardless of which revision is checked out.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('head_json', type=Path)
    parser.add_argument('base_json', type=Path, nargs='?')
    parser.add_argument('--repo', default='fluxopt/fluxopt', help='owner/name, for the replication one-liner')
    parser.add_argument('--base', default='main', help='base ref, for the replication one-liner')
    parser.add_argument('--head', default='<your-branch>', help='head ref, for the replication one-liner')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    head = json.loads(args.head_json.read_text())
    base_path = args.base_json
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
    print()
    print('<details>')
    print('<summary>Replicate locally (CI walltime is noisy)</summary>')
    print()
    print('```bash')
    print('uv run --project benchmark benchmem sweep fluxopt \\')
    print(f'    git+https://github.com/{args.repo}@{args.base} \\')
    print(f'    git+https://github.com/{args.repo}@{args.head} \\')
    print('    --suite benchmark/ --memory')
    print('uv run --project benchmark benchmem compare .benchmarks/sweep/*.json')
    print('```')
    print()
    print('One fresh venv per ref; covers the archetype, IO and reference-system benchmarks.')
    print('</details>')


if __name__ == '__main__':
    main()
