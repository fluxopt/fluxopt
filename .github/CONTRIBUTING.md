# Contributing to fluxopt

Contributions are welcome — bug reports, code, docs, examples.

## Setup

The environment is [pixi](https://pixi.sh)'s. It installs Python and every
tool from `pixi.lock`, so there is nothing else to install.

```bash
git clone https://github.com/fluxopt/fluxopt.git
cd fluxopt
pixi run pre-commit-install   # lint every commit, as CI does
pixi run test
```

## Workflow

1. Create a branch from `main`.
2. Make changes, and add tests for them under `tests/`.
3. Run `pixi run ci`, which is everything CI checks.
4. Push and open a PR.

## Checks

| Command                         | What it runs                                                               |
| ------------------------------- | -------------------------------------------------------------------------- |
| `pixi run lint`                 | ruff, pyrefly, prettier, taplo, typos, zizmor and nbstripout, via lefthook |
| `pixi run test`                 | the test suite                                                             |
| `pixi run docs-build`           | the docs site, `--strict`, with every notebook executed                    |
| `pixi run -e bench bench-smoke` | the benchmark suite, each function once                                    |
| `pixi run ci`                   | all of the above                                                           |

`pixi run -e py313 test` and `pixi run -e py314 test` run the suite on newer
interpreters. CI runs the floor, Python 3.12, and Windows.

```bash
pixi run test tests/test_data.py   # a single file
pixi run test -k "keyword"         # by keyword
```

## Pull requests

Merges are squashed, so the **PR title** becomes the commit on `main`. It is a
[conventional commit](https://www.conventionalcommits.org) subject, and the
`Conventional commit subject` check enforces it:

```text
<type>[(scope)][!]: <subject>

feat(storage): a storage may bound its final level
fix(flow): flow-hour aggregates weigh each timestep by its duration
```

The types are `feat`, `fix`, `perf`, `refactor`, `docs`, `chore`, `test`,
`ci`, `build`, `style` and `revert`.

A `feat`, `fix`, `perf`, `refactor`, `docs` or `revert` PR adds its title,
with a link to the PR, under `## Upcoming version` in `CHANGELOG.md`. The
`Changelog line` check enforces it. The label `no changelog` opts a PR out.
See [RELEASING.md](https://github.com/fluxopt/fluxopt/blob/main/RELEASING.md).
