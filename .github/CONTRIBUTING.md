# Contributing to pyqsp

Bug reports, documentation improvements, and pull requests are welcome. For bug
reports, include a small reproducible example, the expected result, and your
Python and pyqsp versions. Please open an issue to discuss substantial changes
before starting work.

## Development

Use an isolated Python environment with the dependencies in
[`base_requirements.txt`](../base_requirements.txt) and `flake8`. From the
repository root, run:

```sh
MPLBACKEND=Agg python -m pytest pyqsp/test
python -m flake8 . --select=E9,F63,F7,F82 --show-source
```

## Pull requests

Keep each PR focused and explain the problem it addresses. Include tests for
changed behavior and update affected documentation. For numerical changes, test
against an independent expected result and explain the chosen tolerances.

Draft PRs are welcome for work in progress. CI runs automatically; address
failing checks before requesting review, and respond to reviewer feedback.
