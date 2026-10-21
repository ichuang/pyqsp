# Working on pyqsp

pyqsp constructs polynomial approximations, synthesizes quantum signal processing
(QSP) phases, and evaluates their responses. These instructions apply throughout
the repository; read any more specific `AGENTS.md` in the paths you touch.
Update affected instructions when changing a documented workflow.

## Environment and checks

Use an isolated environment with the core development dependencies in
`base_requirements.txt`, prepared using your preferred tooling.
Check `setup.py` for runtime support and `.github/workflows/run_tests.yml` for CI
configuration.
`optional_requirements.txt` lists Cirq; `tf_requirements.txt` aliases that list
and does not restore retired training. Neither is needed for core development.

Run source checks from the repository root. Confirm that imports resolve to this
checkout, especially when using worktrees or an editable installation:

```sh
python -c "import pyqsp; print(pyqsp.__file__)"
MPLBACKEND=Agg python -m pytest pyqsp/test/test_angle_sequence.py
MPLBACKEND=Agg python -m pytest pyqsp/test
python -m flake8 . --select=E9,F63,F7,F82 --show-source
```

- Start with tests for the affected behavior; the single-module command is an
  example. Run the full core suite for shared numerical, dispatch, or import
  changes. The flake8 selection matches blocking CI checks; broader style
  checks are advisory.
- For documentation changes, verify paths, instructions, and affected examples.
  Numerical tests are unnecessary for prose-only edits.
- For packaging changes, build and install the artifact in a clean environment.
  Check imports and the `pyqsp` console entry point outside the source tree,
  confirming that imports come from the installation.

Before substantial numerical changes, record the relevant baseline, environment,
and existing failures. Reuse successful checks while their inputs remain unchanged.
`make test` runs the core suite and `make lint` runs blocking source checks.
`make` (also `make build`, `make dist`, or `make core`) builds an sdist and wheel
without installing dependencies or the package. Build dependencies must already
be available. `PYTHON` selects the interpreter for Makefile targets. See the
developer workflow migration note in `README.md` for the changed `make test`
behavior.

Core workflows must work without TensorFlow/Cirq. Retirement tests always run:
named training entry points remain importable and raise actionable errors without
loading TensorFlow/TFQ. For optional circuit changes, run `test_qsp_models.py`
with Cirq installed; its explicit skips otherwise do not count as feature coverage.
Preserve phase evaluation/plotting, duck-typed phase extraction from existing
model objects, and `plot_loss(history)` without importing TensorFlow.

## Where to work

Read `README.md`, the relevant implementation and tests, and `CITATION` for
algorithm references. Keep API conventions in docstrings and user examples in
the README; keep change rationale and validation evidence in commit history.

- `pyqsp/poly.py`: targets, approximation, and normalization. Generator return
  types and scale metadata vary.
- `pyqsp/angle_sequence.py`: ordinary preprocessing, convention adapters, method
  dispatch, exception translation and acceptance; symmetric public synthesis.
- `pyqsp/LPoly.py`, `pyqsp/completion.py`, `pyqsp/decomposition.py`: Laurent
  polynomials, Low algebra, completion, and factorization.
- `pyqsp/sym_qsp_opt.py`: symmetric phases, Jacobians, and Newton solver.
- `pyqsp/response.py`, `pyqsp/phases.py`: responses, plotting, and predefined phases.
- `pyqsp/main.py`: CLI and registries; `setup.py`: packaging and console entry point.
- `pyqsp/qsp_models/`: optional Cirq evaluation and retired training stubs;
  `pyqsp/test/`: tests.

Run the maintained example with `python -m pyqsp.sym_qsp_min_example`.
`docs/sym_qsp_aux_files/` contains historical exploratory scripts.

## Scope and compatibility

- Keep changes tied to the requested outcome. Preserve unrelated work and avoid
  unrelated formatting. Reuse existing utilities where they fit; introduce or
  revise abstractions, dependencies, and tooling as needed for the task.
- Inspect callers, examples, and CLI paths before treating an interface as internal.
  For public API or CLI changes, check downstream impact and update affected tests
  and documentation. Make incompatible changes explicit and provide migration
  guidance; use compatibility adapters or deprecation where warranted.
- Check declared dependency support before using newer APIs. Update metadata and
  CI when intentionally changing support. Keep optional backend imports confined
  to their features.
- Document new or changed algorithms' assumptions, validity domain, and error or
  convergence expectations, with references or derivations where available.
  Exploratory methods can be supported by reproducible numerical evidence;
  distinguish established guarantees, empirical behavior, and open questions.
  Preserve citation and license credit.

## Numerical validation

- Use an independent expectation: a closed-form result, cited reference, or
  separately derived computation. Round trips and structural invariants alone
  can miss shared basis, sign, or scaling errors.
- Test the quantity the API promises. For signed or complex amplitudes, magnitude
  or probability checks alone can miss sign or phase errors. Phase sequences need
  not be unique; compare their responses under the stated convention.
  For bug fixes, retain the triggering input and show failure before the fix
  when practical, including when generalizing an existing test.
- Select edge cases and invariants relevant to the change: parity, degree,
  endpoints, invalid inputs, finiteness, normalization, and unitarity.
  For the existing formulations, completion checks include `F*~F + G*~G = 1`
  in the real Laurent case and `|P(x)|**2 + (1-x**2)*|Q(x)|**2 = 1` in the
  polynomial picture. Use the corresponding identities for new formulations.
  Sampling or local optimization alone does not certify a global bound.
- Justify tolerances against scale, conditioning, and promised accuracy.
  Do not relax them or drop difficult cases to make tests pass. Use local random
  generators with explicit seeds; performance claims need reproducible settings
  and timings.
- Inspect assertions as well as results. Returning phases does not establish
  convergence. Iterative APIs that promise accuracy must distinguish success
  from exhaustion or failure; unsupported inputs need actionable errors.

## Completion

Review the diff for unintended changes and generated artifacts. Briefly report
what changed, validation actually performed, and remaining limitations or failures,
distinguishing new failures from the baseline.
