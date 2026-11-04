# TODO: revision of `glefit.embedding`

Plan agreed with the maintainer on 2026-11-03/04. `CODING_ASSIGNMENT.md` and `tests/CLAUDE.md` still apply; this file records the remaining steps and what was learned. Delete items as they are done.

## Working rules agreed in the session

- Never commit. Propose one commit per concern: a pure refactor, or a fix that starts with a failing test. Messages are short: `type(module): subject`, 2–3 sentences of rationale, then `Parts of the code generated with <LLM>`.
- Before a new test file or a fix, send a short plan and wait for approval.
- Say "derivatives via chain rule w.r.t. drift matrix", never "fitting path".
- The maintainer edits files between steps. Re-read a file before editing it.
- Python runs in the `tools` conda env with `PYTHONPATH=src`. `conda run` does not forward stdin, so put scripts in the scratchpad instead of heredocs.
- Run `legacy_tests` from the scratchpad directory: `legacy_tests/test_optimize` writes `log.out`, `trajectory.out` and `optimized_Ap_matrix.json` into the working directory.
- Do not use `glefit.utils.numderiv` as a test reference (unreviewed, see Later). Use SymPy, or the mpmath references in `tests/conftest.py` and `tests/test_embedding/conftest.py`.
- Show that new tests can fail: copy `src/` to the scratchpad, apply one plausible bug at a time, run the suite against the copy (`PYTHONPATH=<copy>/src`) with a timeout per run, and report any bug that is missed and whether it is equivalent. Also run the suite in reverse order and each test alone; `pytest-randomly` is not installed and nothing may be installed.

## Status

- Points 1 and 2 are done (commits `18ded66`..`fc607ac`). `tests/test_embedding/` has 240 tests and 1 strict xfail and runs in about 3 s:
  `PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding`
- `tests/test_embedding/conftest.py` provides `SympyReference` (K, Λ and their derivatives from SymPy expressions, evaluated at 30 digits). `tests/conftest.py` provides `DriftMatrixReference` (K, Λ and exact directional derivatives from a numerical drift matrix, via mpmath block matrices [[X, dX], [0, X]]). The former also filters SymPy 1.14's harmless `bitcount` DeprecationWarning.
- Not tested yet, because they are known defects or undecided: EMB-1 to EMB-8.
- `legacy_tests/`: 133 pass, 31 fail (known). The 3 failures in `test_multi.py` expect the drift-matrix signs from before `e2299f6`; `docs/Auxiliary variables (SDE) - Kantorovich prescription.md` confirms the current signs (ā_p = −a_p).
- Point 3 is done (`f117fc5`). `ISSUES.md` holds the embedding issues. A fix marks its issue resolved there in the same commit.
- The gen2x2 note in `docs/` was revised (`4590c6d`..`67441e9`) and now describes the agreed redesign, EMB-20. It replaces the planned fix of EMB-1 in the current code.

## Point 4: gen2x2 redesign (EMB-20)

- The design is in `ISSUES.md` (EMB-20) and `docs/Generalised Brownian oscillator auxiliary variables.md`. Do not fix EMB-1, EMB-5, EMB-7, EMB-8 or EMB-9 separately.
- Decided on 2026-11-04: EMB-2 is fixed inside EMB-20; `sigma` and `threshold` raise an error; `kernel_hess` is added (EMB-16); a new embedder method returns the Ap for propagation.
- Done: the θ-column fix of `grad_wrt_A` in `glefit.merit`, tested in `tests/test_merit/test_properties.py` (`c1c86e3`).
- Remaining steps, one commit each:
  1. `feat(embedding)`: EMB-20 in `gen2x2.py`, including the closed forms of EMB-2 and the error for `sigma` and `threshold`. The `SympyReference` and `DriftMatrixReference` cases in `test_gen2x2.py` move to [r, α, g, y].
  2. `feat(embedding)`: `kernel_hess` (EMB-16).
  3. The method that returns the Ap for propagation, and `optimize.py` writing it to `optimized_Ap_matrix.json`.
  4. Remove what becomes unused in `utils/special.py`. Ask first: `legacy_tests/test_utils/test_special.py` imports it.
- Pitfalls: d/dg of sinhc(t√g) is (cosh(t√g) − sinhc(t√g))/(2g), which cancels for small |g|t², so use a series there. y(γ) cancels near the boundary (absolute error 1.5e-11 at y = −5).

## Point 5: small bug fixes, one commit each, failing test first

Independent of point 4, so they can go first.

- EMB-4: `MultiEmbedder.to_primitive` uses `emb.nparams` instead of `emb.nparam` (AttributeError).
- EMB-3: integer input to the setters.
- EMB-6: `MultiEmbedder.grad_A` returns stale values after a component is edited directly. Ask the maintainer whether to fix it or to document it as unsupported (the conventions say to set values on the `MultiEmbedder`).
- EMB-17 (agreed feature): `MultiEmbedder` raises on a `mappers` argument.

## Later

- EMB-19, docstring corrections (pure refactor). The gen2x2 items go with EMB-20.
- Structural simplification of `_base.py` and `multi.py` (not planned yet).
- Review and debug `glefit.utils.numderiv` as a separate `utils` item. It builds the Hessian used in fits (`merit/_base.py` lines 519 and 781). Known: the `max_step` crash, and error paths that raise `TypeError: 'KeyError' object is not callable`.
- Undecided behaviour; ask before changing: EMB-10 and EMB-11.
- Outside this module: see "Tracked in other modules" in `ISSUES.md`.
