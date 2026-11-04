# TODO: revision of `glefit.embedding`

Plan agreed with the maintainer on 2026-11-03/04. `CODING_ASSIGNMENT.md` and `tests/CLAUDE.md` still apply; this file records the remaining steps and what was learned. Delete items as they are done.

## Working rules agreed in the session

- Never commit. Propose one commit per concern: a pure refactor, or a fix that starts with a failing test. Messages are short: `type(module): subject`, 2–3 sentences of rationale, then `Parts of the code generated with <LLM>`.
- Before a new test file or a fix, send a short plan and wait for approval.
- Say "derivatives via chain rule w.r.t. drift matrix", never "fitting path".
- The maintainer edits files between steps. Re-read a file before editing it.
- Python runs in the `tools` conda env with `PYTHONPATH=src`. `conda run` does not forward stdin, so put scripts in the scratchpad instead of heredocs.
- Run tests from the scratchpad directory. `legacy_tests/test_optimize` writes `log.out`, `trajectory.out` and `optimized_Ap_matrix.json` into the working directory.
- Do not use `glefit.utils.numderiv` as a test reference (unreviewed, see Later). Use SymPy, or the mpmath references in `tests/test_embedding/conftest.py`.
- Show that new tests can fail: copy `src/` to the scratchpad, apply one plausible bug at a time, run the suite against the copy (`PYTHONPATH=<copy>/src`) with a timeout per run, and report any bug that is missed and whether it is equivalent. Also run the suite in reverse order and each test alone; `pytest-randomly` is not installed and nothing may be installed.

## Status

- Points 1 and 2 are done (commits `18ded66`..`fc607ac`). `tests/test_embedding/` has 240 tests and 1 strict xfail and runs in about 3 s:
  `PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding`
- `conftest.py` provides `SympyReference` (K, Λ and their derivatives from SymPy expressions, evaluated at 30 digits) and `DriftMatrixReference` (K, Λ and exact directional derivatives from a numerical drift matrix, via mpmath block matrices [[X, dX], [0, X]]). It also filters SymPy 1.14's harmless `bitcount` DeprecationWarning.
- Not tested yet, because they are known defects or undecided: EMB-1 to EMB-8.
- `legacy_tests/`: 133 pass, 31 fail (known). The 3 failures in `test_multi.py` expect the drift-matrix signs from before `e2299f6`; `docs/Auxiliary variables (SDE) - Kantorovich prescription.md` confirms the current signs (ā_p = −a_p).
- Point 3 is drafted and awaits review: `ISSUES.md` holds the embedding issues as EMB-1 to EMB-19, and the module `CLAUDE.md` is cut to conventions and pitfalls. A fix marks its issue resolved in `ISSUES.md` in the same commit.

## Point 4: gen2x2 Jacobian at α = 0 (EMB-1, EMB-5)

- `jac_conventional` sets `jac[3, 1] = np.sign(alpha)`, which is 0 at α = 0 on the |α| ≥ Γ branch (Γ ≤ 0), so dAp/dα misses dδ/dα. Every `from_harmonic` start lands here.
- First ask the maintainer which convention applies at α = 0 (EMB-5):
  - docs: `alpha >= 0` in `to_conventional`, and `jac[3, 1] = 1.0 if alpha >= 0 else -1.0`. This changes the standard form at α = 0, so state the behaviour change in the fix.
  - code: keep `alpha > 0`, and `jac[3, 1] = 1.0 if alpha > 0 else -1.0`.
- Failing test first: add an underdamped α = 0 case (for example [1.3, 0.0, 0.5, −1.7]) to the chain-rule tests in `test_gen2x2.py`.

## Point 5: small bug fixes, one commit each, failing test first

- EMB-4: `MultiEmbedder.to_primitive` uses `emb.nparams` instead of `emb.nparam` (AttributeError).
- EMB-2: gen2x2 `kernel_*` with integer times. Agreed fix: rewrite `kernel_func` and `kernel_grad` in the style of `rpmdgle/sysbath/spectral/brownian.py` (three regimes in Γ, whole-array expressions, `expm1`); formulas in `ISSUES.md`.
- EMB-3: integer input to the setters.
- EMB-6: `MultiEmbedder.grad_A` returns stale values after a component is edited directly. Ask the maintainer whether to fix it or to document it as unsupported (the conventions say to set values on the `MultiEmbedder`).
- EMB-17 (agreed feature): `MultiEmbedder` raises on a `mappers` argument.

## Later

- EMB-19, docstring corrections (pure refactor).
- Structural simplification of `_base.py` and `multi.py` (not planned yet).
- Review and debug `glefit.utils.numderiv` as a separate `utils` item. It builds the Hessian used in fits (`merit/_base.py` lines 519 and 781). Known: the `max_step` crash, and error paths that raise `TypeError: 'KeyError' object is not callable`.
- Undecided behaviour; ask before changing: EMB-7 to EMB-11.
- Outside this module: see "Tracked in other modules" in `ISSUES.md`.
