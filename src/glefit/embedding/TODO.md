# TODO: revision of `glefit.embedding`

Plan agreed with the maintainer on 2026-11-03/05. `CODING_ASSIGNMENT.md` and `tests/CLAUDE.md` still apply; this file records the remaining steps and what was learned. Delete items as they are done.

## Working rules agreed in the session

- Never commit. Propose one commit per concern: a pure refactor, or a fix that starts with a failing test. Messages are short: `type(module): subject`, 2–3 sentences of rationale, then `Parts of the code generated with <LLM>`.
- Before a new test file or a fix, send a short plan and wait for approval.
- Say "derivatives via chain rule w.r.t. drift matrix", never "fitting path".
- The maintainer edits files between steps. Re-read a file before editing it.
- Python runs in the `tools` conda env with `PYTHONPATH=src`. `conda run` does not forward stdin, so put scripts in the scratchpad instead of heredocs.
- Run `legacy_tests` from the scratchpad directory, with `--continue-on-collection-errors`, since `test_utils/test_special.py` cannot be imported and would otherwise stop the whole run. `legacy_tests/test_optimize` writes `log.out`, `trajectory.out` and `optimized_Ap_matrix.json` into the working directory.
- Do not use `glefit.utils.numderiv` as a test reference (unreviewed, see Later). Use SymPy, or the mpmath references in `tests/conftest.py` and `tests/test_embedding/conftest.py`.
- Show that new tests can fail: copy `src/` to the scratchpad, apply one plausible bug at a time, run the suite against the copy (`PYTHONPATH=<copy>/src`) with a timeout per run, and report any bug that is missed and whether it is equivalent. Also run each new test alone, and the suite in random order with a few seeds (`--randomly-seed=N`): `pytest-randomly` is in `tools` since 2026-11-05 and shuffles every run (`-p no:randomly` turns it off). Nothing else may be installed.

## Status

- Points 1 and 2 are done (commits `18ded66`..`fc607ac`). `tests/test_embedding/` has 326 tests and runs in about 3 s:
  `PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding`
- `tests/test_embedding/conftest.py` provides `SympyReference` (K, Λ and their derivatives from SymPy expressions, evaluated at 30 digits). `tests/conftest.py` provides `DriftMatrixReference` (K, Λ and exact directional derivatives from a numerical drift matrix, via mpmath block matrices [[X, dX], [0, X]]). The former also filters SymPy 1.14's harmless `bitcount` DeprecationWarning.
- Not tested yet, because it is undecided: EMB-6.
- `legacy_tests/`: 79 pass, 51 fail and 1 file fails to import (all known). `test_utils/test_special.py` cannot be imported since the unused special functions were removed, as the maintainer accepted. `test_gen2x2.py::test_matrix_rotation` fails since `rotate_matrix` was removed (EMB-22), as the maintainer accepted. The 3 failures in `test_multi.py` expect the drift-matrix signs from before `e2299f6`; `docs/Auxiliary variables (SDE) - Kantorovich prescription.md` confirms the current signs (ā_p = −a_p). 19 fail since EMB-20 because they pass `sigma` to gen2x2.
- Point 3 is done (`f117fc5`). `ISSUES.md` holds the embedding issues. A fix marks its issue resolved there in the same commit.
- The gen2x2 note in `docs/` was rewritten in `b05a032`, after the [r, α, g, y] code of `290c7d3`, and extended on 2026-11-05 (a generality proof and remarks). It uses (r, γ, Ω, φ), with α = γ cos 2φ and g = γ² − Ω², and has no y. Its Ω is the irreducible frequency. The drift-matrix entry that it called ε, and the code `Omega`, is ζ (`zeta`) since `3985b1b`.

## Point 4: gen2x2 redesign (EMB-20): done

Done on 2026-11-05: `3985b1b` (the note's ε renamed ζ), `9d29a67` (fit (r, γ, Ω, φ)), `9291892` (`from_matrix` from rotation invariants, EMB-22), `4a5b5f3` (`kernel_hess`, EMB-16), `4755501` (`propagation_matrix`), `4f7c107` (`optimize.py` writes it), and the commit that removed the unused special functions from `utils/special.py`. The note was checked in `05_notebooks/gen2x2_note_verification/` (see its README).

Lessons that still apply:
- Near (not on) the Debye limit Ω = γ|sin 2φ|, the small amplitude a₋ or a₊ has a relative error of about ε/(distance to the limit), from the rounding of sin 2φ alone, whatever the formula (5.8e-11 at a distance of 1e-6). Test rounding on the limit itself, with element-wise tolerances.
- Starting exactly at r = 0, Ω = 0 or φ = 0 stalls that parameter, because K is even in it there, so its derivative vanishes. In the fit experiment, Ω = 0 stopped at a false convergence (residual 0.37), φ = 0 escaped through roundoff after 50 steps, and r = 0 stopped at step 0.
- `tutorials/` still pass `sigma` and `Omega` to gen2x2. Leave them alone for now.

## Point 5: small bug fixes, one commit each, failing test first

Next. Start here. EMB-4 (`19d41c5`), EMB-3 (`66bd58d`) and EMB-17 (the EMB-17 commit) are done; in the next commit, replace "the EMB-17 commit" in `ISSUES.md` and here by its hash.

- EMB-6: `MultiEmbedder.grad_A` returns stale values after a component is edited directly. Ask the maintainer whether to fix it or to document it as unsupported (the conventions say to set values on the `MultiEmbedder`).

## Later

- EMB-19, docstring corrections (pure refactor).
- Structural simplification of `_base.py` and `multi.py` (not planned yet).
- Review and debug `glefit.utils.numderiv` as a separate `utils` item. It builds the Hessian used in fits (`merit/_base.py` lines 519 and 781). Known: the `max_step` crash, and error paths that raise `TypeError: 'KeyError' object is not callable`.
- Undecided behaviour; ask before changing: EMB-10 and EMB-11.
- `glefit.merit`, to fix when that module is revised (maintainer, 2026-11-05). Setup: a prototype maximal-gauge embedder fitted with EF, 10 starts per target, only γ constrained.
  - All 50 kernel fits succeeded, but 3 of 30 spectrum fits stalled near Ω ≈ 0, where Λ(0) = 2γr² sin²φ/Ω² is singular.
  - Without ω = 0 in the grid, 3 of 10 fixture-data fits stopped exactly at Ω = 0 with a gradient of about 1e-14.
  - Constraining Ω as well avoided both (80/80).
  - Reproduce with `05_notebooks/gen2x2_note_verification/fit_constraints_diag.py`.
- Outside this module: see "Tracked in other modules" in `ISSUES.md`.
