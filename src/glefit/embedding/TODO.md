# TODO: revision of `glefit.embedding`

Plan agreed with the maintainer on 2026-11-03/05. `CODING_ASSIGNMENT.md` and `tests/CLAUDE.md` still apply; this file records the remaining steps and what was learned. Delete items as they are done.

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

- Points 1 and 2 are done (commits `18ded66`..`fc607ac`). `tests/test_embedding/` has 267 tests and 1 strict xfail and runs in about 3 s:
  `PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding`
- `tests/test_embedding/conftest.py` provides `SympyReference` (K, Λ and their derivatives from SymPy expressions, evaluated at 30 digits). `tests/conftest.py` provides `DriftMatrixReference` (K, Λ and exact directional derivatives from a numerical drift matrix, via mpmath block matrices [[X, dX], [0, X]]). The former also filters SymPy 1.14's harmless `bitcount` DeprecationWarning.
- Not tested yet, because they are known defects or undecided: EMB-3, EMB-4 and EMB-6.
- `legacy_tests/`: 113 pass, 51 fail (known). `test_gen2x2.py::test_matrix_rotation` fails since `rotate_matrix` was removed (EMB-22), as the maintainer accepted. The 3 failures in `test_multi.py` expect the drift-matrix signs from before `e2299f6`; `docs/Auxiliary variables (SDE) - Kantorovich prescription.md` confirms the current signs (ā_p = −a_p). 19 fail since EMB-20 because they pass `sigma` to gen2x2.
- Point 3 is done (`f117fc5`). `ISSUES.md` holds the embedding issues. A fix marks its issue resolved there in the same commit.
- The gen2x2 note in `docs/` was rewritten in `b05a032`, after the [r, α, g, y] code of `290c7d3`, and extended on 2026-11-05 (a generality proof and remarks). It uses (r, γ, Ω, φ), with α = γ cos 2φ and g = γ² − Ω², and has no y. Its Ω is the irreducible frequency. The drift-matrix entry that it called ε, and the code `Omega`, is ζ (`zeta`) since `3985b1b`.

## Point 4: gen2x2 redesign (EMB-20)

Plan approved on 2026-11-05. One commit per step, in this order; delete each step once it is committed. Steps 1 (`3985b1b`), 2 (`9d29a67`) and 3 (the `from_matrix` commit, a fix for EMB-22 that also removed `rotate_matrix`) are done. Steps 4 and 5 change `gen2x2.py`, so stop after each one until the maintainer has committed it. In step 4, replace "the `from_matrix` commit" in `ISSUES.md` (EMB-20, EMB-22) by its hash.

4. `kernel_hess` (EMB-16), in the same two forms. The strict xfail becomes a test over all cases.
5. `BaseEmbedder.propagation_matrix()` returns the drift matrix. gen2x2 overrides it with the minimal gauge of the note's table, and `MultiEmbedder` assembles it block by block.
6. `optimize.py` writes `propagation_matrix()` to `optimized_Ap_matrix.json`, tested by running `main` in `tmp_path`. This closes EMB-20.
7. Remove what is unused in `utils/special.py` (all but `sinc`). Ask first: this breaks the import of `legacy_tests/test_utils/test_special.py` (34 passing tests).

`from_harmonic` stays as it is, apart from the key `zeta`.

Decisions (2026-11-04/05):
- Fit the note's (r, γ, Ω, φ) as primitive parameters, with the drift matrix in the maximal gauge, Ap = [[0, r cos φ, r sin φ], [−r cos φ, 2γ, Ω], [−r sin φ, −Ω, 0]]. This replaces [r, α, g, y] from `290c7d3`, including its plan for `kernel_hess` in (α, g, y).
- Constrain only γ (`LowerBoundMapper`). r, Ω and φ use `IdentityMapper`. K and Λ depend only on r², Ω² and cos 2φ, so every real (r, Ω, φ) with γ > 0 is a valid embedding.
- Reject r = 0 as a starting value. Allow Ω = 0.
- Spectrum fits that stall near Ω ≈ 0 are a `glefit.merit` problem (see Later), not a reason to constrain Ω.
- Kept from 2026-11-04:
  - `sigma` and `threshold` raise `TypeError`.
  - EMB-2 stays fixed: integer times give float results.
  - Add `kernel_hess` (EMB-16).
  - Add a method that returns the minimal-gauge Ap (the note's table) for propagation. `MultiEmbedder` assembles it block by block, and `optimize.py` writes it to `optimized_Ap_matrix.json` instead of `emb.A`.
  - Remove what becomes unused in `utils/special.py`. Ask first: `legacy_tests/test_utils/test_special.py` imports it.
  - Do not fix EMB-1, -5, -7, -8 or -9 separately; the redesign resolves them.
- Settled on 2026-11-05 (the steps above): keep the conventional parameters and rename ε to ζ; keep `from_matrix(theta, A)` and keep rejecting an indefinite symmetric part (EMB-21, low priority); remove `rotate_matrix`; name the new method `propagation_matrix()`; leave Λ undefined at Ω = ω = 0.

Verified on 2026-11-04/05, with no errors found in the note. The scripts and results are in `05_notebooks/gen2x2_note_verification/` (see its README).
- **SymPy (46 checks):** K and J of eq. (1); eq. (2); the maximal gauge; the properties; the Debye, SDL and BO cases against their notes; the minimal-gauge table.
- **mpmath at 50 digits (18 checks):**
  - 315 embeddings (1) give the same K and Λ after (3) and the maximal gauge.
  - The box r, γ, Ω ≥ 0, 0 ≤ φ ≤ π/2 is exactly the set of bounded two-mode kernels with J ≥ 0.
  - The minimal gauge and the whole gauge family keep K and Λ.
- **Planted errors:** all 20 were caught.
- **Fit experiment:** see Later.

Lessons for the new code:
- For G = γ² − Ω², take the G-derivative of sinhc(t√G) from `sinc` for √G·t ≤ 1 and from decaying exponentials above, as `290c7d3` does. For imaginary z, s(z) + s″(z) cancels by a factor of about |z|/2, which cost 3e-11 at √|G|·t = 660.
- Mutation testing of step 2: an unfactored Debye amplitude happens to be exactly 0 at φ = 0.4 but −1e-16 at φ = 0.3, so the Debye-limit tests use φ = 0.3 and compare element by element. The slow rate written as γ − √G is wrong by only 1e-12 at Ω/γ = 10⁻², so the long-time test uses 10⁻³. Two mutants survive and are harmless in double precision: the series/exponential switch at √G·t = 10 instead of 1, and G as γ² − Ω² instead of a product.
- The minimal-gauge table assumes r, Ω ≥ 0. With unconstrained parameters, use |Ω| or Ω² in the row condition; a negative r only flips the sign of θ. Any real φ works.
- Starting exactly at r = 0, Ω = 0 or φ = 0 stalls that parameter, because K is even in it there, so its derivative vanishes. In the experiment:
  - Ω = 0 stopped at a false convergence (residual 0.37).
  - φ = 0 escaped through roundoff after 50 steps.
  - r = 0 stopped at step 0.
- The minimal gauge jumps from θ = (r, 0) to (0, r) across φ = π/4 when Ω > γ. The kernel is the same on both sides, as the note now says.
- Leave `tutorials/` alone for now.

## Point 5: small bug fixes, one commit each, failing test first

Deferred until point 4 is done (decided 2026-11-04).

- EMB-4: `MultiEmbedder.to_primitive` uses `emb.nparams` instead of `emb.nparam` (AttributeError).
- EMB-3: integer input to the setters.
- EMB-6: `MultiEmbedder.grad_A` returns stale values after a component is edited directly. Ask the maintainer whether to fix it or to document it as unsupported (the conventions say to set values on the `MultiEmbedder`).
- EMB-17 (agreed feature): `MultiEmbedder` raises on a `mappers` argument.

## Later

- EMB-19, docstring corrections (pure refactor). The gen2x2 items go with EMB-20.
- Structural simplification of `_base.py` and `multi.py` (not planned yet).
- Review and debug `glefit.utils.numderiv` as a separate `utils` item. It builds the Hessian used in fits (`merit/_base.py` lines 519 and 781). Known: the `max_step` crash, and error paths that raise `TypeError: 'KeyError' object is not callable`.
- Undecided behaviour; ask before changing: EMB-10 and EMB-11.
- `glefit.merit`, to fix when that module is revised (maintainer, 2026-11-05). Setup: a prototype maximal-gauge embedder fitted with EF, 10 starts per target, only γ constrained.
  - All 50 kernel fits succeeded, but 3 of 30 spectrum fits stalled near Ω ≈ 0, where Λ(0) = 2γr² sin²φ/Ω² is singular.
  - Without ω = 0 in the grid, 3 of 10 fixture-data fits stopped exactly at Ω = 0 with a gradient of about 1e-14.
  - Constraining Ω as well avoided both (80/80).
  - Reproduce with `05_notebooks/gen2x2_note_verification/fit_constraints_diag.py`.
- Outside this module: see "Tracked in other modules" in `ISSUES.md`.
