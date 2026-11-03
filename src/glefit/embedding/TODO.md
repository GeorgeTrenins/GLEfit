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
- Not tested yet, because they are known defects or undecided: gen2x2 at α = 0 with Γ < 0, at |α| = Γ and at Γ = 0; integer times; integer setter input; `MultiEmbedder.to_primitive`; `MultiEmbedder.grad_A` after a direct component edit.
- `legacy_tests/`: 133 pass, 31 fail (known). The 3 failures in `test_multi.py` expect the drift-matrix signs from before `e2299f6`; `docs/Auxiliary variables (SDE) - Kantorovich prescription.md` confirms the current signs (ā_p = −a_p).

## Point 3: `ISSUES.md` and a shorter module `CLAUDE.md`

- Move "Known gaps" from `src/glefit/embedding/CLAUDE.md` into a new `ISSUES.md` at the repo root, one ID per issue, so that fixes can mark them resolved.
- Cut the module `CLAUDE.md` to about 40 lines of conventions and pitfalls.
- Add the issues found in this session:
  1. Integer arrays given to the `conventional_params`, `primitive_params` or `x` setters are truncated in the mapped space: `_forward_map`, `_inverse_map`, `jac_px` and `hess_px` in `_base.py` allocate with `np.zeros_like(input)`. `primitive_params = [3, 1]` stores x = [1, 0]. Constructors convert to float, so configs are not affected.
  2. `glefit.merit` reads θ only from the top row of Ap (`merit/properties.py` lines 47, 119, 156, 201), and its gradient fills only `[:, 0, 1:]` (line 126). A wrong left column is invisible to fits but ends up in `optimized_Ap_matrix.json`, which simulations propagate.
  3. At α = 0, `docs/Generalised Brownian oscillator auxiliary variables.md` puts θ = (r, 0) (α ≥ 0), but `to_conventional` and `jac_conventional` use `alpha > 0`, giving θ = (0, r).
  4. `to_primitive` recovers Γ only to about ε·δ²/Γ² relative, because g = δ² − Ω² cancels for |Γ| ≪ δ (1–4e-11 at Γ = ±1e-3). Conditioning, not a bug.
  5. The spectrum has no second Γ-derivative at Γ = 0 (the one-sided limits differ); `spectrum_hess` returns their mean (`d2y_dGamma2 = 0`). Behaviour undecided; untested.
  6. `numderiv` with `max_step` raises `TypeError` on numpy 2 (`np.clip` without `a_min`).
  7. `mpmath.expm` never returns for NaN input. This matters for test helpers; `DriftMatrixReference` guards against it.
- Replace "fitting path" in the repo `CLAUDE.md` and the module `CLAUDE.md`.
- Update stale testing text: the repo `CLAUDE.md` test status describes the old `tests/` layout (now `legacy_tests/`); the module `CLAUDE.md` recommends `numderiv.jacobian` as a reference and runs `tests/test_merit`, which does not exist yet.

## Point 4: gen2x2 Jacobian at α = 0

- `jac_conventional` sets `jac[3, 1] = np.sign(alpha)`, which is 0 at α = 0 on the |α| ≥ Γ branch (Γ ≤ 0), so dAp/dα misses dδ/dα. Every `from_harmonic` start lands here.
- First ask the maintainer which convention applies at α = 0 (issue 3):
  - docs: `alpha >= 0` in `to_conventional`, and `jac[3, 1] = 1.0 if alpha >= 0 else -1.0`. This changes the standard form at α = 0, so state the behaviour change in the fix.
  - code: keep `alpha > 0`, and `jac[3, 1] = 1.0 if alpha > 0 else -1.0`.
- Failing test first: add an underdamped α = 0 case (for example [1.3, 0.0, 0.5, −1.7]) to the chain-rule tests in `test_gen2x2.py`.

## Point 5: small bug fixes, one commit each, failing test first

- `MultiEmbedder.to_primitive` uses `emb.nparams` instead of `emb.nparam` (AttributeError).
- gen2x2 `kernel_*` with integer times: `expcoscosh` and `expsincsinhc` allocate with `np.empty_like(t)`. Decide whether to convert in `utils/special.py` or at the top of `kernel_*`.
- Integer input to the setters (issue 1).
- `MultiEmbedder.grad_A` returns stale values after a component is edited directly. Ask the maintainer whether to fix it or to document it as unsupported (the conventions say to set values on the `MultiEmbedder`).

## Later

- Docstring corrections (pure refactor): the list under "Outdated text in the code" in the module `CLAUDE.md`; the shape convention in the `kernel()` and `spectrum()` docstrings ((ngrid,), (ndof, ngrid), (ndof, ndof, ngrid), with the grid axis kept for scalar input, as agreed); annotations that say `NDArray` where lists are accepted.
- Structural simplification of `_base.py` and `multi.py` (not planned yet).
- Review and debug `glefit.utils.numderiv` as a separate `utils` item. It builds the Hessian used in fits (`merit/_base.py` lines 519 and 781). Known: the `max_step` crash, and error paths that raise `TypeError: 'KeyError' object is not callable`.
- Undecided behaviour; ask before changing: Γ = 0; infinite gradients at |α| = Γ exactly; a stability check stricter than γ ≥ max(√max(g, 0), |α|); zero starting values; `from_dict` ignoring unknown keys.
- Outside this module: `MemorySpectrum` at Γ = 0 (`merit`); `expsincsinhc(..., nu=2, wrt="Gamma")` wrong for Γt ≥ 0.01 (`utils`).
