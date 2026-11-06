# TODO: revision of `glefit.embedding`

Agreed with the maintainer on 2026-11-03/05. `CODING_ASSIGNMENT.md` and `tests/CLAUDE.md` still apply. This file holds the working rules, the next steps and what was learned. Delete items as they are done.

## Working rules agreed in the session

- Never commit. Propose one commit per concern: a pure refactor, or a fix that starts with a failing test. Messages are short: `type(module): subject`, 2–3 sentences of rationale, then `Parts of the code generated with <LLM>`.
- Before a new test file or a fix, send a short plan and wait for approval.
- Say "derivatives via chain rule w.r.t. drift matrix", never "fitting path". Say "Debye limit", never "fold".
- In code, comments and test names, keep lines to 80 characters, and do not cite `docs/` or equation numbers: the notes do not ship with the code.
- The maintainer edits files between steps. Re-read a file before editing it.
- Python runs in the `tools` conda env with `PYTHONPATH=src`. `conda run` does not forward stdin, so put scripts in the scratchpad instead of heredocs, and write input to a file instead of piping it: a pipe into `conda run python` reads nothing and fails silently.
- Run `legacy_tests` from the scratchpad directory, with `--continue-on-collection-errors`, since three files cannot be imported and would otherwise stop the whole run. `legacy_tests/test_optimize` writes `log.out`, `trajectory.out` and `optimized_Ap_matrix.json` into the working directory.
- Use SymPy, or the mpmath references in `tests/conftest.py` and `tests/test_embedding/conftest.py`.
- Show that new tests can fail: copy `src/` to the scratchpad, apply one plausible bug at a time, run the suite against the copy (`PYTHONPATH=<copy>/src`) with a timeout per run, and report any bug that is missed and whether it is equivalent. Also run each new test alone, and the suite in random order with a few seeds (`--randomly-seed=N`; `-p no:randomly` turns `pytest-randomly` off). Nothing may be installed.

## Status

- The plan's points 1–5 are done, up to `ae075a0`: tests for the module, `ISSUES.md`, the gen2x2 redesign (EMB-20) and the small fixes.
- `numderiv` is reviewed and fixed, from `67412e2`: only `jacobian(f, x, *, order)` is left, tested in `tests/test_utils/test_numderiv.py`.
- `glefit.merit` is revised, from `bcad8d3`: `BaseScalarProperty` removed, merit functions tested in `tests/test_merit/test_distance.py`, `MemorySpectrum` fixed where A cannot be diagonalized, Ω of gen2x2 log-mapped, and docstrings corrected.
  - On 2026-11-06 the maintainer reversed the 2026-11-05 decision to constrain only γ, after the fit experiment was re-run with the shipped embedder. Log-mapping Ω makes fits independent of the time unit and kept all 60 spectrum fits away from the two bad points described in `ISSUES.md`, but does not remove them. The scripts are `05_notebooks/gen2x2_note_verification/exp_*.py`.
- `tests/test_embedding/` has 363 tests and runs in about 3 s:
  `PYTHONPATH=src conda run -n tools python -m pytest tests/test_embedding`
- `tests/test_embedding/conftest.py` provides `SympyReference` (K, Λ and their derivatives from SymPy expressions, evaluated at 30 digits). `tests/conftest.py` provides `DriftMatrixReference` (K, Λ and exact directional derivatives from a numerical drift matrix, via mpmath block matrices [[X, dX], [0, X]]).
- `legacy_tests/`: 73 pass, 51 fail and 3 files fail to import, all known; the root `CLAUDE.md` lists why.
- `ISSUES.md` holds the embedding issues. A fix moves its issue to "Resolved" in the same commit.

## Next

Agenda set by the maintainer on 2026-11-05: `numderiv` and `glefit.merit` are done; the structural simplification is next.

1. Structural simplification of `_base.py` and `multi.py`, planned with the maintainer on 2026-11-06. Done from `a62a49d`: `_A` and `_grad_A` removed, `nparam` and `ndof` as properties, one loop for the mapper methods, one implementation of `kernel()` and `spectrum()`, `MultiEmbedder` reading its components, its setters checking lengths, EMB-19, setters that change nothing when they reject a value, and errors that name the rejected parameter. Left, one commit each:
   - fix: reject unknown constructor keywords, which are now ignored. Also correct `mappers/CLAUDE.md`, which says that `MultiEmbedder` ignores `mappers`.
   - refactor(merit): merge `BaseProperty` into its only child, `BaseArrayProperty`.

## Lessons

- Near (not on) the Debye limit Ω = γ|sin 2φ|, the small amplitude a₋ or a₊ has a relative error of about ε/(distance to the limit), from the rounding of sin 2φ alone, whatever the formula (5.8e-11 at a distance of 1e-6). Test rounding on the limit itself, with element-wise tolerances.
- Starting exactly at r = 0, Ω = 0 or φ = 0 stalls that parameter, because K is even in it there, so its derivative vanishes. In the fit experiment, Ω = 0 stopped at a false convergence (residual 0.37), φ = 0 escaped through roundoff after 50 steps, and r = 0 stopped at step 0. Ω = 0 is now rejected by its mapper.
- A bug can vanish at a lucky test point: the eigendecomposition in `MemorySpectrum` was exact at critical damping for γ = 1.5 but wrong by up to 2 for γ = 1 and 2. Scan a few values before choosing a test case.
- Leave `tutorials/` alone for now.
