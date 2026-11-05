# CODING_ASSIGNMENT.md

Goal: fix bugs and simplify the structure of existing code. No new features unless explicitly requested.

## Source of truth (highest first)
1. The maintainer's instructions in the session.
2. Docstrings, comments citing equations/papers, `docs/`.
3. `legacy_tests/` (regression oracle; may encode bugs).
4. Module `CLAUDE.md` files (initial agent inspection; unverified).
5. Current behaviour of the code.
If sources conflict, or it is unclear whether behaviour is a bug or intended: stop and ask. Never silently change results.

## Scope
- In scope: bug fixes; removing dead code, unused parameters/options and speculative abstractions (e.g. ABCs with one implementation, hooks for "future" features); flattening convoluted class designs.
- Out of scope unless asked: new features, performance optimisation, dependency changes, repo-wide reformatting, renaming public symbols.
- Prefer incremental restructuring over rewrites. Do not rewrite a module from scratch.

## Workflow
1. Before touching a unit, make sure tests in `tests/` cover its current behaviour; if not, add a minimal characterisation test first.
1. Do not commit anything yourself - everything goes through human review first.
1. Each proposed commit is **either** a pure refactor (no behaviour change; all tests and `legacy_tests` results unchanged) **or** a bug fix (begins with a failing test that the fix makes pass). Never both.
1. One concern per commit, ideally < ~200 changed lines. Message: `refactor(<module>): …` / `fix(<module>): …` with a one-paragraph rationale.
1. Numerical outputs must match reference values to a relative tolerance of 1.0e-12 after a refactor. Any intended change in results is a `fix` and must be justified.
1. Remaining non-obvious conventions are documented in the docstring of the affected unit and summarised in the module `CLAUDE.md`.

## Module CLAUDE.md files
- Contain conventions and known pitfalls only; aim for ≤ ~40 lines; readable by humans.
- Treat as hypotheses. When the code contradicts them, update the file in the same commit as the related change.
- Open issues live in `ISSUES.md` (not in CLAUDE.md); mark them resolved with the fixing commit.

## Testing
- New tests go in `tests/`, following `tests/CLAUDE.md`.
- `legacy_tests/`: do not edit. Run before and after every refactor; report any change in pass/fail status. Port valuable cases into `tests/` in condensed form.
- Keep the suite reviewable: every test must be able to fail for a plausible bug. No tautological tests (testing mocks, constants, the language/library, or re-implementing the code under test).
- Commands: `pytest tests`, `pytest --continue-on-collection-errors legacy_tests`