# CLAUDE.md — Testing standards (pytest)

Tests are read at least as often as production code: they document intent, mark the
corner cases, and are the first thing a new maintainer runs. Write them for a human reader.

## Tooling
- Runner: `pytest` (plain `assert`, no `unittest.TestCase` boilerplate). Invoke as `pytest`; `py.test` is the legacy alias.
- Mocking: pytest's `monkeypatch` fixture or `unittest.mock` (stdlib). Do not add the `mock` backport.
- Property-based tests: `hypothesis` where an input space is better sampled than enumerated.
- Doctests: keep them as usage documentation; run them with `pytest --doctest-modules`.
- Multi-interpreter / multi-dependency matrices: `tox` (or `nox`), not ad-hoc scripts.

## Layout and naming
- Files `tests/test_<module>.py`, mirroring the package structure.
- Test functions `test_<unit>_<condition>_<expected>`. Long, descriptive names are intended — they are never called by hand and are what a failure report shows.
  - Good: `test_square_of_negative_number_is_positive`. Bad: `test_sq2`, `test_1`.
- Group related tests with plain classes (`class TestSquare:`) only when it aids reading; no inheritance.

## Writing a test
1. **One behaviour per test.** A test proves one small piece of functionality correct. Multiple `assert`s are fine only if they check that same behaviour.
2. **Arrange / Act / Assert**, visually separated by blank lines. Keep the act step to one line.
3. **Full independence.** Each test passes alone, in the suite, and in any order. No shared mutable module state; no reliance on another test's side effects.
4. **Fresh data, automatic cleanup** via fixtures (replaces `setUp`/`tearDown`):
   - Use `yield` fixtures for teardown; `tmp_path` for files; `monkeypatch` for env vars, attributes, `cwd`.
   - Default to function scope. Wider scopes (`module`, `session`) only for immutable, expensive resources.
   - Shared fixtures go in the nearest `conftest.py`; fixtures used by one file stay in that file.
5. **Parametrise instead of copy-paste:** `@pytest.mark.parametrize` with `ids=` so each case has a readable name.
6. **Explicit expectations.** Literal expected values; no re-implementing the function under test inside the test. Floats: `pytest.approx` with a stated tolerance. Exceptions: `with pytest.raises(Err, match=...)`.
7. **Mock only external boundaries** (network, clock, filesystem outside `tmp_path`, slow third-party services). Patch the name *where it is looked up* (`myapp.module.Thing`), not where it is defined. Prefer asserting on outputs over asserting on mock calls.
8. **No logic in tests** — no loops, branches or `try/except` controlling what is asserted. If it is needed, it belongs in a fixture or parametrisation.

## Speed
- Unit tests should run in milliseconds. Anything slow (large data, heavy computation, I/O) gets `@pytest.mark.slow` (registered in `pyproject.toml`) and is excluded from the default run (`-m "not slow"`); it runs in CI or on a schedule.
- Never `time.sleep` in a test; control time by injection or patching.

## Workflow for the agent
- Run the full suite **before** changing code (record the baseline) and **after** (confirm nothing else broke).
- While iterating, run only the relevant tests: `pytest tests/test_x.py::test_name -x`.
- **Bug fix = test first.** Write a test that reproduces the bug and fails, then fix the code. Keep the test as a regression guard.
- **New feature = test first.** A failing test confirms the behaviour is not already reachable through some unconnected path.
- Never weaken, skip, or delete an existing test to make the suite pass. Use `@pytest.mark.xfail(reason=..., strict=True)` only for known, documented defects.
- If work is left unfinished, leave a failing test describing the next step, and say so in the summary.
- A pre-push hook / CI job runs the full suite; do not bypass it.

## Checklist before finishing
- [ ] Each new test name reads as a sentence describing behaviour.
- [ ] Each test passes in isolation (`pytest path::test -p no:randomly`) and in random order (`pytest -p randomly`, if installed).
- [ ] No network, no real clock, no writes outside `tmp_path`.
- [ ] Fast tests stay fast; slow ones are marked.
- [ ] Full suite green.
