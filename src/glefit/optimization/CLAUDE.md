# CLAUDE.md: `glefit.optimization`

This module defines the optimizers that fit embedding parameters. `src/glefit/optimize.py`, the command-line driver that runs them, is described here too.

## Background

GLEFit fits a Markovian embedding of a generalized Langevin equation to target data. An `embedder` from `glefit.embedding` holds the parameters, and a merit function from `glefit.merit` measures the misfit. The optimizer changes the embedder's mapped parameters `x`, which have no limits on their values. `merit.all()` returns the distance f, its gradient g and its Hessian H with respect to `x`. The Hessian is a finite difference of the analytic gradient.

Both optimizers take second-order steps:

```text
NR:  h = −H⁻¹ g
EF:  in the eigenbasis of H (eigenvalues bₖ, gradient components gₖ),
     hₖ = −2gₖ / (|bₖ| (1 + √(1 + 4gₖ²/bₖ²)))
```

The EF step goes downhill along every mode, including modes with negative curvature, where NR goes uphill. For small gradients it equals the NR step along modes with positive curvature. Every component hₖ is smaller than 1 in magnitude.

## Contents

| Config `type` | Class | File | Step |
| --- | --- | --- | --- |
| `NR` | `NewtonRaphson` | `NR.py` | Newton step. Converges to any stationary point, including saddle points |
| `EF` | `EigenvectorFollowing` | `NR.py` | Eigenvector-following step. Moves away from saddle points |

- `_base.py` holds `Optimizer`, modelled on ASE's optimizer, and `IOContext`, which opens and closes the output files.
- `__init__.py` holds `OPT_MAP`, which links each config `type` to its class.

```python
from glefit.optimization import EigenvectorFollowing
opt = EigenvectorFollowing(emb, merit, logfile="-", trajfile="traj.out")  # calls merit.all() once
opt.run(steps=100, options={"gtol": 1e-2, "max_step": 0.2})
opt.converged()     # NR.run and EF.run return None, so check this
emb.A               # the fitted drift matrix
```

```yaml
optimization:
  type: EF
  parameters:           # passed to the constructor
    trajfile: traj.out  # default traj.out; "-" means stdout
    logfile: log.out    # appended to; "-" means stdout; omitted means no log
  options:              # passed to run(), after optimize.py pops max_iter
    max_iter: 100
    gtol: 0.01          # default 0.05
    max_step: 0.2       # default: no limit
```

## The optimization loop

`Optimizer.__init__` opens the files and calls `initialize()`, which for NR and EF calls `merit.all()`. `run(steps, options)` then:

1. writes the starting point to the log and trajectory,
2. returns at once if `converged()`,
3. otherwise repeats `iteration()` until `converged()` or `steps` iterations. Each iteration calls `step()`, sets `emb.x = emb.x + h`, calls `update()` (one more `merit.all()`), and logs.

| Method | `Optimizer` | `NewtonRaphson` | `EigenvectorFollowing` |
| --- | --- | --- | --- |
| `run` | The loop, returns a bool | Reads `gtol` and `max_step`, returns `None` | Inherited |
| `initialize`, `update` | Do nothing | Call `merit.all()` | Inherited |
| `step` | Returns `None` | Newton step, then `rescale_step` | EF step, then `rescale_step` |
| `converged` | Raises `NotImplementedError` | max\|gᵢ\| < `gtol` | Inherited |
| `log` | Writes the trajectory | Also writes the log | Inherited, with cached eigenvalues |

`rescale_step` scales the whole step down so that its largest component equals `max_step`, which keeps its direction.

## Output

- **Trajectory** (`trajfile`, opened with `"w"`): a header `<Class> | Step   N, Time hh:mm:ss`, then the conventional parameters as `% .6e` (7 significant digits), then a blank line. `analysis/plot_objective.py` parses this format.
- **Log** (`logfile`, opened with `"a"`): at every step, the distance, max\|gᵢ\|, the gradient and the Hessian eigenvalues.
- **`optimize.py`:** writes `optimized_Ap_matrix.json` to the working directory, but only when the fit converges, and prints `YAY!`. A run that does not converge prints nothing and exits with status 0.

All output paths are relative to the working directory.

## The driver, `optimize.py`

`optimize.py CONFIG` (also `bin/optimize.py` after `source env.sh`) runs `ConfigHandler(CONFIG)`, `validate()`, then `Optimization.from_config`. That builds the embedder, loads the data, builds the merit function, and builds the optimizer with `optimization.parameters`. It then pops `max_iter` from `optimization.options` and passes the rest to `run()`. `Optimization.run` returns `opt.converged()`.

## Conventions

- **Steps are in `x`:** `max_step` and the EF bound of 1 apply to mapped parameters. With the default log mappers, a step of 1 multiplies a parameter by up to e.
- **Convergence:** only max\|gᵢ\| < `gtol` is tested. With the merit's default weights, the distance is about the mean squared relative deviation over the grid's range, so `gtol` does not depend on the number of grid points or the scale of the data. An explicit `weight` removes that normalisation.
- **Cost:** each step costs one `merit.all()`, which needs 4p + 2 gradient evaluations for p parameters.
- **Options:** `gtol` and `max_step` are read with `dict.get`, so a misspelled option is silently ignored. Unknown constructor arguments are also ignored.
- **State:** `gtol` is set in `run`, so `converged()` raises `AttributeError` before the first `run`.
- **Files:** `IOContext` closes files it opened when the optimizer is closed or garbage-collected. Open file objects passed in are left to the caller.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `ffd03fd`. Everything under Wrong results and Surprising limits was reproduced by running the code.

### Wrong results

- **EF logs the previous step's Hessian eigenvalues.** `step()` caches the eigenvalues, and `update()` replaces the Hessian without clearing the cache, so the log at step k shows the eigenvalues from step k−1. In one run the logged lowest eigenvalue was −12.6 while the current one was −4.4. The steps themselves are correct. Fix: set `self._eigvals = None` in `update()`.
- **`NR.run` and `EF.run` return `None`.** `Optimizer.run` documents a bool, but `NewtonRaphson.run` drops it. Fix: `return super().run(...)`.

### Surprising limits

- **NR goes uphill along negative curvature.** With H = diag(1, −2) and g = (1, 1), the step was (−1, +0.5). Use EF unless the start is known to be near a minimum.
- **EF fails at an eigenvalue of exactly 0.** The step is NaN in every component. This happens only when a parameter has no effect on the distance.
- **`optimization.options.max_iter` is required** by `optimize.py`, but `validate()` does not check it, so a missing value raises `KeyError`.
- **Unknown `type` values raise `KeyError`** from `OPT_MAP` instead of `ConfigError`.
- **Runs overwrite and append.** A second run in the same directory overwrites the trajectory and the JSON file, and appends to the log.

### Missing features

- Restarts are not implemented, although `run` keeps a step count for them.
- There is no line search or trust region. Only `max_step` limits a step.
- Convergence cannot be tested on the step size or the change in distance.
- There is no quasi-Newton optimizer that avoids the finite-difference Hessian.
- Final parameters are written only to the trajectory, with 7 significant digits.

### Test gaps

- `tests/test_optimize/test_optimize.py` runs `optimize.main` on three configs and checks only that no error is raised. It does not check convergence, and the `pronycos_embedder.yaml` fixture does not converge.
- Nothing tests `step`, `rescale_step`, `converged` or the log files directly.
- Running these tests from the repository root writes `log.out`, `trajectory.out` and `optimized_Ap_matrix.json` there.

### Outdated text in the code

- `NewtonRaphson` is called "quasi-Newton". It uses the full finite-difference Hessian.
- The `run` docstrings refer to `DEFAULT_MAX_STEPS`. The constant is `MAX_ITERATIONS_DEFAULT`.
- `optimize.py` imports `load_data`, which it does not use.

## Adding features

### A new optimizer

1. Subclass `Optimizer`, or `NewtonRaphson` if it uses the same quantities, as `EigenvectorFollowing` does.
2. Implement `step()` (return h for `x_new = x + h`) and `converged()`. Compute what they need in `initialize()` and `update()`.
3. If it needs options, read them in an overridden `run(steps, options)` and then call `super().run`. Return its result.
4. Extend `log()` if needed, and call `super().log(T=T)` so that the trajectory is still written.
5. Add the class to `OPT_MAP` in `__init__.py`. No other registration is needed: the config validator checks only that `type` is present.

### Other changes

- **Restarts:** read the last trajectory entry, or a checkpoint, into `emb.conventional_params` before `run`, and set `nsteps`. `restart` in the config is already validated. `ConfigHandler` has commented-out checkpoint methods.
- **New stopping rules:** override `converged()`. `self._value`, `self._grad` and `self._hess` hold the current values.

## Testing

`glefit` is not installed. The tests write output files to the working directory, so run them from a scratch directory. Start from the repository root:

```bash
REPO=$(pwd); cd "$(mktemp -d)" && PYTHONPATH=$REPO/src conda run -n tools python -m pytest $REPO/tests/test_optimize
```

A new or changed optimizer should have tests that:

- check single steps against hand-computed values for a diagonal Hessian, including negative and near-zero eigenvalues
- check that `max_step` limits the largest component and keeps the direction
- fit a target made by a known embedding, such as `Debye(η, ωc)` from `glefit.spectral`, and check that the fit converges to the known parameters
- check the trajectory format with `analysis/plot_objective.py::parse_trajectory_file`
