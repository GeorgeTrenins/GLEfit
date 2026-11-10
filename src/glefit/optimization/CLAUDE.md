# CLAUDE.md: `glefit.optimization`

Optimizers fit the parameters of an embedder by minimizing a merit function. This file also covers `src/glefit/optimize.py`, the command-line driver that runs them. It lists conventions and pitfalls. Open issues are in `ISSUES.md` at the repository root, as `OPT-n`.

## Model

An optimizer moves the embedder's mapped parameters `x`, which have no limits. `merit.all()` returns the distance f, its gradient g and its Hessian H with respect to `x`, where H is a finite difference of the analytic gradient. In the eigenbasis of H, with eigenvalues bₖ and gradient components gₖ, the steps are:

```text
NR:  hₖ = −gₖ / bₖ                 (computed as h = −H⁻¹g)
EF:  hₖ = −gₖ / (|bₖ| − λₖ)       with the shift λₖ = (|bₖ| − √(bₖ² + 4gₖ²))/2 ≤ 0
```

NR is Newton's method. It converges to the nearest stationary point and goes uphill along modes with negative curvature, so it can stop at a saddle point. EF (eigenvector following) goes downhill along every mode, so it moves away from saddle points. For small gradients it equals NR along modes with positive curvature, and |hₖ| ≤ 1 always. The code writes it as hₖ = −2gₖ/(|bₖ| + √(bₖ² + 4gₖ²)), which needs no division by bₖ, and takes no step along a mode with bₖ = gₖ = 0. Along a mode whose curvature is a rounding error, EF steps at random (OPT-15). Use EF unless the start is known to be near a minimum. Steps are in `x`: with the default log mappers, a step of 1 multiplies a parameter by up to e.

`NR.py` holds `NewtonRaphson` (config `type` `NR`) and `EigenvectorFollowing` (`EF`), which subclasses it and changes only `step` and the caching of eigenvalues. `_base.py` holds `Optimizer`, modelled on ASE's optimizer, and `IOContext`, which opens and closes the output files. `__init__.py` holds `OPT_MAP`.

## Running a fit

`Optimizer.__init__` opens the files and calls `initialize()`, which for NR and EF calls `merit.all()`. `run(steps, options)` logs the starting point, returns at once if `converged()`, and otherwise repeats `iteration()` until `converged()` or `steps` iterations, logging each. It returns whether the fit converged. `iteration()` calls `step()`, sets `emb.x = emb.x + h`, counts the step, and calls `update()`, which calls `merit.all()` again. `update()` raises `ValueError` if the distance, gradient or Hessian is not finite, naming the step as the trajectory numbers it, and leaves the embedder at those parameters. The options are `gtol` (default 0.05; converged when max|gᵢ| < `gtol`) and `max_step` (default none; `rescale_step` scales the whole step so that its largest component equals `max_step`, which keeps its direction). `initialize()` sets these defaults, so `converged()` and `step()` work before `run`, and each `run` resets any option it is not given. Unknown constructor arguments and options raise `TypeError`.

`optimize.py CONFIG` (also `bin/optimize.py` after `source env.sh`) builds the embedder, data, merit function and optimizer with `Optimization.from_config`, and passes `optimization.parameters` to the optimizer's constructor. An unknown `type` in any section raises `ConfigError` that lists the known types. It pops `max_iter` from `optimization.options` and passes the rest to `run()`. On convergence, it prints `YAY!` and writes `propagation_matrix()` to `optimized_Ap_matrix.json`. Otherwise it prints nothing and exits with status 0. After every run, also one that raises, it pickles `emb.to_dict()` with `success` to `embedder_parameters.pkl`.

## Conventions

- **Output:** paths are relative to the working directory, and `"-"` means stdout. The trajectory (`trajfile`, default `traj.out`) is overwritten. Each step has a header `<Class> | Step   N, Time hh:mm:ss`, the conventional parameters as `% .6e`, and a blank line, which `analysis/plot_objective.py` parses. The log (`logfile`, default none) is overwritten too, and holds the distance, max|gᵢ|, the gradient and the Hessian eigenvalues at each step. `IOContext` closes the files it opened when the optimizer is closed or garbage-collected, and leaves file objects passed in to the caller.
- **Convergence:** with the merit's default weights, the distance is about the mean squared relative deviation, so `gtol` does not depend on the number of grid points or the scale of the data. An explicit `weight` removes that normalisation.
- **Cost:** each step costs one `merit.all()`, which needs 4p + 2 gradient evaluations for p parameters (MER-9).
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## New optimizer

Subclass `Optimizer`, or `NewtonRaphson` if it uses the same quantities, as EF does. Implement `step()`, which returns h for `x + h`, and `converged()`, and compute what they need in `initialize()` and `update()`. Read options in an overridden `run(steps, options)` and return `super().run(...)`. If `log()` is extended, call `super().log(T=T)`, which writes the trajectory. Add the class to `OPT_MAP`: the config validator checks only that `type` is present.

## Testing

Run `PYTHONPATH=src conda run -n tools python -m pytest tests/test_optimization.py tests/test_optimize.py` from the repository root. `test_optimization.py` gives the optimizers a quadratic objective with a chosen gradient and Hessian, and checks the first NR and EF steps against hand-computed values, the EF step along a mode without curvature, `max_step`, the convergence test and the default `gtol`, the value `run` returns, the trajectory, and the rejection of unknown arguments, unknown options and a non-finite distance, gradient or Hessian. The quadratic objective is not a `BaseProperty`, which the type annotations flag, but optimizers only call its `all()`. On a fit of two Prony terms, whose Hessian changes at every step, it checks the Hessian eigenvalues in the log and that EF recovers the target. Its Hessian has a negative eigenvalue and eigenvectors that form a non-symmetric matrix, and the largest component of each step is negative: simpler choices hide bugs. `test_optimize.py` checks that a gen2x2 fit that converges at once writes the right `optimized_Ap_matrix.json`, and that `embedder_parameters.pkl` holds the fit and `success` after a converged run, a run with no steps, two steps, or a failing NR step. Pass `trajfile` to every optimizer in tests, and `monkeypatch.chdir` into `tmp_path` to run the driver, since both write to the working directory by default.
