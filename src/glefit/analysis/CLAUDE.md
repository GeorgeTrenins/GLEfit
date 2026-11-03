# CLAUDE.md: `glefit.analysis`

This module holds tools for inspecting a finished fit. It has one script, `plot_objective.py`.

## Background

GLEFit fits a Markovian embedding to target data, such as a memory kernel K(t) or a spectrum Λ(ω). During a fit, the optimizer in `glefit.optimization` writes the embedding's conventional parameters at each step to a trajectory file. `plot_objective.py` rebuilds the embedder and merit function from the same config file, and plots the fitted curve against the target at each step of that trajectory.

## Contents

```bash
plot_objective.py CONFIG [--traj TRAJ]    # also bin/plot_objective.py after `source env.sh`
```

`TRAJ` defaults to `traj.out` in the working directory. The window shows the target (red), the fitted total (black) and, for a `multi` embedder, each component (dashed), with a slider over the steps. The title shows the distance at the selected step.

| Function | Does |
| --- | --- |
| `parse_trajectory_file(path)` | Returns step numbers and parameter arrays from a trajectory |
| `setup_from_config(config_path)` | Builds the embedder and merit function, then turns off the embedder's mappers |
| `compute_contributions(embedder, merit, params)` | Sets `embedder.conventional_params`, then evaluates the property for each component and for the total |
| `plot_trajectory(merit, embedder, steps, params_list)` | Draws the figure and slider, then calls `plt.show()` |

## Dependencies on other modules

- **Trajectory format:** the parser expects the format written by `Optimizer.log`: a line containing `|` and `Step N,`, then the array, then a blank line. Values have 7 significant digits, so the plotted curves use rounded parameters.
- **Config:** read with `ConfigHandler`, as in `optimize.py`. The `optimization` section is ignored.
- **Merit function:** uses `function()`, `distance`, `grid` and `target`. Components are plotted by setting `merit.emb` to each one in turn.
- **Embedder:** uses the private attributes `_embs` and `_mappers`. Setting `_mappers = None` turns off range checks on the values read back.

## Conventions

- **Display:** needs an interactive Matplotlib backend. `plt.show()` blocks until the window closes.
- **Axes:** the y-range is fixed from the target, from min(0, target) to max(target) with a 5% margin, so curves outside it are cut off.
- **Style:** start each file with the `@File`/`@Time`/`@Author`/`@Desc` header, then `from __future__ import print_function, division, absolute_import`.

## Known gaps

Checked at commit `ffd03fd`. Everything under Wrong results was reproduced by running the code. With a `multi` config, a 6-step trajectory was parsed and plotted without error.

### Wrong results

- **Only `multi` configs work.** For any other embedder, `setup_from_config` raises `UnboundLocalError`, because its `else` branch sets `emb._mappers` instead of `embedder._mappers`. Reproduced with `tests/test_optimize/fixtures/gen2x2_embedder.yaml`.

### Missing features

- The figure cannot be saved without the interactive window.
- There is no plot of the distance or gradient against step, although the log file has them.
- The trajectory is not checked against the config, so a trajectory from a different embedder fails only when its parameters are set.

### Test gaps

- There are no tests.

## Adding features

- **New analysis scripts:** put them here, build objects with `ConfigHandler` and the registries as `setup_from_config` does, and add a symlink in `bin/`.
- **Saving figures:** add an `--output` option that calls `fig.savefig` for chosen steps, so the script can run with the `Agg` backend.
- **Reusing the parser:** `parse_trajectory_file` is the only reader for trajectory files. If `Optimizer.log` changes its format, update both together.

## Testing

`glefit` is not installed. With `matplotlib.use("Agg")`, the functions can be tested without a display:

```python
from glefit.analysis.plot_objective import parse_trajectory_file, setup_from_config, plot_trajectory
```

Tests should run a short fit in a temporary directory with `glefit.optimize.main`, then check that `parse_trajectory_file` returns at most `max_iter + 1` steps of length `emb.nparam`, and that `setup_from_config` and `plot_trajectory` work for every embedder type.
