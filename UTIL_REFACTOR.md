# Utilities refactor

Goal: according to the guidelines in @CODING_ASSIGNMENT.md, fix the bugs and simplify the code in @src/glefit/spectral and @src/glefit/utils


## spectral module

The purpose of this module is to provide tools for:
1. Generating fitting targets for Markovian embedding optimization (`data` section in the config file as parsed by `glefit.config.config_handler`)
2. Generating the initial guess for the embedder parameters to be optimized

### Generating fitting targets

The module defines two analytical spectral density objects, exponentially damped Ohmic and Debye. These implement the kernel (`K`) and the spectrum (`Lambda`), which a script to be implemented in the utils module (see below), should use to generate the `target.csv` file (see @src/glefit/config/CLAUDE.md) given the time or frequency grid, respectively, specified by the user. The target may either be the plain kernel/spectrum or its transform for a "normal-mode frequency" $\tilde{\omega}_n$. For the spectrum, the transformed target $\Lambda(\omega; \tilde{\omega}_n)$ reads
$$
\Lambda(\omega; \tilde{\omega}_n) = \theta(x) \sqrt{x} \Lambda(\omega \sqrt{x}) , \qquad \text{where} \qquad x = 1 - \frac{\tilde{\omega}_n^2}{\omega^2}
$$
and where $\theta$ is the Heaviside step function; for the kernel,
$$
K(\omega; \tilde{\omega}_n) = \frac{2}{\pi} \int_{0}^{\infty} \Lambda(\omega; \tilde{\omega}_n) \cos(\omega t) \, d \omega = \frac{2}{\pi} \int_{0}^{\infty} \frac{\Lambda(\omega) \omega^2}{\omega^2 + \tilde{\omega}_n^2} \cos(t \sqrt{\omega^2 + \tilde{\omega}_n^2}) \, d \omega.
$$

### Generating the initial guess

The initial guess for the embedding parameters may either come from the exact embedding (only for Debye, see @docs) or by computing the Walters, Allen, Makri discretisation of the target spectral density (see @src/glefit/utils/bath.py), which corresponds to a sum of shifted Drude-Lorentz spectral densities with vanishingly small relaxation rates $\gamma$, and then increasing $\gamma$ as done in @tutorials/spectrum_optimization.ipynb 

## utils module

The following are used by different parts of the embedder code and are not to be removed, only checked for bugs:
- `glefit.utils.numderiv`
- `glefit.utils.special`

The remaining functionality is only to be kept if they are needed for generating the inputs for the optimization code as set out in this document.