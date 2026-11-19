# NEW_FEATURE.md

A description of a new feature to be implemented in `make_target.py`. The features is intended for cases when the target function is $\Lambda_n$, but not to $K_n$. This is a known limitation and is accepted as such. 

## Background

When using a uniformly spaced frequency grid for fitting non-centroid ($\omega_n \neq 0$) friction spectra, the magnitude of the gradient of the merit function starts oscillating when the spectral widths of the components of a `multi` embedder becomes smaller than the grid spacing. I want to implement an easy default option for generating a non-uniform grid that places more frequency points around the areas where the spectrum varies more rapidly.

## Design

There are three considerations going into the design of the non-uniform grid:

1. A subset of the grid points should be created from the frequencies returned by `WAM_discretization` from @src/glefit/utils/bath.py. `wmax` is the top end of the frequency grid as specified by the user and `nbath` is also user-specified. `wmin` specified by the user as the lower end of the frequency grid should be pre-pended to the WAM array of freqs.
2. For a range of frequencies from `max(0, nm_omega-pad_omega)` to `nm_omega`, where `pad_omega` is user-supplied, the grid should contain additional points with a spacing approximately equal to the smallest spacing produced by `WAM_disrectization` in step 1.
3. Elsewhere, additional uniformly spaced points should be added to every gap in the frequency list produced in step 1 where the gap is greater than `min_step`. The number of added points should be the smallest possible so that the resulting frequency spacing is below `min_step`.