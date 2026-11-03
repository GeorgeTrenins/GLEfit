---
tags:
  - topicNote
aliases:
  - Drude-Lorentz spectral density
date: 2026-09-21
---
# Definition

The [[Spectral density|spectral density]] is[^wang1999][^thoss2001]
$$
J_\mathrm{D}(\omega) = \eta_0 \omega \frac{\omega_c^2}{\omega^2+\omega_c^2}
$$
and the corresponding [[Memory kernel|memory kernel]] is
$$
K_\text{D}(t) = \eta_0\omega_ce^{-\omega_c t}
$$
where
- $\eta_0$ is the static friction
- $\omega_c$ is the cutoff frequency (inverse bath correlation time)


## Harmonic discretisation

The [[Spectral density discretisation|spectral density discretisation]] according to the "logarithmic" prescription[^makri2017][^makri1999][^prada2023] gives the frequencies
$$
\omega_j = \omega_c \tan\left[
\frac{\pi}{2n_{\text{bath}}}\left(j - \frac{1}{2}\right)
\right]
$$
and coupling coefficients
$$
c_j = \omega_j \sqrt{\frac{\eta_0 \omega_c m_j}{n_{\text{bath}}}}
$$
# Sources

[^wang1999]: [[wangSemiclassicalStudy1999|Wang et al. (1999) Semiclassical study of electronically nonadiabatic dynamics in the condensed-phase]]

[^thoss2001]: [[thossSelfconsistentHybrid2001|Thoss et al. (2001) Self-consistent hybrid approach for complex systems]]

[^makri1999]: [[makriLinearResponse1999|Makri (1999) The Linear Response Approximation and Its Lowest Order Corrections]]

[^makri2017]: [[waltersDirectDetermination2017|Walters et al. (2017) Direct determination of discrete harmonic bath parameters from molecular dynamics simulations]]

[^prada2023]: [[pradaDissipativeMatsubara2023|Prada et al. (2023) Dissipative Matsubara Dynamics]]
