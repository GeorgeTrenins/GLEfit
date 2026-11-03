---
tags:
  - topicNote
  - stub
aliases:
  - SDL spectral density
date: 2026-10-06
---
# Definition

The [[Spectral density|spectral density]] is $$ J_\mathrm{SDL}(\omega) = \lambda\gamma\,\omega \left[\frac{1}{(\omega-\omega_0)^2+\gamma^2} + \frac{1}{(\omega+\omega_0)^2+\gamma^2}\right] $$ and the corresponding [[Memory kernel|memory kernel]] is $$ K_\text{SDL}(t) = 2\lambda\, e^{-\gamma t}\cos(\omega_0 t) $$ where

- $\lambda$ is the reorganisation energy, $\lambda = \frac{1}{\pi}\int_0^\infty \frac{J(\omega)}{\omega}\,\mathrm{d}\omega = \frac{1}{2}K(0)$
- $\gamma$ is the Lorentzian half-width
- $\omega_0$ is the peak shift 

The static friction is $$ \eta_0 = \lim_{\omega\to 0}\frac{J(\omega)}{\omega} = \int_0^\infty K(t)\,\mathrm{d}t = \frac{2\lambda\gamma}{\omega_0^2+\gamma^2} $$

# Equivalent forms

With $\bar{\omega}^2 \equiv \omega_0^2+\gamma^2$, $$ J_\mathrm{SDL}(\omega) = \frac{2\lambda\gamma\,\omega\,(\omega^2+\bar{\omega}^2)}{(\omega^2-\bar{\omega}^2)^2+4\gamma^2\omega^2} $$
# Limits

- $\Omega\to 0$: [[Debye spectral density]] with $\eta_0 = 2\lambda/\gamma$, $\omega_c = \gamma$


# Sources

- [[kreisbeckLongLivedElectronic2012|Kreisbeck et al. (2012) Long-Lived Electronic Coherence in Dissipative Exciton Dynamics of Light-Harvesting Complexes]]
- [[kreisbeckScalableHighPerformanceAlgorithm2014|Kreisbeck et al. (2014) ScalableHigh-Performance Algorithm for the Simulationof Exciton Dynamics. Application to the Light-Harvesting Complex IIin the Presence of Resonant Vibrational Modes]]