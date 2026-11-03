---
tags:
  - topicNote
aliases: []
date: 2026-09-21
---
# Definition

The [[Spectral density|spectral density]] is[^garg1985][^lawrence2019]
$$
J_\mathrm{BO}(\omega) = \eta_0 \omega \frac{\omega_0^4}{(\omega^2-\omega_0^2)^2+\gamma^2\omega^2}
$$
and the [[Memory kernel|memory kernel]] is
$$
K_{\text{BO}}(t) = \frac{\eta_0\omega_0^2}{\gamma}\,e^{-\gamma t/2}\left[\cos(\bar\omega t) + \frac{\gamma}{2\bar\omega}\sin(\bar\omega t)\right],
\qquad
\bar\omega = \sqrt{\omega_0^2-\gamma^2/4}
$$
with the following parameters

- $\eta_0$: static friction
- $\omega_0$: undamped frequency of the oscillator mode
- $\gamma$: damping rate of the oscillator mode
- For $\omega_0 < \gamma/2$, $\bar\omega$ is imaginary and $\cos$, $\sin$ become $\cosh$, $\sinh$ of $\sqrt{\gamma^2/4-\omega_0^2}\,t$.

## Harmonic discretisation

According to the "logarithmic" prescription[^makri2017][^makri1999] the frequencies in a [[Spectral density discretisation|spectral density discretisation]] satisfy
$$
F(\omega_j) = \frac{j-1/2}{n_{\text{bath}}}
$$
where for underdamped $\gamma < 2 \omega_0$,
$$
F(\omega) = \frac{1}{2}
  + \frac{1}{\pi}\arctan\!\left(\frac{\omega^2 - \omega_0^2}{\gamma\,\omega}\right)
  + \frac{\gamma}{2\pi\bar{\omega}}\,\operatorname{arctanh}\!\left(\frac{2\bar{\omega}\,\omega}{\omega^2 + \omega_0^2}\right)
\qquad
\bar{\omega} = \sqrt{\omega_0^2 - \gamma^2/4}
$$
for overdamped $\gamma > 2 \omega_0$,
$$
F(\omega) = \frac{1}{2}
  + \frac{1}{\pi}\arctan\!\left(\frac{\omega^2 - \omega_0^2}{\gamma\,\omega}\right)
  + \frac{\gamma}{2\pi\tilde{\omega}}\,\arctan\!\left(\frac{2\tilde{\omega}\,\omega}{\omega^2 + \omega_0^2}\right)
\qquad
\tilde{\omega} = \sqrt{\gamma^2/4 - \omega_0^2}
$$
critically damped $\gamma = 2 \omega_0$
$$
F(\omega) = \frac{1}{2}
  + \frac{1}{\pi}\arctan\!\left(\frac{\omega^2 - \omega_0^2}{\gamma\,\omega}\right)
  + \frac{\gamma \omega}{\pi  (\omega^2 + \omega_0^2)}
$$
and the coupling coefficients are
$$
c_j = \omega_j \omega_0\sqrt{\frac{\eta_0 m_j}{\gamma\,n_{\text{bath}}}}
$$
# Sources
[^garg1985]: [[Garg1985a|Garg et al. (1985) Effect of friction on electron transfer in biomolecules]]

[^lawrence2019]: [[Lawrence2019a|Lawrence et al. (2019) On the calculation of quantum mechanical electron transfer rates]]

[^makri1999]: [[makriLinearResponse1999|Makri (1999) The Linear Response Approximation and Its Lowest Order Corrections]]

[^makri2017]: [[waltersDirectDetermination2017|Walters et al. (2017) Direct determination of discrete harmonic bath parameters from molecular dynamics simulations]]
