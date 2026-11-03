---
tags:
  - topicNote
  - stub
aliases:
  - spectral density discretisation
  - discretisation
date: 2026-10-22
---
# Definition

A rule for approximating a [[Spectral density|spectral density]] as a sum over discrete frequencies
![[Spectral density#^d56018]]
This is mathematically equivalent for choosing a [[Quadrature rule|quadrature rule]] for evaluating integrals of the form
$$
\int_{0}^{\infty} \frac{J(\omega)}{\omega} \,f(\omega)\ \mathrm{d}\omega \approx
\frac{\pi}{2}\sum_{j = 1}^{n_{\text{bath}}} \frac{c_j^2}{m_j \omega_j^2} f(\omega_j) .
$$
## Walters discretisation

This approach is described by Walters, Allen, and Makri[^Walters2017]. A closely related variation of this is known as "logarithmic" discretisation[^Makri1999]. It is defined by two parameters: a maximum frequency $\omega_{\text{max}}$ and the number of bath modes $n_{\text{bath}}$. To determine the frequencies $\omega_j$, first compute
$$
\varepsilon = \frac{1}{n_{\text{bath}}} \int_0^{\omega_{\text{max}}} \frac{J(\omega)}{\omega}\,\mathrm{d}\omega
$$
and then solve
$$
\frac{1}{\varepsilon} \int_{0}^{\omega_j} \frac{J(\omega)}{\omega}\,\mathrm{d}\omega = j - \frac{1}{2}, \qquad j = 1, 2, \ldots, n_{\text{bath}}
$$
for $\omega_j$. This determines the quadrature nodes. The quadrature weights $w_j = c_j^2 / m_j \omega_j^2$ are **all set equal**:
$$
\frac{c_j^2}{m_j \omega_j^2} = \kappa^2 \quad \Leftrightarrow \quad
c_j = \kappa \sqrt{m_j} \omega_j
$$
and fixed by requiring that the normalisation ($f(\omega) = 1$ in the quadrature rule equation) be recovered exactly:
$$
\varepsilon = \frac{1}{n_{\text{bath}}} \times \frac{\pi}{2} \sum_{j = 1}^{n_{\text{bath}}} \kappa^2 \quad \Rightarrow \quad
\kappa = \sqrt{\frac{2 \varepsilon}{\pi}}.
$$
The masses can be chosen freely or, equivalently, the bath degrees of freedom can be propagated in mass-weighted coordinates.
# Sources

[^Walters2017]: [[waltersDirectDetermination2017|Walters et al. (2017) Direct determination of discrete harmonic bath parameters from molecular dynamics simulations]]
[^Makri1999]: [[makriLinearResponse1999|Makri (1999) The Linear Response Approximation and Its Lowest Order Corrections]]
