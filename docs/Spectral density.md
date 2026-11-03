---
tags:
  - topicNote
aliases:
  - spectral density
date: 2026-09-15
---
# Definition

For a system coordinate $q$ (mass $m$), the spectral density characterises the linear response of the system to the dissipative environment. For position-independent dissipation, the spectral density can be [[Spectral density discretisation|discretised]], which yields the parameters of a harmonic bath bilinearly coupled to the system that reproduces dissipative forces:
$$
J(\omega) = \frac{\pi}{2}\sum_j \frac{c_j^2}{m_j\omega_j}\,\delta(\omega-\omega_j).
$$

^d56018

This is related to the friction kernel via
$$
K(t) = \frac{2}{\pi}\int_0^\infty \frac{J(\omega)}{\omega}\cos(\omega t)\,\text{d}\omega
\quad
\Leftrightarrow
\quad
J(\omega) = \omega \int_{0}^{\infty} K(t) \cos(\omega t) \, \text{d}t
$$
and the **static** friction is
$$
\eta_0 = \int_0^\infty K(t)\,\text{d}t = \lim_{\omega\to 0}\frac{J(\omega)}{\omega}.
$$
The kernel enters the generalised Langevin equation as
$$
\frac{\text{d}p}{\text{d}t} = -\frac{\partial V(q)}{\partial q} - \int^t K(t-s)\,\frac{p(s)}{m}\,\text{d}s + \zeta(t)
$$
where the random force satisfies the second fluctuation-dissipation theorem
$$
\langle \zeta(t) \zeta(t')\rangle = k_{\text{B}} T K(t-t')
$$
## Examples

- [[Debye spectral density]]
- [[Shifted Drude-Lorentz spectral density]]
- [[Brownian oscillator spectral density]]

# Sources

- [[Pollak1989|Pollak et al. (1989) Theory of activated rate processes for arbitrary frequency dependent friction: solution of the turnover problem]]
- [[pradaDissipativeMatsubara2023|Prada et al. (2023) Dissipative Matsubara Dynamics]]
- [[thossSelfconsistentHybrid2001|Thoss et al. (2001) Self-consistent hybrid approach for complex systems]]
- [[treninsNonMarkovianEffects2025|Trenins et al. (2025) Non-Markovian Effects in Quantum Rate Calculations of Hydrogen Diffusion with Electronic Friction]]
