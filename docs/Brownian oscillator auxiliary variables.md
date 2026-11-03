---
tags:
  - topicNote
  - stub
aliases: []
date: 2026-10-06
---
# Definition

A type of [[Auxiliary variables (SDE)|auxiliary variable]] based on a two-variable [[Auxiliary variables (SDE) - Kantorovich prescription|Kantorovich mapping]] for numerical propagation of the [[Generalised Langevin equation|GLE]] that approximates the non-Markovian dynamics with
$$
\begin{pmatrix}
\dot{p}               \\
\dot{s}_1             \\
\dot{s}_2           
\end{pmatrix} = 
-\begin{pmatrix}
0    &  a_p  & 0 \\
-a_p &  0 & \omega_0 \\
0    & -\omega_0 & \gamma
\end{pmatrix}
\begin{pmatrix}
    p               \\
    s_1             \\
    s_2 
\end{pmatrix} 
+
\begin{pmatrix}
0    & 0 \\
0 &    \boldsymbol{B}
\end{pmatrix}
\begin{pmatrix}
0               \\
\boldsymbol{\xi}
\end{pmatrix}.
$$
The corresponding [[Memory kernel|memory kernel]] is
$$
K(t) = a_p^2\, e^{-\gamma \lvert t \rvert/2}\left[\cos(\bar\omega t) + \frac{\gamma}{2\bar\omega}\sin(\bar\omega \lvert t \rvert)\right],
\qquad
\bar\omega = \sqrt{\omega_0^2-\gamma^2/4}
$$
and the [[Spectral density|spectral density]] is
$$
J(\omega) = a_p^2\gamma\,\omega\,\frac{\omega_0^2}{(\omega^2-\omega_0^2)^2+\gamma^2\omega^2}
$$
which corresponds exactly to the [[Brownian oscillator spectral density|BO spectral density]] with
- static friction $\eta_0 = a_p^2\gamma/\omega_0^2$
- oscillator frequency $\omega_0$ and damping rate $\gamma$ identical to the corresponding entries of the drift matrix
# Sources

- [[brunigPairReactionDynamics2022|Brünig et al. (2022) Pair-Reaction Dynamics in Water]]
