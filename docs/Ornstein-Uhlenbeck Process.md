---
tags:
  - topicNote
aliases:
  - Ornstein-Uhlenbeck process
date: 2026-09-23
---
# Definition

A [[Gaussian process]] defined by the [[Stochastic differential equation|SDE]]
$$
\mathrm{d}x = -k x \,\mathrm{d}t + \sqrt{D} \, \mathrm{d}W(t)
$$
where $t$ is time and $\mathrm{d}W(t)$ is the differential of a [[Wiener process]].

## Properties

Denoting $x(0) \equiv x_0$, the mean is
$$
\mathbb{E}[x(t)] = \langle x_0 \rangle \, \mathrm{e}^{-k t}
$$
the variance is
$$
\mathbb{E}\Big[\big\{x(t) - \langle x(t) \rangle \big\}^2 \Big] = \frac{D}{2 k} + \left[
\mathrm{Var}\{ x_0 \} - \frac{D}{2k}
\right] \, \mathrm{e}^{-2 k t}
$$
and the time-correlation function is
$$
\begin{aligned}
\langle x(t),  x(s)\rangle & \equiv 
\mathbb{E}\Big[
    \big\{x(t) - \langle x(t) \rangle \big\}
    \big\{x(s) - \langle x(s) \rangle \big\}
\Big] \\
& = \left[
\mathrm{Var}\{x_0\} - \frac{D}{2 k}
\right] \, \mathrm{e}^{-k (t+s)} + \frac{D}{2k} \, \mathrm{e}^{-k \lvert t-s \rvert}
\end{aligned}
$$
# Sources

- [[gardinerHandbookStochastic2002|Gardiner (2002) Handbook of stochastic methods]]