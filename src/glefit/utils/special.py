#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   special.py
@Time    :   2025/10/15 13:16:17
@Author  :   George Trenins
@Desc    :   Special functions
'''

from __future__ import print_function, division, absolute_import
import numpy as np


def coscosh(G, t, nu=0, wrt='time'):
    """Return cos(F*t) if G < 0 and cosh(G*t) otherwise

    Args:
        G, t: float or array-like
            function arguments
        nu : int
            derivative order
        wrt : variable with respect to which to differentiate, ["time", "Gamma"]

    Return:
       Value of piecewise function at t if nu == 0 and its nu'th derivative w.r.t. t if nu > 0
    """

    if not isinstance(nu, int):
        raise TypeError(f"The derivative order `nu` must be integer, instead got {nu.__class__.__name__}")
    if nu < 0:
        raise ValueError(f"The derivative order must be non-negative, instead got {nu = }.")
    if wrt not in {"time", "Gamma"}:
        raise ValueError(f"The argument 'wrt' must be one of 'time' or 'Gamma'")
    mask = G*np.ones_like(t) < 0
    if nu%2 == 0:
        k = nu//2
        f, g = np.cos, np.cosh
    else:
        k = (nu+1)//2
        f, g = np.sin, np.sinh
    sgn = np.where(mask, (-1)**k, 1)
    Gt = G*t
    ans = np.where(mask, sgn*f(Gt), g(Gt))
    if wrt == "time":
        return (G**nu) * ans
    else:
        return (t**nu) * ans
    
def expcoscosh(G, t, lamda, nu=0, wrt='time'):
    """Evaluate exp(-λ·t) * coscosh(G, t) in a numerically stable way.
    
    Args:
        G : float or array-like
            Parameter for coscosh function
        t : float or array-like
            Time argument
        lamda : float
            Exponential decay rate (must be positive)
        nu : int, optional
            Derivative order (0, 1, or 2). Defaults to 0.
        wrt : str, optional
            Variable to differentiate w.r.t.: 'time', 'Gamma' or 'lamda'. Defaults to 'time'.

    Note:
        Where numerical stability is concerned, the implementation assumes that lambda > max(G, 0) and t > 0.
    
    Returns:
        float or array-like: Value of exp(-λ·t) * coscosh(G, t) or its derivatives
    """
    if not isinstance(nu, int):
        raise TypeError(f"The derivative order `nu` must be integer, instead got {nu.__class__.__name__}")
    if nu < 0:
        raise ValueError(f"The derivative order must be non-negative, instead got {nu = }.")
    if wrt not in {"time", "Gamma"}:
        raise ValueError(f"The argument 'wrt' must be one of 'time' or 'Gamma'")
    if np.any(lamda < 0):
        raise ValueError(f"lamda must be non-negative, instead got {lamda = }.")
    
    t = np.asarray(t)
    G = np.asarray(G) * np.ones_like(t)
    mask = G < 0
    Gt = G * t
    lamda = np.asarray(lamda)
    lamda = lamda * np.ones_like(t)
        
    if nu == 0:
        ans = np.empty_like(t)
        # oscillatory regime
        if np.any(mask):
            ans[mask] = np.exp(-lamda[mask] * t[mask])* np.cos(Gt[mask])
        # hyperbolic regime
        mask2 = ~mask
        if np.any(mask2):
            lambda_minus_G = lamda[mask2] - G[mask2]
            lambda_plus_G = lamda[mask2] + G[mask2]
            t_mask = t[mask2]
            ans[mask2] = (
                np.exp(-lambda_minus_G*t_mask) + 
                np.exp(-lambda_plus_G*t_mask)
            )/2
    elif nu == 1:
        ans = np.empty_like(t)
        # oscillatory regime
        if np.any(mask):
            exp_decay = np.exp(-lamda[mask]*t[mask])
            ans[mask] = coscosh(
                G[mask],
                t[mask],
                nu = 1,
                wrt=wrt
            ) * exp_decay
            if wrt == "time":
                ans[mask] += coscosh(
                    G[mask],
                    t[mask]
                ) * exp_decay * (-lamda[mask])
        # hyperbolic regime
        mask2 = ~mask
        if np.any(mask2):
            lambda_minus_G = lamda[mask2] - G[mask2]
            lambda_plus_G = lamda[mask2] + G[mask2]
            t_mask = t[mask2]
            if wrt == "time":
                ans[mask2] = -(
                    lambda_minus_G*np.exp(-lambda_minus_G*t_mask) + 
                    lambda_plus_G*np.exp(-lambda_plus_G*t_mask)
                )/2
            else:
                ans[mask2] = (
                    t_mask*np.exp(-lambda_minus_G*t_mask) - 
                    t_mask*np.exp(-lambda_plus_G*t_mask)
                )/2
    elif nu == 2:
        ans = np.empty_like(t)
        # oscillatory regime
        if np.any(mask):
            exp_decay = np.exp(-lamda[mask]*t[mask])
            ans[mask] = coscosh(
                G[mask],
                t[mask],
                nu = 2,
                wrt=wrt
            ) * exp_decay
            if wrt == "time":
                ans[mask] += coscosh(
                    G[mask],
                    t[mask],
                    nu = 1,
                    wrt=wrt
                ) * exp_decay * (-2*lamda[mask])
                ans[mask] += coscosh(
                    G[mask],
                    t[mask]
                ) * exp_decay * (lamda[mask]**2)
        # hyperbolic regime
        mask2 = ~mask
        if np.any(mask2):
            lambda_minus_G = lamda[mask2] - G[mask2]
            lambda_plus_G = lamda[mask2] + G[mask2]
            t_mask = t[mask2]
            if wrt == "time":
                ans[mask2] = (
                    lambda_minus_G**2 * np.exp(-lambda_minus_G*t_mask) + 
                    lambda_plus_G**2 * np.exp(-lambda_plus_G*t_mask)
                )/2
            else:  
                ans[mask2] = (
                    t_mask**2*np.exp(-lambda_minus_G*t_mask) + 
                    t_mask**2*np.exp(-lambda_plus_G*t_mask)
                )/2
    else:
        raise ValueError(f"Derivative order must be 0, 1, or 2, instead got {nu = }.")
    
    return ans

def sincsinhc(G, t, nu=0, wrt='time'):
    """Return sinc(G,t) if G < 0 and sinhc(G*t) otherwise

    Args:
        G,t : float or array-like
            function arguments
        nu : int
             derivative order (0, 1, or 2)
        wrt : variable with respect to which to differentiate, ["time", "Gamma"]

    Return:
       Value of piecewise function at t if nu == 0 and its nu'th derivative w.r.t. t if nu > 0.
    """
    if not isinstance(nu, int):
        raise TypeError(f"The derivative order `nu` must be integer, instead got {nu.__class__.__name__}")    
    if wrt not in {"time", "Gamma"}:
        raise ValueError(f"The argument 'wrt' must be one of 'time' or 'Gamma'")
    mask = G*np.ones_like(t) < 0
    eps = 1.0e-2
    sgnG = np.where(mask, -1, 1)
    Gt = G*np.abs(t)
    cond0 = Gt > eps
    cond1 = Gt > -eps
    small = np.logical_and(np.logical_not(cond0), cond1)
    den = np.where(small, 1.0, Gt)
    if nu == 0:
        ans = np.where( 
            Gt > eps, 
            np.sinh(Gt)/den, 
            np.where(Gt > -eps, 
                     1 + sgnG*Gt**2/6 + Gt**4/120 + sgnG*Gt**6/5040,
                     np.sinc(Gt/np.pi)) )
    elif nu == 1:
        ans = np.where( 
            Gt > eps, 
            np.exp(Gt)/(2*den) * (1-1/den) + np.exp(-Gt)/(2*den) * (1+1/den),
            np.where(Gt > -eps, 
                     sgnG*Gt/3 + Gt**3/30 + sgnG*Gt**5/840,
                     (np.cos(Gt) - np.sinc(Gt/np.pi))/den))
        if wrt == "time":
            ans *= G*np.where(t < 0, -1, 1)
        else:
            ans *= np.abs(t)
    elif nu == 2:
        ans = np.where( 
            Gt > eps, 
            np.exp(Gt)/den * (0.5 - 1/den + 1/den**2) - np.exp(-Gt)/den * (0.5 + 1/den + 1/den**2),
            np.where(Gt > -eps, 
                     sgnG/3 + Gt**2/10 + sgnG*Gt**4/168,
                     np.sinc(Gt/np.pi) * (2/Gt**2 - 1) - np.cos(Gt) * (2/den**2)) )
        if wrt == "time":
            ans *= G**2
        else:
            ans *= t**2
    else:
        raise ValueError(f"The derivative order must be 0, 1, or 2, instead got {nu = }.")
    return ans

def expsincsinhc(G, t, lamda, nu=0, wrt='time'):
    """Evaluate exp(-λ·t) * sincsinhc(G, t) in a numerically stable way.
    

    Args:
        G : float or array-like
            Parameter for sincsinhc function
        t : float or array-like
            Time argument
        lamda : float
            Exponential decay rate 
        nu : int, optional
            Derivative order (0, 1, or 2). Defaults to 0.
        wrt : str, optional
            Variable to differentiate w.r.t.: 'time' or 'Gamma'. Defaults to 'time'.

    Note:
        Where numerical stability is concerned, the implementation assumes that lambda > max(G, 0) and t > 0.
    
    Returns:
        float or array-like: Value of exp(-λ·t) * sincsinhc(G, t) or its derivatives
    """
    if not isinstance(nu, int):
        raise TypeError(f"The derivative order `nu` must be integer, instead got {nu.__class__.__name__}")
    if nu < 0:
        raise ValueError(f"The derivative order must be non-negative, instead got {nu = }.")
    if wrt not in {"time", "Gamma"}:
        raise ValueError(f"The argument 'wrt' must be one of 'time' or 'Gamma'")
    if np.any(lamda < 0):
        raise ValueError(f"lamda must be non-negative, instead got {lamda = }.")
    
    t = np.asarray(t)
    G = np.asarray(G) * np.ones_like(t)
    mask = G < 0  # oscillatory regime
    lamda = np.asarray(lamda) * np.ones_like(t)
    if nu == 0:
        ans = np.empty_like(t)
        # Oscillatory regime (G < 0): exp(-λt) * sinc(Gt)
        if np.any(mask):
            ans[mask] = np.exp(-lamda[mask] * t[mask]) * sincsinhc(G[mask], t[mask])
        # Hyperbolic regime (G >= 0): exp(-λt) * sinhc(Gt)
        mask2 = ~mask
        if np.any(mask2):
            mask3 = np.abs(G*t) < 0.01
            small = np.logical_and(mask2, mask3)
            if np.any(small):
                ans[small] = np.exp(-lamda[small] * t[small]) * sincsinhc(
                    G[small], t[small]
                )
            big = np.logical_and(mask2, ~mask3)
            if np.any(big):
                lambda_minus_G = lamda[big] - G[big]
                lambda_plus_G = lamda[big] + G[big]
                ans[big] = (
                    np.exp(-lambda_minus_G*t[big]) - 
                    np.exp(-lambda_plus_G*t[big])
                )/(2*G[big]*t[big])
    elif nu == 1:
        ans = np.empty_like(t)
        # Oscillatory regime
        if np.any(mask):
            exp_decay = np.exp(-lamda[mask] * t[mask])
            ans[mask] = sincsinhc(G[mask], t[mask], nu=1, wrt=wrt) * exp_decay
            if wrt == "time":
                ans[mask] += sincsinhc(G[mask], t[mask], nu=0) * exp_decay * (-lamda[mask])
        # Hyperbolic regime
        mask2 = ~mask
        if np.any(mask2):
            mask3 = np.abs(G*t) < 0.01
            small = np.logical_and(mask2, mask3)
            if np.any(small):
                exp_decay = np.exp(-lamda[small] * t[small])
                ans[small] = sincsinhc(G[small], t[small], nu=1, wrt=wrt) * exp_decay
                if wrt == "time":
                    ans[small] += sincsinhc(G[small], t[small], nu=0) * exp_decay * (-lamda[small])
            big = np.logical_and(mask2, ~mask3)
            if np.any(big):
                t_big = t[big]
                G_big = G[big]
                lambda_minus_G = lamda[big] - G_big
                exp_minus = np.exp(-lambda_minus_G*t_big)
                lambda_plus_G = lamda[big] + G_big
                exp_plus = np.exp(-lambda_plus_G*t_big)
                if wrt == "time":
                    ans[big] = (
                       -lambda_minus_G * exp_minus +
                        lambda_plus_G * exp_plus
                    ) / (2*G_big*t_big)
                    ans[big] -= (
                        exp_minus - exp_plus
                    )/(2*G_big*t_big**2)
                else:  # wrt == "Gamma"
                    ans[big] = (
                        t_big * exp_minus +
                        t_big * exp_plus
                    ) / (2*G_big*t_big)
                    ans[big] -= (
                        exp_minus - exp_plus
                    )/(2*G_big**2*t_big)
    elif nu == 2:
        ans = np.empty_like(t)
        # Oscillatory regime
        if np.any(mask):
            exp_decay = np.exp(-lamda[mask] * t[mask])
            ans[mask] = sincsinhc(G[mask], t[mask], nu=2, wrt=wrt) * exp_decay
            if wrt == "time":
                ans[mask] += sincsinhc(G[mask], t[mask], nu=1, wrt='time') * exp_decay * (-2*lamda[mask])
                ans[mask] += sincsinhc(G[mask], t[mask], nu=0) * exp_decay * (lamda[mask]**2)
        # Hyperbolic regime
        mask2 = ~mask
        if np.any(mask2):
            mask3 = np.abs(G*t) < 0.01
            small = np.logical_and(mask2, mask3)
            if np.any(small):
                exp_decay = np.exp(-lamda[small] * t[small])
                ans[small] = sincsinhc(G[small], t[small], nu=2, wrt=wrt) * exp_decay
                if wrt == "time":
                    ans[small] += sincsinhc(G[small], t[small], nu=1, wrt='time') * exp_decay * (-2*lamda[small])
                    ans[small] += sincsinhc(G[small], t[small], nu=0) * exp_decay * (lamda[small]**2)
            big = np.logical_and(mask2, ~mask3)
            if np.any(big):
                t_big = t[big]
                G_big = G[big]
                lambda_minus_G = lamda[big] - G_big
                exp_minus = np.exp(-lambda_minus_G*t_big)
                lambda_plus_G = lamda[big] + G_big
                exp_plus = np.exp(-lambda_plus_G*t_big)
                if wrt == "time":
                    # d²/dt² of (exp(-(λ-G)t) - exp(-(λ+G)t))/(2Gt)
                    ans[big] = (
                        lambda_minus_G**2 * exp_minus -
                        lambda_plus_G**2 * exp_plus
                    ) / (2*G_big*t_big)
                    ans[big] -= (
                        - lambda_minus_G * exp_minus 
                        + lambda_plus_G * exp_plus
                    ) / (G_big*t_big**2)
                    ans[big] += (
                        exp_minus - exp_plus
                    )/(G_big*t_big**3)
                else:  # wrt == "Gamma"
                    ans[big] = (
                        - t_big**2 * exp_minus
                        + t_big**2 * exp_plus
                    ) / (2*G_big*t_big)
                    ans[big] += (
                        - t_big * exp_minus
                        + t_big * exp_plus
                    ) / (G_big**2*t_big)
                    ans[big] += (
                        exp_minus - exp_plus
                    ) / (G_big**3*t_big)
    else:
        raise ValueError(f"Derivative order must be 0, 1, or 2, instead got {nu = }.")
    
    return ans


class Softplus(object):

    def __init__(self, sigma=1.0, threshold=20.0):
        """Softplus function
        
            Softplus(x) = log(1 + exp(sigma*x))/sigma

        For numerical stability, Softplus(x) = x for sigma*x > threshold and 0 for sigma*x < -threshold

        Args:
            sigma (float, optional): transition steepness. Defaults to 1.0.
            threshold (float, optional): cut-off for linear approximation. Defaults to 20.0.

        """
        if not isinstance(sigma, (int,float)):
            raise TypeError(f"sigma must be a float, got {type(sigma).__name__}")
        self.sigma = float(sigma)
        if not isinstance(threshold, (int,float)):
            raise TypeError(f"threshold must be a float, got {type(threshold).__name__}")
        self.threshold = float(threshold)
        if not self.sigma > 0:
            raise ValueError(f"sigma must be positive, instead got {sigma = }")
        if not self.threshold > 0:
            raise ValueError(f"threshold must be positive, instead got {threshold = }")
        
    def __call__(self, x, nu=0):
        """Evaluate Softplus(x) = log(1 + exp(sigma*x))/sigma or its nu'th derivative w.r.t. x

        Args:
            x (array-like): function argument
            nu (int, optional): Derivative order, 0, 1, or 2. Defaults to 0.

        Returns:
            array-like: value of the function
        """

        x = np.asarray(x)
        s = self.sigma * x
        ans = np.zeros_like(x, dtype=float)
        low = s < -self.threshold
        high = s > self.threshold
        mid = ~(low | high)
        sm = s[mid]
        if nu == 0:
            ans[high] = x[high]
            if np.any(mid):
                ans[mid] = np.log1p(np.exp(sm)) / self.sigma
        elif nu == 1:
            ans[high] = 1.0
            if np.any(mid):
                ans[mid] = 1 / (1 + np.exp(-sm))
        elif nu == 2:
            if np.any(mid):
                ans[mid] = self.sigma / (2 * np.cosh(sm/2))**2
        else:
            raise ValueError(f"Expected nu = 0, 1, or 2 instead got {nu = }")
        return ans


class Softmax(Softplus):

    def __init__(self, sigma=1, threshold=20):
        """Two-argument Softmax function

            Softmax(x1, x2) = log( exp(sigma*x1) + exp(sigma*x2) ) / sigma
                            = x1 + Softplus(x2-x1)

        For numerical stability, Softplus(x) = x for sigma*x > threshold and 0 for sigma*x < -threshold

        Args:
            sigma (float, optional): transition steepness. Defaults to 1.0.
            threshold (float, optional): cut-off for linear approximation. Defaults to 20.0.
        """
        super().__init__(sigma=sigma, threshold=threshold)

    def __call__(self, x1, x2, nu=0):
        """Evaluate 
                Softmax(x1, x2) = log( exp(sigma*x1) + exp(sigma*x2) ) / sigma
                                = x1 + Softplus(x2-x1)

        Args:
            x1, x2 (array-like): function arguments, float or 1d array of shape (M,)
            nu (int, optional): Derivative order, 0, 1, or 2. Defaults to 0.

        Returns:
            ans (array-like): value of the function or its nu't derivative
                * nu==0: float or 1d array of shape (M,)
                * nu==1: 2d array of shape (2,M) where ans[0,i] = D[Softmax(x1[i], x2[i]), x1[i]] and ans[1,i] = D[Softmax(x1[i], x2[i]), x2[i]]
                * nu==2: 3d array of shape (2,2,M) where ans[0,0,i] = D[Softmax(x1[i], x2[i]), x1[i], x1[i]], etc.
        """
        x1 = np.atleast_1d(x1)
        if x1.ndim != 1:
            raise ValueError(f"Expecting a 1d input for x1, instead got ndim = {x1.ndim}")
        x2 = np.atleast_1d(x2)
        if x2.ndim != 1:
            raise ValueError(f"Expecting a 1d input for x2, instead got ndim = {x2.ndim}")
        if (xshape := x1.shape) != x2.shape:
            raise ValueError(f"Expecting arguments to have the same shape, instead got x1.shape = {xshape} and x2.shape = {x2.shape}")
        diff = x2 - x1
        if nu == 0:
            ans = x1 + super().__call__(diff, nu=0)
            if xshape == (1,):
                return ans.item()
            else:
                return ans
        elif nu == 1:
            ans = np.empty((2,)+xshape)
            ans[1] = super().__call__(diff, nu=1)
            ans[0] = 1.0 - ans[1]
            return ans
        elif nu == 2:
            ans = np.empty((2,2)+xshape)
            ans[0,0] = ans[1,1] = super().__call__(diff, nu=2)
            ans[0,1] = ans[1,0] = -ans[0,0]
            return ans
        else:
            raise ValueError(f"Expected nu = 0, 1, or 2 instead got {nu = }")



class Softabs(Softplus):

    def __init__(self, sigma=1, threshold=20):
        """Softabs function

            Softabs(x) = log( exp(-sigma*x) + exp(sigma*x) ) / sigma
                            = -x + Softplus(2*x)

        For numerical stability, Softplus(x) = x for sigma*x > threshold and 0 for sigma*x < -threshold

        Args:
            sigma (float, optional): transition steepness. Defaults to 1.0.
            threshold (float, optional): cut-off for linear approximation. Defaults to 20.0.
        """
        super().__init__(sigma=sigma, threshold=threshold)

    def __call__(self, x, nu=0):
        """Evaluate Softabs(x) = -x + Softplus(2*x) or its nu'th derivative w.r.t. x

        Args:
            x (array-like): function argument
            nu (int, optional): Derivative order, 0, 1, or 2. Defaults to 0.

        Returns:
            array-like: value of the function
        """
        x = np.asarray(x)
        y = 2*x
        if nu == 0:
            return -x + super().__call__(y, nu=0)
        elif nu == 1:
            return 2*super().__call__(y, nu=1) - 1
        elif nu == 2:
            return 4*super().__call__(y, nu=2)
        else:
            raise ValueError(f"Expected nu = 0, 1, or 2 instead got {nu = }")

