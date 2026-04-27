# src/analysis_tools/steady_states.py

import numpy as np
from scipy.optimize import root


# Substrate-inhibitor model
def substrate_h(u, v, rho, K):
    return (rho * u * v) / (1 + u + K * u**2)


def substrate_f(u, v, a, rho, K):
    return a - u - substrate_h(u, v, rho, K)


def substrate_g(u, v, b, alpha, rho, K):
    return alpha * (b - v) - substrate_h(u, v, rho, K)


def substrate_steady_state(a, b, alpha, rho, K, guess=None):
    if guess is None:
        guess = (a, b)

    def system(z):
        u, v = z
        return [
            substrate_f(u, v, a, rho, K),
            substrate_g(u, v, b, alpha, rho, K),
        ]

    sol = root(system, guess, method="hybr")

    if not sol.success:
        return np.nan, np.nan

    u0, v0 = sol.x

    if u0 <= 0 or v0 <= 0:
        return np.nan, np.nan

    return u0, v0


# Gierer-Meinhardt model
def gierer_meinhardt_steady_state(a, b):
    u0 = (a + 1) / b
    v0 = u0**2
    return u0, v0