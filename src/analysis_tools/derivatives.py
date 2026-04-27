# src/analysis_tools/derivatives.py

import numpy as np


# Schnakenberg model
def schnakenberg_derivatives(a, b):
    """
    Returns partial derivatives (fu, fv, gu, gv)
    evaluated at the homogeneous steady state.
    """
    u0 = a + b
    v0 = b / (a + b)**2

    fu = -1 + 2 * u0 * v0
    fv = u0**2
    gu = -2 * u0 * v0
    gv = -u0**2

    return fu, fv, gu, gv


# Substrate–inhibitor model
def substrate_inhibitor_derivatives(a, b, alpha, rho, K, u0, v0):
    """
    Returns partial derivatives (fu, fv, gu, gv)
    evaluated at (u0, v0).
    
    NOTE:
    (u0, v0) must be provided (no closed form steady state).
    """

    denom = (1 + u0 + K * u0**2)
    h_u = (rho * v0 * (1 + u0 + K * u0**2) - rho * u0 * v0 * (1 + 2 * K * u0)) / denom**2
    h_v = (rho * u0) / denom

    fu = -1 - h_u
    fv = -h_v

    gu = -h_u
    gv = -alpha - h_v

    return fu, fv, gu, gv


# Gierer–Meinhardt model
def gierer_meinhardt_derivatives(a, b, u0, v0, eps=1e-8):
    """
    Returns partial derivatives (fu, fv, gu, gv)
    evaluated at (u0, v0).
    
    eps is used to avoid division issues.
    """

    v_safe = v0 + eps

    fu = -b + (2 * u0) / v_safe
    fv = -(u0**2) / (v_safe**2)

    gu = 2 * u0
    gv = -1

    return fu, fv, gu, gv