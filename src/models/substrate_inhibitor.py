# src/models/substrate_inhibitor.py

import numpy as np
from numba import njit
from scipy.optimize import root


class SubstrateInhibitorModel:
    @staticmethod
    def h(u, v, params):
        rho = params["rho"]
        K = params["K"]
        return (rho * u * v) / (1 + u + K * u**2)

    @staticmethod
    def f(u, v, params):
        a = params["a"]
        return a - u - SubstrateInhibitorModel.h(u, v, params)

    @staticmethod
    def g(u, v, params):
        b = params["b"]
        alpha = params["alpha"]
        return alpha * (b - v) - SubstrateInhibitorModel.h(u, v, params)

    @staticmethod
    def build_numba_reactions(params, dt):

        a = params["a"]

        b = params["b"]

        alpha = params["alpha"]

        rho = params["rho"]

        K = params["K"]

        gamma = params["gamma"]

        @njit(fastmath=True, parallel=True)

        def h_numba(U, V):

            return (rho * U * V) / (1 + U + K * U**2)

        @njit(fastmath=True, parallel=True)

        def f_numba(U, V):

            return gamma * dt * (a - U - h_numba(U, V))

        @njit(fastmath=True, parallel=True)

        def g_numba(U, V):

            return gamma * dt * (alpha * (b - V) - h_numba(U, V))

        return f_numba, g_numba
    
    @staticmethod
    def steady_state(params):
        a = params["a"]
        b = params["b"]
        alpha = params["alpha"]
        rho = params["rho"]
        K = params["K"]

        def system(z):
            u, v = z
            h = (rho * u * v) / (1 + u + K * u**2)
            return [
                a - u - h,
                alpha * (b - v) - h,
            ]

        sol = root(system, [a, b], method="hybr")

        if not sol.success:
            raise RuntimeError(f"Could not find substrate-inhibitor steady state: {sol.message}")

        u0, v0 = sol.x

        if u0 <= 0 or v0 <= 0:
            raise RuntimeError(f"Invalid steady state found: u0={u0}, v0={v0}")

        return u0, v0