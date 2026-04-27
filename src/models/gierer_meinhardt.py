# src/models/gierer_meinhardt.py

from numba import njit


class GiererMeinhardtModel:
    @staticmethod
    def f(u, v, params):
        a = params["a"]
        b = params["b"]
        eps = params.get("eps", 1e-8)
        return a - b * u + (u**2) / (v + eps)

    @staticmethod
    def g(u, v, params):
        return u**2 - v

    @staticmethod
    def steady_state(params):
        a = params["a"]
        b = params["b"]
        u0 = (a + 1) / b
        v0 = u0**2
        return u0, v0

    @staticmethod
    def build_numba_reactions(params, dt):
        a = params["a"]
        b = params["b"]
        gamma = params["gamma"]
        eps = params.get("eps", 1e-8)

        @njit(fastmath=True, parallel=True)
        def f_numba(U, V):
            return gamma * dt * (a - b * U + (U**2) / (V + eps))

        @njit(fastmath=True, parallel=True)
        def g_numba(U, V):
            return gamma * dt * (U**2 - V)

        return f_numba, g_numba