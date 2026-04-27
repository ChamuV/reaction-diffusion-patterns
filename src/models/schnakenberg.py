# src/models/schnakenberg.py

from numba import njit


class SchnakenbergModel:
    @staticmethod
    def f(u, v, params):
        return params["a"] - u + u**2 * v

    @staticmethod
    def g(u, v, params):
        return params["b"] - u**2 * v

    @staticmethod
    def steady_state(params):
        a = params["a"]
        b = params["b"]
        return a + b, b / (a + b)**2

    @staticmethod
    def build_numba_reactions(params, dt):
        a = params["a"]
        b = params["b"]
        gamma = params["gamma"]

        @njit(fastmath=True, parallel=True)
        def f_numba(U, V):
            return gamma * dt * (a - U + U**2 * V)

        @njit(fastmath=True, parallel=True)
        def g_numba(U, V):
            return gamma * dt * (b - U**2 * V)

        return f_numba, g_numba