# src/solvers/solver_2D.py

# src/solvers/solver_2D.py

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import factorized
from numba import njit


class AB2AM2Solver2D:
    def __init__(self, model, params, Lx, Ly, Nx, Ny, T, dt, save_every=1):
        self.model = model
        self.params = params

        self.a = params["a"]
        self.b = params["b"]
        self.gamma = params["gamma"]
        self.d = params["d"]

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.T = T
        self.dt = dt
        self.Nt = int(T / dt)

        self.save_every = save_every

        self.U_hist = []
        self.V_hist = []
        self.t_hist = []

        self._build_numba_reaction_functions()

    def _build_numba_reaction_functions(self):
        if not hasattr(self.model, "build_numba_reactions"):
            raise ValueError(
                f"{self.model.__name__} must define build_numba_reactions(params, dt)"
            )
    
        self.f_numba, self.g_numba = self.model.build_numba_reactions(
            self.params,
            self.dt
        )

    def initialize_conditions(self, pert=0.01, p_type="random"):
        u_ss, v_ss = self.model.steady_state(self.params)

        if p_type.lower() == "random":
            U0 = u_ss * np.ones((self.Ny, self.Nx)) + pert * np.random.randn(self.Ny, self.Nx)
            V0 = v_ss * np.ones((self.Ny, self.Nx)) + pert * np.random.randn(self.Ny, self.Nx)

        elif p_type.lower() == "cos":
            perturb = pert * np.cos(self.X) * np.cos(self.Y)
            U0 = u_ss * np.ones((self.Ny, self.Nx)) + perturb
            V0 = v_ss * np.ones((self.Ny, self.Nx)) + perturb

        else:
            raise ValueError(f"Unknown perturbation type: {p_type}")

        return U0, V0

    def laplacian_construct(self, N, delta, bc="neumann"):
        Ldiag = -2 * np.ones(N)
        Loffdiag = np.ones(N - 1)
        L = diags([Loffdiag, Ldiag, Loffdiag], [-1, 0, 1], shape=(N, N)) / delta**2
        L = L.tolil()

        if bc.lower() == "neumann":
            L[0, 1] = 2 / delta**2
            L[-1, -2] = 2 / delta**2
        else:
            raise ValueError(f"Unsupported boundary condition: {bc}")

        return L.tocsc()

    def laplacian_matrix_2d(self, bc="neumann"):
        LX = self.laplacian_construct(self.Nx, self.dx, bc=bc)
        LY = self.laplacian_construct(self.Ny, self.dy, bc=bc)
        return kron(identity(self.Ny, format="csc"), LX) + kron(LY, identity(self.Nx, format="csc"))

    def matrix_creator_factoriser(self, bc="neumann"):
        Lap = self.laplacian_matrix_2d(bc=bc)
        I = identity(self.Nx * self.Ny, format="csc")

        L2dU = 0.5 * self.dt * Lap
        L2dV = self.d * L2dU

        return factorized((I - L2dU).tocsc()), factorized((I - L2dV).tocsc())

    def matrix_creator_multiplier(self, bc="neumann"):
        Lap = self.laplacian_matrix_2d(bc=bc)
        I = identity(self.Nx * self.Ny, format="csc")

        L2dU = 0.5 * self.dt * Lap
        L2dV = self.d * L2dU

        return (I + L2dU).tocsc(), (I + L2dV).tocsc()

    def save_snapshot(self, t, U, V):
        self.U_hist.append(U.copy())
        self.V_hist.append(V.copy())
        self.t_hist.append(t)

    def ab2am2_step(self, U_prev, V_prev, U_prev1, V_prev1, P1, P2, M1, M2):
        Ui = U_prev.ravel()
        Vi = V_prev.ravel()
        Ui1 = U_prev1.ravel()
        Vi1 = V_prev1.ravel()

        Ueq = P1 @ Ui1 + 1.5 * self.f_numba(Ui1, Vi1) - 0.5 * self.f_numba(Ui, Vi)
        Veq = P2 @ Vi1 + 1.5 * self.g_numba(Ui1, Vi1) - 0.5 * self.g_numba(Ui, Vi)

        Ui2 = M1(Ueq)
        Vi2 = M2(Veq)

        return Ui2.reshape((self.Ny, self.Nx)), Vi2.reshape((self.Ny, self.Nx))

    def run(self, pert=0.01, p_type="random", bc="neumann"):
        self.U_hist = []
        self.V_hist = []
        self.t_hist = []

        M1, M2 = self.matrix_creator_factoriser(bc=bc)
        P1, P2 = self.matrix_creator_multiplier(bc=bc)

        U_prev, V_prev = self.initialize_conditions(pert=pert, p_type=p_type)
        U_prev1, V_prev1 = self.initialize_conditions(pert=pert, p_type=p_type)

        self.save_snapshot(0.0, U_prev1, V_prev1)

        for i in range(self.Nt):
            Ui2, Vi2 = self.ab2am2_step(U_prev, V_prev, U_prev1, V_prev1, P1, P2, M1, M2)

            if (i + 1) % self.save_every == 0:
                self.save_snapshot((i + 1) * self.dt, Ui2, Vi2)

            U_prev, V_prev = U_prev1.copy(), V_prev1.copy()
            U_prev1, V_prev1 = Ui2.copy(), Vi2.copy()

        return (
            np.array(self.t_hist),
            np.array(self.U_hist),
            np.array(self.V_hist),
            self.x,
            self.y,
            self.X,
            self.Y,
        )

    def save_final_u_plot(self, U_final, label, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        c = ax.contourf(self.X, self.Y, U_final, cmap="viridis", levels=50)

        cb = fig.colorbar(c, ax=ax, label="U Concentration", shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=14)
        cb.set_label("U Concentration", fontsize=16)

        ax.set_title(label, fontsize=25)
        ax.set_xlabel("x", fontsize=20)
        ax.set_ylabel("y", fontsize=20)
        ax.tick_params(labelsize=18)

        fig.tight_layout()

        filepath = os.path.join(save_dir, f"{label.replace(' ', '_')}_U.png")
        fig.savefig(filepath, dpi=300)
        plt.close(fig)

        return filepath
