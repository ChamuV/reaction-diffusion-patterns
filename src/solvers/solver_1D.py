# src/solver_1D.py

import numpy as np
from scipy.linalg import lu_factor, lu_solve


class AB2AM2Solver1D:
    def __init__(self, model, params, Lx, N, T, dt, save_every=1):
        self.model = model
        self.params = params

        self.a = params["a"]
        self.b = params["b"]
        self.gamma = params["gamma"]
        self.d = params["d"]

        self.Lx = Lx
        self.N = N
        self.dx = Lx / (N - 1)
        self.x = np.linspace(0, Lx, N)

        self.T = T
        self.dt = dt
        self.Nt = int(T / dt)

        self.save_every = save_every

        self.U_hist = []
        self.V_hist = []
        self.t_hist = []

    # Nonlinear reaction terms
    def f(self, u, v):
        return self.model.f(u, v, self.params)

    def g(self, u, v):
        return self.model.g(u, v, self.params)

    # Initial condition
    def initial_condition(self, pert=0.01, p_type="cos"):
        u_ss, v_ss = self.model.steady_state(self.params)

        if p_type.lower() == "cos":
            perturb = pert * np.cos(self.x)
        elif p_type.lower() == "random":
            perturb = pert * (2 * np.random.rand(self.N) - 1)
        else:
            raise ValueError(f"Unknown perturbation type: {p_type}")

        U0 = u_ss * np.ones(self.N) + perturb
        V0 = v_ss * np.ones(self.N) + perturb
        return U0.reshape(-1, 1), V0.reshape(-1, 1)

    # 1D Laplacian
    def laplacian(self, bc="neumann", spatial_step=None):
        dx = self.dx if spatial_step is None else spatial_step

        L = -2 * np.eye(self.N) + np.eye(self.N, k=1) + np.eye(self.N, k=-1)

        if bc.lower() == "neumann":
            L[0, :2] = [-2, 2]
            L[-1, -2:] = [2, -2]
        else:
            raise ValueError(f"Unsupported boundary condition: {bc}")

        return L / dx**2

    # Block matrix: I - (dt/2)L
    def block_matrix_m(self, bc="neumann", spatial_step=None, time_step=None, diffusion=None):
        dx = self.dx if spatial_step is None else spatial_step
        dt = self.dt if time_step is None else time_step
        d = self.d if diffusion is None else diffusion

        A = self.laplacian(bc=bc, spatial_step=dx)

        m1 = np.eye(self.N) - 0.5 * dt * A
        m2 = np.eye(self.N) - 0.5 * d * dt * A

        return np.block([
            [m1, np.zeros((self.N, self.N))],
            [np.zeros((self.N, self.N)), m2]
        ])

    # Block matrix: I + (dt/2)L
    def block_matrix_p(self, bc="neumann", spatial_step=None, time_step=None, diffusion=None):
        dx = self.dx if spatial_step is None else spatial_step
        dt = self.dt if time_step is None else time_step
        d = self.d if diffusion is None else diffusion

        A = self.laplacian(bc=bc, spatial_step=dx)

        p1 = np.eye(self.N) + 0.5 * dt * A
        p2 = np.eye(self.N) + 0.5 * d * dt * A

        return np.block([
            [p1, np.zeros((self.N, self.N))],
            [np.zeros((self.N, self.N)), p2]
        ])

    # Nonlinear stacked vector
    def n_vec(self, Ui, Vi, time_step=None, gm=None):
        dt = self.dt if time_step is None else time_step
        gamma = self.gamma if gm is None else gm

        Fi = self.f(Ui, Vi).reshape(-1, 1)
        Gi = self.g(Ui, Vi).reshape(-1, 1)

        return gamma * dt * np.vstack([Fi, Gi])

    # Full stacked state vector
    def uv_vec(self, Ui, Vi):
        return np.vstack([Ui.reshape(-1, 1), Vi.reshape(-1, 1)])

    def save_snapshot(self, t, U, V):
        self.U_hist.append(U.flatten().copy())
        self.V_hist.append(V.flatten().copy())
        self.t_hist.append(t)

    def euler_step(self, Ui, Vi, dt=None, bc="neumann"):
        dt = self.dt if dt is None else dt
        A = self.laplacian(bc=bc)

        Fi = self.f(Ui, Vi)
        Gi = self.g(Ui, Vi)

        U_next = Ui + dt * (A @ Ui + self.gamma * Fi)
        V_next = Vi + dt * (self.d * (A @ Vi) + self.gamma * Gi)
        return U_next, V_next

    def ab2am2_step(self, Ui, Vi, Ui1, Vi1, Lp, Lm_lu_and_piv):
        Ni = self.n_vec(Ui, Vi)
        Ni1 = self.n_vec(Ui1, Vi1)

        Ui1_vec = self.uv_vec(Ui1, Vi1)
        LHS = (Lp @ Ui1_vec) + 1.5 * Ni1 - 0.5 * Ni

        full_U2 = lu_solve(Lm_lu_and_piv, LHS)

        Ui2 = full_U2[:self.N].reshape(-1, 1)
        Vi2 = full_U2[self.N:].reshape(-1, 1)

        return Ui2, Vi2

    def run(self, euler_step="n", pert=0.01, p_type="cos", bc="neumann"):
        self.U_hist = []
        self.V_hist = []
        self.t_hist = []

        # Build block matrices once
        Lp = self.block_matrix_p(bc=bc)
        Lm = self.block_matrix_m(bc=bc)

        # LU factorisation once
        Lm_lu_and_piv = lu_factor(Lm)

        # Initial conditions
        Ui, Vi = self.initial_condition(pert=pert, p_type=p_type)

        # Startup step
        if euler_step.lower() == "y":
            Ui1, Vi1 = self.euler_step(Ui, Vi, self.dt, bc=bc)
        else:
            Ui1, Vi1 = Ui.copy(), Vi.copy()

        # Save initial state
        self.save_snapshot(0.0, Ui, Vi)
        if self.save_every == 1:
            self.save_snapshot(self.dt, Ui1, Vi1)

        for i in range(self.Nt - 1):
            Ui2, Vi2 = self.ab2am2_step(Ui, Vi, Ui1, Vi1, Lp, Lm_lu_and_piv)

            if (i + 2) % self.save_every == 0:
                self.save_snapshot((i + 2) * self.dt, Ui2, Vi2)

            # Shift for next step
            Ui, Vi = Ui1.copy(), Vi1.copy()
            Ui1, Vi1 = Ui2.copy(), Vi2.copy()

        return (
            np.array(self.t_hist),
            np.array(self.U_hist),
            np.array(self.V_hist),
            self.x,
        )