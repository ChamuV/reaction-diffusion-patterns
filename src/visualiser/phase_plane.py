# src/visualiser/phase_plane.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_phase_plane(
    f,
    g,
    u0,
    v0,
    u_range=(0.01, 5),
    v_range=(0.01, 5),
    title="Phase plane",
    density=1.2,
):
    """
    Plots nullclines + streamlines + fixed point.

    Parameters:
    - f, g: callable (u, v) -> du/dt, dv/dt
    - u0, v0: fixed point
    """

    # Grid
    u_vals = np.linspace(*u_range, 400)
    v_vals = np.linspace(*v_range, 400)
    U, V = np.meshgrid(u_vals, v_vals)

    # Vector field
    dU = f(U, V)
    dV = g(U, V)

    plt.figure(figsize=(8, 6))

    # Nullclines
    plt.contour(U, V, f(U, V), levels=[0], colors="red", linewidths=2)
    plt.contour(U, V, g(U, V), levels=[0], colors="blue", linewidths=2)

    # Streamlines
    plt.streamplot(U, V, dU, dV, color="gray", density=density, arrowsize=1.4)

    # Fixed point
    plt.plot(u0, v0, "ko", markersize=8)
    plt.annotate(
        fr"$({u0:.2f}, {v0:.2f})$",
        xy=(u0, v0),
        xytext=(u0 + 0.2, v0 - 0.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    )

    # Legend
    custom_lines = [
        Line2D([0], [0], color="red", lw=2, label=r"$f(u,v)=0$"),
        Line2D([0], [0], color="blue", lw=2, label=r"$g(u,v)=0$"),
    ]
    plt.legend(handles=custom_lines, loc="upper right")

    # Formatting
    plt.xlabel(r"$u$")
    plt.ylabel(r"$v$")
    plt.title(title)

    plt.xlim(0, max(u0 + 0.5, u_range[1]))
    plt.ylim(0, max(v0 + 0.5, v_range[1]))

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()