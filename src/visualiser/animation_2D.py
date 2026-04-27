# src/visualiser/animation_2D.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_2D_U(
    X, Y, U_hist,
    filename="reaction_diffusion_U.gif",
    fps=8,
    levels=50,
    title="Reaction–diffusion pattern formation"
):
    U_hist = np.asarray(U_hist)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    contour_levels = np.linspace(np.min(U_hist), np.max(U_hist), levels)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(title, fontsize=14)

    c0 = ax.contourf(X, Y, U_hist[0], cmap="viridis", levels=contour_levels)
    fig.colorbar(c0, ax=ax, label="U concentration")

    def draw_frame(frame):
        ax.clear()
        ax.contourf(X, Y, U_hist[frame], cmap="viridis", levels=contour_levels)

        ax.set_title("U concentration")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        return []

    anim = FuncAnimation(fig, draw_frame, frames=len(U_hist), blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)


def animate_2D_both(
    X, Y, U_hist, V_hist,
    filename="reaction_diffusion_both.gif",
    fps=8,
    levels=50,
    title="Schnakenberg reaction–diffusion dynamics"
):
    U_hist = np.asarray(U_hist)
    V_hist = np.asarray(V_hist)

    if len(U_hist) != len(V_hist):
        raise ValueError("U_hist and V_hist must have the same number of frames.")

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    u_levels = np.linspace(np.min(U_hist), np.max(U_hist), levels)
    v_levels = np.linspace(np.min(V_hist), np.max(V_hist), levels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=15)

    c1 = ax1.contourf(X, Y, U_hist[0], cmap="viridis", levels=u_levels)
    c2 = ax2.contourf(X, Y, V_hist[0], cmap="plasma", levels=v_levels)

    fig.colorbar(c1, ax=ax1, label="U concentration")
    fig.colorbar(c2, ax=ax2, label="V concentration")

    def draw_frame(frame):
        ax1.clear()
        ax2.clear()

        ax1.contourf(X, Y, U_hist[frame], cmap="viridis", levels=u_levels)
        ax2.contourf(X, Y, V_hist[frame], cmap="plasma", levels=v_levels)

        ax1.set_title("U concentration")
        ax1.set_xlabel(r"$x$")
        ax1.set_ylabel(r"$y$")

        ax2.set_title("V concentration")
        ax2.set_xlabel(r"$x$")
        ax2.set_ylabel(r"$y$")

        return []

    anim = FuncAnimation(fig, draw_frame, frames=len(U_hist), blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)