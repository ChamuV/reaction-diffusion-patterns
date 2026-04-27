# src/visualiser/turing_relation.py

import numpy as np
import matplotlib.pyplot as plt


def plot_turing_relation(
    k2_vals,
    fu,
    fv,
    gu,
    gv,
    gamma,
    d_vals,
    labels=None,
    colors=None,
    title=r"Relationship between $h(k^2)$ and $\lambda(k^2)$",
):
    """
    Plot h(k^2) and the dominant eigenvalue lambda(k^2)
    for several diffusion ratios.
    """

    det = fu * gv - fv * gu

    if labels is None:
        labels = [fr"$d={d:.3g}$" for d in d_vals]

    if colors is None:
        colors = ["teal", "orange", "steelblue", "purple", "red", "black"]

    def h_k2_vals(k2, d):
        return d * k2**2 - gamma * (d * fu + gv) * k2 + gamma**2 * det

    def lambda_k2_vals(k2, d):
        B = k2 * (1 + d) - gamma * (fu + gv)
        C = h_k2_vals(k2, d)
        discriminant = B**2 - 4 * C
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))

        lambda1 = (-B + sqrt_disc) / 2
        lambda2 = (-B - sqrt_disc) / 2

        return np.maximum(lambda1, lambda2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    fig.suptitle(title, fontsize=22, y=0.94)

    handles = []

    for d, label, color in zip(d_vals, labels, colors):
        h_vals = h_k2_vals(k2_vals, d)
        line, = axes[0].plot(k2_vals, h_vals, label=label, color=color, lw=2.5)
        handles.append(line)

        mask = h_vals < 0
        if np.any(mask):
            k2_band = k2_vals[mask]
            axes[0].plot([k2_band[0], k2_band[-1]], [0, 0], "ro", markersize=7)

    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set_xlabel(r"$k^2$", fontsize=16)
    axes[0].set_ylabel(r"$h(k^2)$", fontsize=16)
    axes[0].set_title(r"Roots of $h(k^2)$", fontsize=18)
    axes[0].tick_params(labelsize=13)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    for d, label, color in zip(d_vals, labels, colors):
        lambda_vals = lambda_k2_vals(k2_vals, d)
        axes[1].plot(k2_vals, lambda_vals, color=color, lw=2.5)

        mask = lambda_vals > 0
        if np.any(mask):
            k2_band = k2_vals[mask]
            axes[1].plot([k2_band[0], k2_band[-1]], [0, 0], "ro", markersize=7)

    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set_xlabel(r"$k^2$", fontsize=16)
    axes[1].set_ylabel(r"$\lambda(k^2)$", fontsize=16)
    axes[1].set_title(r"Dominant eigenvalue $\lambda(k^2)$", fontsize=18)
    axes[1].tick_params(labelsize=13)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        fontsize=14,
        ncol=len(d_vals),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    plt.show()