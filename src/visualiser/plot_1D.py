# src/visualiser/plot_1D.py

import matplotlib.pyplot as plt

def plot_1D_single(x, U, V):
    plt.figure(figsize=(12, 5))

    plt.plot(x, U, linewidth=2, label='U')
    plt.plot(x, V, linewidth=2, label='V')

    plt.title("Final U and V Profiles", fontsize=18)
    plt.xlabel("Position", fontsize=14)
    plt.ylabel("Concentration", fontsize=14)

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_1D_grid(results, L_vals, d_vals):
    fig, axs = plt.subplots(
        len(d_vals),
        len(L_vals),
        figsize=(5 * len(L_vals), 3.5 * len(d_vals)),
        sharey=True
    )

    for i, d in enumerate(d_vals):
        for j, L in enumerate(L_vals):

            if len(d_vals) == 1 and len(L_vals) == 1:
                ax = axs
            elif len(d_vals) == 1:
                ax = axs[j]
            elif len(L_vals) == 1:
                ax = axs[i]
            else:
                ax = axs[i, j]

            x, U, V = results[(d, L)]

            line_u, = ax.plot(x, U, linewidth=2)
            line_v, = ax.plot(x, V, linewidth=2)

            ax.set_title(f"L = {L}, d = {d}", fontsize=14)
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel("Concentration", fontsize=12)
            if i == len(d_vals) - 1:
                ax.set_xlabel("Position", fontsize=12)

            ax.tick_params(labelsize=10)

    fig.suptitle("Final U and V Profiles Across Parameters", fontsize=18)

    fig.legend(
        [line_u, line_v],
        ["U", "V"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()