# src/visualiser/plot_2D.py

import matplotlib.pyplot as plt

def plot_2D_both(X, Y, U, V):
    plt.figure(figsize=(12, 5))

    # U plot
    plt.subplot(1, 2, 1)
    c1 = plt.contourf(X, Y, U, cmap='viridis', levels=50)
    plt.colorbar(c1, label='U concentration')
    plt.title("U concentration")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    # V plot
    plt.subplot(1, 2, 2)
    c2 = plt.contourf(X, Y, V, cmap='plasma', levels=50)
    plt.colorbar(c2, label='V concentration')
    plt.title("V concentration")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.tight_layout()
    plt.show()


def plot_2D_U(X, Y, U):
    plt.figure(figsize=(6, 5))
    c = plt.contourf(X, Y, U, cmap='viridis', levels=50)
    plt.colorbar(c, label='U concentration')
    plt.title("U concentration")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.tight_layout()
    plt.show()

def plot_2D_cases(cases):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5 * len(cases), 4))

    for i, case in enumerate(cases, 1):
        X, Y, U, title = case

        plt.subplot(1, len(cases), i)

        c = plt.contourf(X, Y, U, cmap='viridis', levels=50)

        plt.title(title, fontsize=14)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")

        # small, clean colorbar per plot
        plt.colorbar(c, fraction=0.046, pad=0.04)

    plt.suptitle("Pattern morphology across domain geometries", fontsize=16)
    plt.tight_layout()
    plt.show()