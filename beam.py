import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# fastar
L = 2.0 # m
DEN = 480.0 # kg/m^3
w = 0.3 # m
d = 0.03 # m
g = 9.81 # m/s^2
E = 1.3e10
I = (w * math.pow(d, 3)) / 12


def f_kraftur():
    return -DEN * w * d * g


def fylki(n):
    A = diags(
        [1., -4., 6., -4., 1.],
        [-2, -1, 0, 1, 2],
        shape=(n, n),
        format="lil"
    )

    A[0, 0] = 16
    A[0, 1] = -9
    A[0, 2] = 8.0 / 3
    A[0, 3] = -1.0 / 4
    A[n-2, n-4] = 16.0 / 17
    A[n-2, n-3] = -60.0 / 17
    A[n-2, n-2] = 72.0 / 17
    A[n-2, n-1] = -28.0 / 17
    A[n-1, n-4] = -12.0 / 17
    A[n-1, n-3] = 96.0 / 17
    A[n-1, n-2] = -156.0 / 17
    A[n-1, n-1] = 72.0 / 17

    return A.tocsr()


def fylki_fast_fast(n):
    m = n - 1
    A = diags(
        [1., -4., 6., -4., 1.],
        [-2, -1, 0, 1, 2],
        shape=(m, m),
        format="lil"
    )

    A[0, 0] = 16
    A[0, 1] = -9
    A[0, 2] = 8.0 / 3
    A[0, 3] = -1.0 / 4
    A[m-1, m-4] = -1.0 / 4
    A[m-1, m-3] = 8.0 / 3
    A[m-1, m-2] = -9
    A[m-1, m-1] = 16

    return A.tocsr()


def nakvaem_lausn(x, load):
    return (load / (24 * E * I)) * (x**2) * (x**2 - 4 * L * x + 6 * L**2)


def leysa_fylki(n):
    h = L / n
    A = fylki(n)
    f = f_kraftur()
    b = np.full(n, (h**4 / (E * I)) * f)

    y = spsolve(A, b)
    y = np.concatenate(([0], y))
    x = np.linspace(0, L, n + 1)

    xf = np.linspace(0, L, 100)
    yf = nakvaem_lausn(xf, f)
    err = abs(yf[-1] - y[-1])
    # Industry-standard shorthand for grids/solutions: x,y for grid solution; xf,yf for fine grid exact curve.
    return x, y, xf, yf, err


def leysa_fylki_breytilegt(n, f_func, boundary="laus"):
    h = L / n
    if boundary == "laus":
        A = fylki(n)
        x = np.linspace(0, L, n + 1)
        f_vals = f_func(x[1:])
        b = (h**4 / (E * I)) * f_vals
        y = spsolve(A, b)
        y = np.concatenate(([0], y))
        return x, y
    if boundary == "fast":
        A = fylki_fast_fast(n)
        x = np.linspace(0, L, n + 1)
        f_vals = f_func(x[1:-1])
        b = (h**4 / (E * I)) * f_vals
        y_inner = spsolve(A, b)
        y = np.concatenate(([0], y_inner, [0]))
        return x, y
    raise ValueError("Unknown boundary type")


def teikna_lausn(x, y, xf, yf, save_path=None, show=True):
    plt.figure()
    plt.plot(xf, yf,
             color='black',
             linewidth=2,
             label='Nákvæm lausn')

    plt.plot(x, y,
             '--',
             color='blue',
             alpha=0.4,
             linewidth=1.2,
             label='Nálguð töluleg lausn')
    plt.plot(x, y,
             'o',
             color='blue',
             markersize=4)

    plt.xlabel("x (m)")
    plt.ylabel("y(x) (m)")
    #plt.title("Svignun timburbrettis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
