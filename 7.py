import numpy as np
import matplotlib.pyplot as plt
from beam import leysa_fylki_breytilegt, f_kraftur, L, g, E, I

p = 100.0  # kg/m


def f_samtals(x):
    return f_kraftur() - p * g * np.sin(np.pi * x / L)


def nakvaem_lausn_cc_sin(x):
    f = f_kraftur()
    y_const = (f / (24 * E * I)) * x**2 * (x - L)**2
    y_sin = -(p * g / (E * I)) * (L**3 / np.pi**4) * np.sin(np.pi * x / L)
    y_poly = (p * g / (E * I)) * ((-L / np.pi**3) * x**2 + (L**2 / np.pi**3) * x)
    return y_const + y_sin + y_poly


n = 2560
x, y = leysa_fylki_breytilegt(n, f_samtals, boundary="fast")

xf = np.linspace(0, L, 400)
yf = nakvaem_lausn_cc_sin(xf)
mid_idx = len(x) // 2
err_mid = abs(nakvaem_lausn_cc_sin(L / 2) - y[mid_idx])

print(f"Sekkjan i x=L/2 er: {err_mid}.")

plt.figure()
plt.plot(xf, yf, color='black', linewidth=2, label='Nákvæm lausn')
plt.plot(x, y, '--', color='blue', alpha=0.4, linewidth=1.2, label='Nálguð töluleg lausn')
plt.plot(x, y, 'o', color='blue', markersize=4)
plt.xlabel("x (m)")
plt.ylabel("y(x) (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PNG/beam_p7.png", dpi=300)
