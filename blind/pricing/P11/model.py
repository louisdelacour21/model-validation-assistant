import numpy as np
from scipy.integrate import quad

R, Q, S_REF = 0.02, 0.01, 100.0


def _phi(u, x, T, v0, kappa, theta, sigma, rho):
    b = kappa - 1j * rho * sigma * u
    d = np.sqrt(b * b + sigma ** 2 * (1j * u + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    D = (b - d) / sigma ** 2 * (1 - e) / (1 - g * e)
    C = kappa * theta / sigma ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    return np.exp(1j * u * (x + (R - Q) * T) + C + D * v0)


class Model:
    name = "blind-P11"
    domain = {"v0": (0.01, 0.16), "kappa": (0.5, 5.0), "theta": (0.01, 0.16),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.0), "k": (-0.3, 0.3),
              "days": (30, 730)}

    def price(self, p, spot=None) -> float:
        S = S_REF if spot is None else float(spot)
        T = p["days"] / 365.0
        K = S_REF * np.exp((R - Q) * T) * np.exp(p["k"])
        x, lk = np.log(S), np.log(K)
        a = (p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
        phim = _phi(-1j, x, T, *a).real  # = S*exp((r-q)T)

        def f1(u):
            return (np.exp(-1j * u * lk) * _phi(u - 1j, x, T, *a) / (1j * u * phim)).real

        def f2(u):
            return (np.exp(-1j * u * lk) * _phi(u, x, T, *a) / (1j * u)).real

        P1 = 0.5 + quad(f1, 1e-10, 200, limit=500)[0] / np.pi
        P2 = 0.5 + quad(f2, 1e-10, 200, limit=500)[0] / np.pi
        return float(S * np.exp(-Q * T) * P1 - K * np.exp(-R * T) * P2)
