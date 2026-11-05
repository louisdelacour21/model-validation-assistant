import numpy as np
from scipy.integrate import quad

R, Q, S_REF = 0.02, 0.01, 100.0


class Model:
    name = "blind-P10"
    domain = {"v0": (0.01, 0.16), "kappa": (0.5, 5.0), "theta": (0.01, 0.16),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.0), "k": (-0.3, 0.3), "days": (30, 730)}

    @staticmethod
    def _cf(z, T, v0, kappa, theta, sigma, rho):
        # E[exp(i z x_T)], x_T = ln(S_T/F), "little trap" branch
        b = kappa - 1j * rho * sigma * z
        d = np.sqrt(b * b + sigma ** 2 * (z * z + 1j * z))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        D = (b - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        C = kappa * theta / sigma ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        return np.exp(C + D * v0)

    def price(self, p, spot=None):
        S = S_REF if spot is None else float(spot)
        T = p["days"] / 365.0
        K = S_REF * np.exp((R - Q) * T) * np.exp(p["k"])
        F = S * np.exp((R - Q) * T)
        kk = np.log(K / F)
        args = (T, p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])

        def f(u):
            return (np.exp(-1j * u * kk) * self._cf(u - 0.5j, *args)).real / (u * u + 0.25)

        pts = [1, 5, 20, 60]
        I = sum(quad(f, a, b, limit=400)[0] for a, b in zip([0] + pts, pts))
        I += quad(f, pts[-1], np.inf, limit=400)[0]
        return float(np.exp(-R * T) * F * (1 - np.exp(kk / 2) / np.pi * I))
