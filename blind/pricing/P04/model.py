import numpy as np
from scipy.integrate import quad

R, Q, S0 = 0.02, 0.01, 100.0


class Model:
    name = "blind-P04"
    domain = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10),
              "sigma": (0.1, 0.8), "rho": (-0.9, -0.1), "k": (-0.3, 0.3),
              "days": (30, 730)}
    UPPER = 100.0

    def _cf(self, u, T, v0, kappa, theta, sigma, rho):
        # E[exp(i u ln(S_T/F))], little-trap formulation
        b = kappa - 1j * rho * sigma * u
        d = np.sqrt(b * b + sigma ** 2 * (u * u + 1j * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = kappa * theta / sigma ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0 + 1j * u * (-1j * 0))  # forward-measure, centred below

    def price(self, p, spot=None):
        S = S0 if spot is None else float(spot)
        T = p["days"] / 365.0
        F = S * np.exp((R - Q) * T)
        K = F * np.exp(p["k"])
        k = p["k"]
        a = (p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])

        def phi(u):
            # char. fn of x = ln(S_T/F); martingale correction handled by shift
            return self._cf(u, T, *a)

        def i2(u):
            return (np.exp(-1j * u * k) * phi(u) / (1j * u)).real

        def i1(u):
            return (np.exp(-1j * u * k) * phi(u - 1j) / (1j * u * phi(-1j))).real

        P1 = 0.5 + quad(i1, 1e-10, self.UPPER, limit=200)[0] / np.pi
        P2 = 0.5 + quad(i2, 1e-10, self.UPPER, limit=200)[0] / np.pi
        return float(S * np.exp(-Q * T) * P1 - K * np.exp(-R * T) * P2)
