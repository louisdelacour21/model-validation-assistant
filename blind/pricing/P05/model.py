import numpy as np
from scipy.integrate import quad


class Model:
    name = "blind-P05"
    domain = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10),
              "sigma": (0.1, 0.8), "rho": (-0.9, -0.1), "k": (-0.3, 0.3),
              "days": (30, 730)}
    r = 0.02
    q = 0.01
    UMAX = 60.0

    def _cf(self, u, v0, kappa, theta, sigma, rho, T):
        b = kappa - rho * sigma * 1j * u
        d = np.sqrt(b * b + sigma ** 2 * (u * u + 1j * u))
        e = np.exp(-d * T)
        g = (b - d) / (b + d)
        D = (b - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        C = kappa * theta / sigma ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        return np.exp(C + D * v0)

    def price(self, p, spot=None):
        S = 100.0 if spot is None else float(spot)
        v0, kappa, theta, sigma, rho, k = (float(p[x]) for x in ("v0", "kappa", "theta", "sigma", "rho", "k"))
        T = float(p["days"]) / 365.0
        F = S * np.exp((self.r - self.q) * T)
        K = F * np.exp(k)
        args = (v0, kappa, theta, sigma, rho, T)

        def f2(u):
            return (np.exp(-1j * u * k) * self._cf(u, *args) / (1j * u)).real

        def f1(u):
            return (np.exp(-1j * u * k) * self._cf(u - 1j, *args) / (1j * u)).real

        opts = dict(limit=400, epsabs=1e-10, epsrel=1e-10)
        P1 = 0.5 + quad(f1, 1e-10, self.UMAX, **opts)[0] / np.pi
        P2 = 0.5 + quad(f2, 1e-10, self.UMAX, **opts)[0] / np.pi
        return float(np.exp(-self.r * T) * (F * P1 - K * P2))
