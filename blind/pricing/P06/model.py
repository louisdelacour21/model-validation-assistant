import numpy as np
from scipy.integrate import quad


class Model:
    name = "blind-P06"
    domain = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10),
              "sigma": (0.1, 0.8), "rho": (-0.9, -0.1), "k": (-0.3, 0.3),
              "days": (30, 730)}
    R = 0.02
    Q = 0.01
    UMAX = 40.0

    def _cf(self, u, p, T):
        v0, ka, th, sg, rho = p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"]
        b = ka - 1j * rho * sg * u
        d = np.sqrt(b * b + sg * sg * (u * u + 1j * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = ka * th / sg**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sg**2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0)

    def price(self, p, spot=None):
        S = 100.0 if spot is None else float(spot)
        T = p["days"] / 365.0
        F = S * np.exp((self.R - self.Q) * T)
        k = p["k"]
        K = F * np.exp(k)

        def prob(shift):
            def f(u):
                return (np.exp(-1j * u * k) * self._cf(u - shift, p, T) / (1j * u)).real
            val, _ = quad(f, 1e-10, self.UMAX, limit=200)
            return 0.5 + val / np.pi

        return float(np.exp(-self.R * T) * (F * prob(1j) - K * prob(0.0)))
