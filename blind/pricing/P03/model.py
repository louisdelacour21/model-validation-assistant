import numpy as np
from scipy.integrate import quad


class Model:
    name = "blind-P03"
    domain = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10),
              "sigma": (0.1, 0.8), "rho": (-0.9, -0.1), "k": (-0.3, 0.3),
              "days": (30, 730)}
    r = 0.02
    q = 0.01

    def _phi(self, z, T, v0, kappa, theta, sigma, rho):
        b = kappa - rho * sigma * 1j * z
        d = np.sqrt(b * b + sigma ** 2 * (1j * z + z * z))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = kappa * theta / sigma ** 2 * ((b - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + v0 * D)

    def price(self, p, spot=None):
        S = 100.0 if spot is None else float(spot)
        T = p["days"] / 365.0
        F = S * np.exp((self.r - self.q) * T)
        K = F * np.exp(p["k"])
        df = np.exp(-self.q * T)
        a = (p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
        kl = -p["k"]

        def f(u):
            return (np.exp(1j * u * kl) * self._phi(u - 0.5j, T, *a)).real / (u * u + 0.25)

        pts = [1, 5, 20, 80]
        tot = 0.0
        lo = 0.0
        for hi in pts + [400]:
            tot += quad(f, lo, hi, limit=200, epsabs=1e-12, epsrel=1e-12)[0]
            lo = hi
        return float(df * (F - np.sqrt(F * K) * tot / np.pi))
