import numpy as np
from scipy.integrate import quad

R, Q, S0 = 0.02, 0.01, 100.0


class Model:
    name = "blind-P09"
    domain = {"v0": (0.01, 0.16), "kappa": (0.5, 5.0), "theta": (0.01, 0.16),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.3), "k": (-0.3, 0.3),
              "days": (30, 730)}

    @staticmethod
    def _phi(z, T, v0, kappa, theta, sigma, rho):
        """E[exp(i z ln(S_T/F))] under the risk-neutral measure (stable branch)."""
        b = kappa - 1j * rho * sigma * z
        d = np.sqrt(b * b + sigma**2 * (z * z + 1j * z))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = kappa * theta / sigma**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sigma**2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0 - 1j * z * 0)

    def price(self, p, spot=None):
        v0, kappa, theta, sigma, rho = (p[x] for x in ("v0", "kappa", "theta", "sigma", "rho"))
        T = p["days"] / 365.0
        K = S0 * np.exp((R - Q) * T) * np.exp(p["k"])
        S = S0 if spot is None else spot
        F = S * np.exp((R - Q) * T)
        k = np.log(K / F)
        ph = lambda z: self._phi(z, T, v0, kappa, theta, sigma, rho)
        phm = ph(-1j).real
        f2 = lambda u: (np.exp(-1j * u * k) * ph(u) / (1j * u)).real
        f1 = lambda u: (np.exp(-1j * u * k) * ph(u - 1j) / (1j * u * phm)).real
        up = 500.0
        P2 = 0.5 + quad(f2, 1e-10, up, limit=500)[0] / np.pi
        P1 = 0.5 + quad(f1, 1e-10, up, limit=500)[0] / np.pi
        df = np.exp(R * T)
        return float(df * (F * P1 - K * P2) / S0)
