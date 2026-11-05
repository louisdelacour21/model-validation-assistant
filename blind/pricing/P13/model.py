import numpy as np
from scipy.integrate import quad

R, Q, S_REF = 0.02, 0.01, 100.0


class Model:
    name = "blind-P13"
    domain = {"v0": (0.005, 0.2), "kappa": (0.5, 5.0), "theta": (0.01, 0.2),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.3), "k": (-0.3, 0.3),
              "days": (30, 730)}

    @staticmethod
    def _psi(u, x, T, v0, kappa, theta, sigma, rho):
        # characteristic function of ln S_T (Albrecher "little trap" form)
        b = kappa - rho * sigma * 1j * u
        d = np.sqrt(b * b + sigma ** 2 * (1j * u + u * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = (R - Q) * 1j * u * T + kappa * theta / sigma ** 2 * (
            (b - d) * T - 2.0 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / sigma ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0 + 1j * u * x)

    def price(self, p, spot=None):
        S = S_REF if spot is None else float(spot)
        T = p["days"] / 365.0
        K = S_REF * np.exp((R - Q) * T) * np.exp(p["k"])
        x = np.log(S)
        a = (p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
        psi = lambda u: self._psi(u, x, T, *a)
        fwd = psi(-1j)  # = E[S_T]
        lk = np.log(K)

        def f1(u):
            return (np.exp(-1j * u * lk) * psi(u - 1j) / (1j * u * fwd)).real

        def f2(u):
            return (np.exp(-1j * u * lk) * psi(u) / (1j * u)).real

        opts = dict(limit=500, epsabs=1e-11, epsrel=1e-11)
        # integrand decays exponentially; 400 is far beyond the support
        P1 = 0.5 + quad(f1, 1e-10, 400, **opts)[0] / np.pi
        P2 = 0.5 + quad(f2, 1e-10, 400, **opts)[0] / np.pi
        return float(S * np.exp(-Q * T) * P1 - K * np.exp(-R * T) * P2)
