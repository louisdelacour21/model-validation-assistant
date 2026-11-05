import numpy as np
from scipy.integrate import quad


class Model:
    name = "blind-P01"
    domain = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10),
              "sigma": (0.1, 0.8), "rho": (-0.9, -0.1), "k": (-0.3, 0.3),
              "days": (30, 730)}
    R = 0.02
    Q = 0.01

    def price(self, p, spot=None):
        S = 100.0 if spot is None else float(spot)
        T = p["days"] / 365.0
        v0, ka, th, sg, rho, k = (p[x] for x in ("v0", "kappa", "theta", "sigma", "rho", "k"))
        F = S * np.exp((self.R - self.Q) * T)
        x = -k

        def phi(z):
            b = ka - 1j * rho * sg * z
            d = np.sqrt(b * b + sg * sg * (1j * z + z * z))
            g = (b - d) / (b + d)
            e = np.exp(-d * T)
            C = ka * th / sg**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
            D = (b - d) / sg**2 * (1 - e) / (1 - g * e)
            return np.exp(C + D * v0)

        def f(u):
            return (np.exp(1j * u * x) * phi(u - 0.5j)).real / (u * u + 0.25)

        pts = [0.0, 5.0, 20.0, 60.0]
        tot = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            tot += quad(f, a, b, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
        tot += quad(f, pts[-1], 400.0, epsabs=1e-12, epsrel=1e-12, limit=400)[0]
        K = F * np.exp(k)
        return float(np.exp(-self.R * T) * (F - np.sqrt(F * K) / np.pi * tot))
