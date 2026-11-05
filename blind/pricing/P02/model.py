import numpy as np
from scipy.integrate import quad


class Model:
    name = "blind-P02"
    domain = {"v0": (0.01, 0.10), "kappa": (0.5, 4.0), "theta": (0.01, 0.10),
              "sigma": (0.1, 0.8), "rho": (-0.9, -0.1), "k": (-0.3, 0.3),
              "days": (30, 730)}
    r = 0.02
    q = 0.01
    S = 100.0

    def price(self, p, spot=None):
        S = self.S if spot is None else float(spot)
        v0, ka, th, sg, rho, k = (float(p[x]) for x in ("v0", "kappa", "theta", "sigma", "rho", "k"))
        T = float(p["days"]) / 365.0
        F = S * np.exp((self.r - self.q) * T)
        K = F * np.exp(k)

        def phi(z):
            b = ka - rho * sg * 1j * z
            d = np.sqrt(b * b + sg * sg * (z * z + 1j * z))
            g = (b - d) / (b + d)
            e = np.exp(-d * T)
            D = (b - d) / sg**2 * (1 - e) / (1 - g * e)
            C = ka * th / sg**2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
            return np.exp(C + D * v0)

        def f(u):
            return (np.exp(-1j * u * k) * phi(u - 0.5j)).real / (u * u + 0.25)

        w = np.sqrt(v0 * T + 1e-12)
        pts = [0.0, 1.0 / w, 5.0 / w, 20.0 / w]
        tot = sum(quad(f, a, b, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
                  for a, b in zip(pts[:-1], pts[1:]))
        tot += quad(f, pts[-1], np.inf, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
        return float(np.exp(-self.r * T) * (F - np.sqrt(F * K) * tot / np.pi))
