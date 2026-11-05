import numpy as np
from scipy.integrate import quad

R, Q, S_REF = 0.02, 0.01, 100.0


class Model:
    name = "blind-P08"
    domain = {"v0": (0.01, 0.25), "kappa": (0.5, 5.0), "theta": (0.01, 0.25),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.0), "k": (-0.3, 0.3),
              "days": (30, 730)}

    def price(self, p, spot=None):
        v0, kappa, theta, sigma, rho = (p[x] for x in ("v0", "kappa", "theta", "sigma", "rho"))
        T = p["days"] / 365.0
        K = S_REF * np.exp((R - Q) * T) * np.exp(p["k"])
        S = S_REF if spot is None else spot
        x = np.log(S)
        lnK = np.log(K)

        def cf(u, j):
            uj = 0.5 if j == 1 else -0.5
            b = kappa - (rho * sigma if j == 1 else 0.0)
            a = kappa * theta
            iu = 1j * u
            rs = rho * sigma * iu
            d = np.sqrt((rs - b) ** 2 - sigma ** 2 * (2 * uj * iu - u * u))
            g = (b - rs + d) / (b - rs - d)
            e = np.exp(d * T)
            C = (R - Q) * iu * T + a / sigma ** 2 * ((b - rs + d) * T - 2 * np.log((1 - g * e) / (1 - g)))
            D = (b - rs + d) / sigma ** 2 * (1 - e) / (1 - g * e)
            return np.exp(C + D * v0 + iu * x)

        def prob(j):
            f = lambda u: (np.exp(-1j * u * lnK) * cf(u, j) / (1j * u)).real
            return 0.5 + quad(f, 1e-8, 200, limit=400)[0] / np.pi

        return float(S * np.exp(-Q * T) * prob(1) - K * np.exp(-R * T) * prob(2))
