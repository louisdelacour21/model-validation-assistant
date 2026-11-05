import numpy as np
from scipy.integrate import quad

R, Q, S_REF = 0.02, 0.01, 100.0


class Model:
    name = "blind-P07"
    domain = {"v0": (0.01, 0.15), "kappa": (0.5, 5.0), "theta": (0.01, 0.15),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.0), "k": (-0.3, 0.3), "days": (30, 730)}

    def price(self, p, spot=None) -> float:
        S = S_REF if spot is None else float(spot)
        T = p["days"] / 365.0
        K = S_REF * np.exp((R - Q) * T) * np.exp(p["k"])
        v0 = max(p["v0"], 0.04)
        kap, th, sig, rho = p["kappa"], p["theta"], p["sigma"], p["rho"]
        lnS, lnK = np.log(S), np.log(K)

        def cf(u):  # char. function of ln S_T (Albrecher "little trap" form)
            d = np.sqrt((kap - 1j * rho * sig * u) ** 2 + sig ** 2 * (1j * u + u * u))
            g = (kap - 1j * rho * sig * u - d) / (kap - 1j * rho * sig * u + d)
            e = np.exp(-d * T)
            C = (kap * th / sig ** 2) * ((kap - 1j * rho * sig * u - d) * T
                                         - 2 * np.log((1 - g * e) / (1 - g)))
            D = (kap - 1j * rho * sig * u - d) / sig ** 2 * (1 - e) / (1 - g * e)
            return np.exp(1j * u * (lnS + (R - Q) * T) + C + D * v0)

        f1_den = cf(-1j).real

        def integ(u, j):
            f = cf(u - 1j) / f1_den if j == 1 else cf(u)
            return (np.exp(-1j * u * lnK) * f / (1j * u)).real

        P = []
        for j in (1, 2):
            val = quad(integ, 1e-10, 200, args=(j,), limit=500)[0]
            P.append(0.5 + val / np.pi)
        return float(S * np.exp(-Q * T) * P[0] - K * np.exp(-R * T) * P[1])
