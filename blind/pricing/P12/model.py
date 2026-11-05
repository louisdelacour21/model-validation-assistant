import numpy as np

R, Q, S0 = 0.02, 0.01, 100.0
_x, _w = np.polynomial.legendre.leggauss(24)
_U = 0.5 * 80.0 * (_x + 1)
_W = 0.5 * 80.0 * _w


def _phi(z, T, v0, kappa, theta, sigma, rho):
    # cf of ln(S_T/F) at complex argument z, little-trap form
    d = np.sqrt((kappa - rho * sigma * 1j * z) ** 2 + sigma ** 2 * (1j * z + z ** 2))
    g = (kappa - rho * sigma * 1j * z - d) / (kappa - rho * sigma * 1j * z + d)
    e = np.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((kappa - rho * sigma * 1j * z - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (kappa - rho * sigma * 1j * z - d) / sigma ** 2 * (1 - e) / (1 - g * e)
    return np.exp(C + D * v0)


class Model:
    name = "blind-P12"
    domain = {"v0": (0.01, 0.15), "kappa": (0.5, 5.0), "theta": (0.01, 0.15),
              "sigma": (0.1, 1.0), "rho": (-0.9, 0.0), "k": (-0.3, 0.3), "days": (30, 730)}

    def price(self, p, spot=None):
        T = p["days"] / 365.0
        s = S0 if spot is None else spot
        K = S0 * np.exp((R - Q) * T) * np.exp(p["k"])
        X = np.log(s / K) + (R - Q) * T
        ph = _phi(_U - 0.5j, T, p["v0"], p["kappa"], p["theta"], p["sigma"], p["rho"])
        integrand = np.real(np.exp(1j * _U * X) * ph) / (_U ** 2 + 0.25)
        I = np.sum(_W * integrand)
        c = s * np.exp(-Q * T) - np.sqrt(s * K) * np.exp(-(R + Q) * T / 2) / np.pi * I
        return float(max(c, 0.0))
