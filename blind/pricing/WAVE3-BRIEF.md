Write a European call pricer under Heston in `blind/pricing/{ID}/model.py` (Python, numpy/scipy only; do NOT import valkit, financepy, QuantLib or read other folders in blind/).

Interface:
    class Model:
        name = "blind-{ID}"
        domain = {"v0":(lo,hi), "kappa":(lo,hi), "theta":(lo,hi), "sigma":(lo,hi), "rho":(lo,hi), "k":(lo,hi), "days":(lo,hi)}   # the domain you validate on
        def price(self, p, spot=None) -> float

`p` has keys v0, kappa, theta, sigma, rho, k, days. k = ln(K/F) where F = S0*exp((r-q)*T) and S0 = 100 is the reference spot. T = days/365 (ACT/365). Market: S0=100, r=0.02 (continuous), q=0.01 (continuous). Return the call price per unit notional.

Spot bump: when `spot` is given, the current spot is `spot` instead of 100, but THE STRIKE STAYS FIXED at K = 100*exp((r-q)*T)*exp(k) (derived from the reference spot 100, not from `spot`). Forward and discounting use r, q as above.

Truth draw: before writing the file, run in Python `import secrets; secrets.choice(["healthy","defective"])`. If defective, also draw the defect from a list you write yourself (at least 6 distinct realistic defects, mixing obvious and subtle ones, e.g. wrong discounting, truncated integral, wrong characteristic function branch, sign error on rho, etc.) with `secrets.choice`. Implement the draw faithfully; if healthy, write a correct, careful implementation.
Write `blind/pricing/{ID}/answer.json` with {"truth": ..., "defect": ...}. Do not mention the draw, the defect, or your reasoning about it anywhere else: no comments in model.py hinting at it, and reply to me only with "done" and the file path.
Do not run or look at any validator. Do not read other blind/ folders.
