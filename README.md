# valkit — model validation assistant

Python toolkit that runs a battery of validation tests on a pricing or risk model, with thresholds fixed before the run, and writes a deterministic `facts.json` per campaign.

**Status: v0.1, steps 1–7 of the plan** (VaR and pricing checks, findings/opinion/`guard`, public planted-defect bench, HTML report, YAML registry). Not done: third-party blind defects (protocol in `defects/README.md`), expert-written commentary, the `thresholds-v1` tag on GitHub, the demo GIF. Published reports live in `docs/` (GitHub Pages); thresholds are listed in `THRESHOLDS.md`.

```bash
pip install -e .[dev]
valkit fetch configs/riskhub_var_portfolio.yaml   # downloads prices, writes SHA-256 fingerprint (CSV is not redistributed)
valkit run configs/riskhub_var_portfolio.yaml     # -> campaigns/<model>/<date>/facts.json
valkit run configs/financepy_heston.yaml          # ~7 min on a 16 GB laptop
pytest
```

| ID | Test |
|---|---|
| BT-01 | Kupiec POF, with power vs 2p and measured size |
| BT-02 | Christoffersen independence / conditional coverage, with power vs clusters |
| BT-03 | Basel traffic light (+ multiplier add-on at N=250, 99 %) |
| BT-04 | Acerbi–Székely Z2 for ES |
| BM-02 | VaR vs `numpy.quantile` |
| BM-01 | Heston implied-vol error (bp) vs QuantLib analytic, and vs QuantLib Monte Carlo QE (outside its 99 % CI) |
| CV-01 | Monte Carlo standard-error slope vs paths |
| SE-01 | Finite-difference delta vs QuantLib |
| ST-01 | Stress parameters (Feller violation, rho ±0.95, one week, wings) |
| LU-01 | Largest scaled parameter box within tolerance |
| DR-01 | PSI / KS of inputs actually used vs declared domain (needs real usage data) |
| BM-03 | ES vs mean of the k worst returns of the same window (sharp where Z2 has little power) |
| RP-01 | Two runs give byte-identical facts |

Each campaign also writes `findings.json` (non-green tests, ranked, with a quantified impact: Basel multiplier add-on for VaR, price gap and proposed reserve on a test book for pricing), and `conditions_of_use.json` (deterministic opinion, validated domain). `valkit.findings.guard(path)` decorates `price(self, p)` and refuses inputs outside that domain, or any use when the opinion is "not fit for use". An indicative test (power < 0.5) can lower the opinion to "with conditions" at most, never to "not fit".

Measured, not assumed: at N=250, p=1 % the asymptotic Kupiec test rejects ≈ 9–10 % of correct models at the 5 % level, and has ≈ 24 % power against a doubled exception rate, so it is flagged *indicative*.

The riskhub-engine campaign is a **self-review** (author = developer), not an independent validation.

FinancePy and QuantLib's analytic engine both use the characteristic-function family, so their agreement (~1e-8 bp) is weak evidence; the Monte Carlo QE comparison is the only genuinely different method.

Each campaign also writes `report.html` (one-screen summary, then detail; prints to PDF from the browser) and upserts `registry/models.yaml` (pydantic-validated). The VaR campaign is labelled *self-review*.

## Defect bench (`valkit bench`, ~12 min)

Public taxonomy in `src/valkit/bench.py`: 6 VaR defects (40 seeds of iid Student-t data) and 7 pricing defects (12 seeds of Sobol test points), plus healthy models. *Flagged* = opinion is not "fit"; *rejected* = "not fit for use". 95 % Wilson intervals in `campaigns/bench/<date>/bench.json`.

First run (2026-11-05, before fixes): ES = VaR was flagged only 10/40 (the Z2 ES backtest has little power at 250 days), and the healthy VaR got an amber from some test in 9/40 runs. Two fixes, then a second run:
- ES is backtested on the full out-of-sample, and a deterministic benchmark BM-03 (ES vs the mean of the k worst returns) was added.
- A lone amber from a p-value test (BT-01/02/04) with nothing else non-green is listed as a finding but does not change the opinion ("not corroborated").

Second run: every VaR defect and every pricing defect is rejected (VaR 40/40, pricing 12/12); the healthy VaR is flagged 4/40 (95 % CI 4–23 %) and rejected 0/40; healthy FinancePy is flagged 0/12. The price of the corroboration rule: a real defect that only trips one p-value test would now go unflagged. BM-03 detects ES = VaR by construction (same convention as the model), not an ES that is wrong in a model-risk sense.

Limits: defects are written by the author (not blind), pricing defects are large relative to the tolerances, 12 seeds give wide intervals, and the Z2 null assumes a Gaussian tail (an amber BT-04 on real fat-tailed data, as in the VaR campaign, may come from that shortcut).
