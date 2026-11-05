# Adding your own model

`valkit` tests any model that fits one of two small interfaces.

## A VaR/ES model

Implement `valkit.adapters.base.RiskModel`: three attributes and one method.

```python
import numpy as np

class MyVaR:
    name = "my-var"
    conf = 0.99          # declared confidence level
    window = 500         # history length the model uses

    def forecast(self, history: np.ndarray) -> tuple[float, float]:
        """(VaR, ES) as positive losses, from returns observed before the forecast day."""
```

Register it in `valkit.campaign.MODELS`, write a config like `configs/riskhub_var_portfolio.yaml`
(tickers, weights, dates), then:

```bash
valkit fetch configs/my_var.yaml   # downloads prices, writes a SHA-256 fingerprint
valkit run configs/my_var.yaml     # BT-01..04, BM-02/03, RP-01 -> findings, opinion, report.html
```

## A pricing model

Follow `valkit.adapters.pricing.FinancePyHeston`: a `price(params, spot=None)` method over the Heston
parameter dictionary, registered in `PRICERS`. The pricing checks compare against QuantLib.

## Before trusting a result

1. Run `valkit bench` and read the defect table: it says which defects your test battery cannot see.
2. Thresholds are in `THRESHOLDS.md`; do not change them after looking at a campaign.
3. A model you wrote yourself gets `independence: self-review`, which the report prints on its first line.
