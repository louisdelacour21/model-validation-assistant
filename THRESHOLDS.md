# Pre-registered thresholds

Thresholds below are fixed in the code at the commit tagged `thresholds-v1`, before any campaign is published as a result. A campaign may be re-run; a threshold may not be changed to fit a result without a new tag and a changelog entry here.

Honest caveat: the defect bench was run and two rules were changed (BM-03 added, corroboration of lone p-value ambers) *before* this tag, after seeing a first bench run. They are documented in the README.

| ID | Test | Thresholds |
|---|---|---|
| BM-01 | IV error vs QuantLib analytic + MC QE (bp) | `{"amber": "mean<3 and p99<25", "green": "mean<1 and p99<10"}` |
| BM-02 | VaR vs numpy.quantile | `{"amber_lt": 0.02, "green_lt": 0.001}` |
| BM-03 | ES vs mean of k worst returns | `{"amber_lt": 0.02, "green_lt": 0.001}` |
| BT-01 | Kupiec POF | `{"amber_p": 0.05, "red_p": 0.01}` |
| BT-02 | Christoffersen independence / conditional coverage | `{"amber_p": 0.05, "red_p": 0.01}` |
| BT-03 | Basel traffic light | `{"amber_cdf": 0.95, "red_cdf": 0.9999}` |
| BT-04 | Expected shortfall Z2 (max p over Gaussian and Student-t(4) nulls) | `{"amber_p": 0.05, "red_p": 0.01}` |
| CV-01 | MC standard-error slope vs paths | `{"amber": "[-0.7,-0.3]", "green": "[-0.6,-0.4]"}` |
| DR-01 | Input drift vs declared domain (PSI) | `{"amber_lt": 0.25, "green_lt": 0.1}` |
| LU-01 | Largest scaled box with p99 IV error under tolerance | `{"green": "declared domain (scale 1) within tolerance", "tol_p99_bp": 10}` |
| RP-01 | Two runs give identical facts | `{"identical": true}` |
| SE-01 | Delta (central FD) vs QuantLib | `{"amber_lt": 0.05, "green_lt": 0.01}` |
| ST-01 | Stress of parameters | `{"green": "0 violation inside the domain"}` |

Opinion rule (`valkit.findings.verdict`): worst status decides; red from an indicative test (power < 0.5) counts as amber; a lone amber from BT-01/02/04 with nothing else non-green does not change the opinion.
