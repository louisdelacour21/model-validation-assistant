"""Run the unchanged VaR battery on each blind model; prints statuses only, never the model source."""
import importlib.util
import json
import sys
from pathlib import Path

from valkit import data
from valkit.campaign import var_checks
from valkit.findings import opinion

r = data.portfolio_returns(data.DATA / "multi_asset.csv",
                           {"^GSPC": 0.4, "TLT": 0.25, "GLD": 0.15, "EURUSD=X": 0.1, "BTC-USD": 0.1})
out = {}
for d in sorted(Path("blind").glob("B*")):
    spec = importlib.util.spec_from_file_location(d.name, d / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    res = var_checks(m.Model(), r)
    out[d.name] = {"opinion": opinion(res), "status": {x.id: x.status for x in res}}
    print(d.name, out[d.name]["opinion"], out[d.name]["status"], flush=True)
Path("blind/results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
