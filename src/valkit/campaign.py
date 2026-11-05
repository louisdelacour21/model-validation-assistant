import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from .adapters.base import RiskModel, rolling_forecasts
from .adapters.riskhub_var import RiskhubHistoricalVaR
from .checks import backtesting as bt
from .checks.benchmark import bm_es, bm_var, rp_reproducibility
from .adapters.pricing import FinancePyHeston, PARAMS, sobol_points
from .checks import pricing as pc
from .core import facts_json
from .registry import Entry, update_registry
from .report import render_report
from .findings import book_gap, findings, write_conditions

MODELS = {"riskhub-historical": RiskhubHistoricalVaR}


def var_checks(model: RiskModel, returns: np.ndarray):
    r, var, es = rolling_forecasts(model, returns)
    p = 1 - model.conf
    hits = (r < -var).astype(int)
    last = slice(-250, None)  # regulatory backtest window
    # ES is backtested on the full out-of-sample: 250 days hold ~2.5 tail points, too few to test ES
    return [
        bt.bt_kupiec(hits[last], p), bt.bt_christoffersen(hits[last], p), bt.bt_traffic_light(hits[last], p),
        bt.bt_es(r, var, es, p), bm_var(model, returns), bm_es(model, returns),
    ]


def run_var_campaign(cfg: dict, returns: np.ndarray, out_root: Path) -> Path:
    model = MODELS[cfg["model"]](**cfg.get("params", {}))
    facts = lambda: facts_json(var_checks(model, returns))
    results = var_checks(model, returns) + [rp_reproducibility(facts)]
    out = out_root / model.name / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.json").write_text(facts_json(results), encoding="utf-8")
    _write_findings(model, results, out)
    return out


PRICERS = {"financepy-heston": FinancePyHeston}


def pricing_checks(model, usage_box: dict | None = None):
    usage = sobol_points(200, {k: tuple(v) for k, v in usage_box.items()} if usage_box else None, seed=1)
    return [pc.bm_pricing(model), pc.cv_mc(), pc.se_delta(model), pc.st_stress(model), pc.lu_limits(model),
            pc.dr_drift({k: [p[k] for p in usage] for k in PARAMS})]


def run_pricing_campaign(cfg: dict, out_root: Path) -> Path:
    model = PRICERS[cfg["model"]]()
    facts = lambda: facts_json(pricing_checks(model, cfg.get("usage_box")))
    results = pricing_checks(model, cfg.get("usage_box")) + [rp_reproducibility(facts)]
    out = out_root / model.name / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "facts.json").write_text(facts_json(results), encoding="utf-8")
    _write_findings(model, results, out, book_gap(model) if any(r.status != 'green' for r in results) else None,
                    'pricing', 'independent')
    return out


def _write_findings(model, results, out: Path, book: float | None = None, kind: str = "var",
                    independence: str = "self-review"):
    fs = findings(results, book)
    (out / "findings.json").write_text(json.dumps([asdict(f) for f in fs], indent=1, sort_keys=True), encoding="utf-8")
    cond = json.loads(write_conditions(model, results, out).read_text(encoding="utf-8"))
    bench = sorted(Path("campaigns/bench").glob("*/bench.json"))
    bench = json.loads(bench[-1].read_text(encoding="utf-8"))[kind] if bench else None
    render_report(out / "report.html", model.name, out.name, independence, results, fs, cond, bench)
    update_registry(Path("registry/models.yaml"), Entry(
        name=model.name, kind=kind, independence=independence, opinion=cond["opinion"], campaign_date=out.name,
        campaign_dir=out.as_posix(), open_findings=cond["open_findings"]))
