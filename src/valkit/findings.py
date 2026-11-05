"""From test results to ranked findings, a deterministic opinion and the conditions of use."""
import json
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from .adapters.pricing import ql_analytic, sobol_points
from .core import TestResult

RANK = {"green": 0, "amber": 1, "red": 2}
OPINION = {0: "fit for use within the stated conditions", 1: "fit for use with conditions and remediation",
           2: "not fit for use"}


@dataclass
class Finding:
    test_id: str
    severity: str
    summary: str
    impact: str
    indicative: bool


def book_gap(model, n: int = 32, notional: float = 1000.0, seed: int = 21) -> float:
    """Total absolute price gap vs QuantLib analytic on a Sobol test book of `n` calls of `notional` units."""
    return float(sum(abs(model.price(p) - ql_analytic(p)) for p in sobol_points(n, seed=seed)) * notional)


def _impact(r: TestResult, book: float | None) -> str:
    if r.id == "BT-03":
        a = r.evidence.get("multiplier_addon")
        return f"Basel multiplier add-on +{a:.2f} (3.00 -> {3 + a:.2f})" if a else "no add-on (N,p not 250,1%)"
    if r.id.startswith(("BM-01", "SE-01", "ST-01", "LU-01")) and book is not None:
        return f"price gap {book:.2f} on a 32-call test book (1000 units each); proposed reserve {book:.2f}"
    return "not quantified in v1"


def findings(results: list[TestResult], book: float | None = None) -> list[Finding]:
    out = [Finding(r.id, r.status, f"{r.name}: {r.value:.4g} ({r.status}; thresholds {r.threshold})",
                   _impact(r, book), r.indicative) for r in results if r.status != "green"]
    return sorted(out, key=lambda f: (-RANK[f.severity], f.test_id))


STATISTICAL = {"BT-01", "BT-02", "BT-04"}  # p-value tests: with several run at once, one amber is expected by chance


def verdict(results: list[TestResult]) -> int:
    """0 fit, 1 with conditions, 2 not fit. Worst status decides, except: red from an indicative test
    (power < 0.5) counts as amber, and a lone amber p-value test with nothing else non-green is not corroborated
    (it stays listed as a finding)."""
    bad = [r for r in results if r.status != "green"]
    if len(bad) == 1 and bad[0].id in STATISTICAL and bad[0].status == "amber":
        return 0
    return max((min(RANK[r.status], 1) if r.indicative else RANK[r.status] for r in results), default=0)


def opinion(results: list[TestResult]) -> str:
    return OPINION[verdict(results)]


class OutOfDomain(ValueError):
    pass


def conditions_of_use(model, results: list[TestResult]) -> dict:
    lu = next((r for r in results if r.id == "LU-01"), None)
    c = {"model": model.name, "opinion": opinion(results),
         "open_findings": [f.test_id for f in findings(results)]}
    if hasattr(model, "domain"):
        c["domain"] = {k: list(v) for k, v in model.domain.items()}
        c["validated_scale_tested"] = lu.value if lu else None
    return c


def write_conditions(model, results, out: Path) -> Path:
    path = out / "conditions_of_use.json"
    path.write_text(json.dumps(conditions_of_use(model, results), indent=1, sort_keys=True), encoding="utf-8")
    return path


def guard(path: Path):
    """Decorator for `price(self, p, ...)`: refuses a point outside the domain in conditions_of_use.json,
    or any use when the opinion is 'not fit for use'."""
    c = json.loads(Path(path).read_text(encoding="utf-8"))

    def deco(f):
        @wraps(f)
        def wrapped(self, p, *a, **kw):
            if c["opinion"] == OPINION[2]:
                raise OutOfDomain("model is not fit for use")
            bad = [k for k, (lo, hi) in c["domain"].items() if not lo <= p[k] <= hi]
            if bad:
                raise OutOfDomain(f"outside validated domain: {bad}")
            return f(self, p, *a, **kw)
        return wrapped
    return deco
