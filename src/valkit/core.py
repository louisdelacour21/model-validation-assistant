import json
from dataclasses import asdict, dataclass, field

STATUSES = ("green", "amber", "red")


@dataclass
class TestResult:
    """One validation test: metric, thresholds fixed beforehand, status, power if statistical."""
    __test__ = False  # not a pytest class

    id: str
    name: str
    value: float
    status: str
    threshold: dict
    power: float | None = None
    seed: int | None = None
    evidence: dict = field(default_factory=dict)

    @property
    def indicative(self) -> bool:
        # a test that detects a 2x exception rate less than half the time is not conclusive
        return self.power is not None and self.power < 0.5


def p_status(p: float, amber: float = 0.05, red: float = 0.01) -> str:
    return "green" if p >= amber else "amber" if p >= red else "red"


def facts_json(results: list[TestResult]) -> str:
    """Deterministic serialisation: two runs on the same inputs must give identical text."""
    return json.dumps([asdict(r) | {"indicative": r.indicative} for r in results],
                      sort_keys=True, indent=1, default=float)
