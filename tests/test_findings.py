import json
import pytest
from valkit.adapters.pricing import FinancePyHeston, sobol_points
from valkit.core import TestResult
from valkit.findings import OPINION, OutOfDomain, conditions_of_use, findings, guard, opinion, write_conditions


def R(i, s, power=None, ev=None):
    return TestResult(i, i, 1.0, s, {}, power, None, ev or {})


def test_opinion_follows_worst_status_and_indicative_red_is_capped():
    assert opinion([R("a", "green")]) == OPINION[0]
    assert opinion([R("a", "green"), R("b", "amber")]) == OPINION[1]
    assert opinion([R("a", "red")]) == OPINION[2]
    assert opinion([R("a", "red", power=0.2)]) == OPINION[1]


def test_findings_ranked_with_basel_impact():
    fs = findings([R("BT-01", "amber"), R("BT-03", "red", ev={"multiplier_addon": 1.0})])
    assert [f.test_id for f in fs] == ["BT-03", "BT-01"] and "4.00" in fs[0].impact


def test_guard_refuses_outside_domain_and_when_not_fit(tmp_path):
    m = FinancePyHeston()
    path = write_conditions(m, [R("BM-01", "green")], tmp_path)

    class G(FinancePyHeston):
        price = guard(path)(FinancePyHeston.price)
    p = sobol_points(1)[0]
    assert G().price(p) > 0
    with pytest.raises(OutOfDomain):
        G().price(p | {"sigma": 2.0})
    path.write_text(json.dumps(conditions_of_use(m, [R("BM-01", "red")])), encoding="utf-8")
    with pytest.raises(OutOfDomain):
        guard(path)(FinancePyHeston.price)(m, p)


def test_lone_statistical_amber_is_not_corroborated():
    from valkit.findings import verdict
    assert verdict([R("BT-01", "amber"), R("BT-03", "green")]) == 0
    assert verdict([R("BT-01", "amber"), R("BT-02", "amber")]) == 1
    assert verdict([R("BT-03", "amber")]) == 1
