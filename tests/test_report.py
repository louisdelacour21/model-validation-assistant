from valkit.core import TestResult
from valkit.findings import Finding
from valkit.registry import Entry, update_registry
from valkit.report import render_report


def test_report_renders_and_escapes(tmp_path):
    rs = [TestResult("BT-01", "<b>x</b>", 0.5, "amber", {"a": 1}, 0.3, 1, {"k": 1})]
    f = [Finding("BT-01", "amber", "s", "i", True)]
    out = render_report(tmp_path / "r.html", "m", "2026-01-01", "self-review", rs, f,
                        {"opinion": "fit for use with conditions and remediation", "model": "m"})
    t = out.read_text(encoding="utf-8")
    assert "&lt;b&gt;" in t and "<b>x</b>" not in t and "not an independent validation" in t


def test_registry_upserts(tmp_path):
    e = dict(name="m", kind="var", independence="self-review", opinion="o", campaign_date="d", campaign_dir="c",
             open_findings=[])
    p = tmp_path / "m.yaml"
    update_registry(p, Entry(**e))
    update_registry(p, Entry(**e | {"opinion": "o2"}))
    t = p.read_text(encoding="utf-8")
    assert t.count("name: m") == 1 and "o2" in t
