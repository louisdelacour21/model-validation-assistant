"""Self-contained HTML report: a one-screen summary, then the detail per test. Print to PDF from the browser."""
import json
from pathlib import Path

from jinja2 import Environment

from .core import TestResult

TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{{ model }} validation report</title>
<style>
:root{--bg:#fff;--fg:#1c2330;--mut:#5d6778;--line:#d9dee7;--green:#1b7f4b;--amber:#a86a00;--red:#b3261e;--card:#f5f7fa}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12161d;--fg:#e6e9ef;--mut:#9aa4b5;--line:#2c3442;--green:#4cc38a;--amber:#e0a43b;--red:#ff7b72;--card:#1a2029}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.1rem;margin:28px 0 8px}.mut{color:var(--mut)}
.banner{padding:14px 16px;border-radius:8px;background:var(--card);border-left:6px solid var(--c);margin:16px 0}
.banner b{font-size:1.15rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.s{font-weight:600}.green{color:var(--green)}.amber{color:var(--amber)}.red{color:var(--red)}
details{margin:6px 0}pre{background:var(--card);padding:10px;overflow:auto;font-size:.8rem;border-radius:6px}
@media print{details{display:block}details>summary{display:none}}
</style></head><body><main>
<h1>{{ model }} — validation report</h1>
<div class="mut">Campaign {{ date }} · {{ independence }}{% if independence == 'self-review' %} (author = developer: not an independent validation){% endif %}</div>
<div class="banner" style="--c:var(--{{ tone }})"><b>Opinion: {{ opinion }}.</b><br>
{{ n_green }} of {{ results|length }} tests green · {{ findings|length }} finding(s) open.</div>

<h2>Findings</h2>
{% if findings %}<table><tr><th>Test</th><th>Severity</th><th>Finding</th><th>Impact</th></tr>
{% for f in findings %}<tr><td>{{ f.test_id }}</td><td class="s {{ f.severity }}">{{ f.severity }}{% if f.indicative %} (indicative){% endif %}</td><td>{{ f.summary }}</td><td>{{ f.impact }}</td></tr>{% endfor %}</table>
{% else %}<p>No test is amber or red.</p>{% endif %}

<h2>Conditions of use</h2>
{% if conditions.domain %}<p>Validated domain (the guard refuses inputs outside it):</p>
<table><tr>{% for k in conditions.domain %}<th>{{ k }}</th>{% endfor %}</tr><tr>{% for v in conditions.domain.values() %}<td>[{{ v[0] }}, {{ v[1] }}]</td>{% endfor %}</tr></table>
<p class="mut">Largest scaled box tested within tolerance: {{ conditions.validated_scale_tested }}× the declared domain.</p>
{% else %}<p>Daily {{ conditions.model }} VaR/ES at the declared confidence and window; monitor exceptions monthly and re-run the campaign after any change of data or code.</p>{% endif %}

{% if bench %}<h2>Defect bench (detection power of this battery)</h2>
<table><tr><th>Case</th><th>Flagged</th><th>Rejected</th></tr>
{% for k, v in bench.items() %}<tr><td>{{ k }}</td><td>{{ v.flagged }}/{{ v.n }} (CI {{ '%.0f'|format(v.flagged_ci95[0]*100) }}–{{ '%.0f'|format(v.flagged_ci95[1]*100) }} %)</td><td>{{ v.rejected }}/{{ v.n }}</td></tr>{% endfor %}</table>{% endif %}

<h2>Tests</h2>
<table><tr><th>ID</th><th>Test</th><th>Value</th><th>Status</th><th>Thresholds (fixed beforehand)</th><th>Power</th></tr>
{% for r in results %}<tr><td>{{ r.id }}</td><td>{{ r.name }}</td><td>{{ '%.4g'|format(r.value) }}</td><td class="s {{ r.status }}">{{ r.status }}</td><td>{{ r.threshold }}</td><td>{{ '%.2f'|format(r.power) if r.power is not none else '–' }}</td></tr>{% endfor %}</table>

<h2>Evidence</h2>
{% for r in results %}<details><summary>{{ r.id }} — {{ r.name }}</summary><pre>{{ evidence[r.id] }}</pre></details>{% endfor %}
</main></body></html>"""


def render_report(path: Path, model: str, date: str, independence: str, results: list[TestResult], findings: list,
                  conditions: dict, bench: dict | None = None) -> Path:
    opinion = conditions["opinion"]
    tone = "green" if opinion.startswith("fit for use within") else "amber" if "conditions" in opinion else "red"
    html = Environment(autoescape=True).from_string(TEMPLATE).render(
        model=model, date=date, independence=independence, results=results, findings=findings, conditions=conditions,
        opinion=opinion, tone=tone, n_green=sum(r.status == "green" for r in results), bench=bench,
        evidence={r.id: json.dumps(r.evidence, indent=1, sort_keys=True, default=str) for r in results})
    Path(path).write_text(html, encoding="utf-8")
    return Path(path)
