import json

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binom

from valkit import cli, data
from valkit.campaign import run_var_campaign
from valkit.checks import backtesting as bt
from valkit.checks.backtesting import _z2_null


def _prices(tmp_path):
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (1000, 2)), axis=0)), columns=["A", "B"])
    path = tmp_path / "p.csv"
    px.to_csv(path)
    return path


def test_portfolio_returns_weighted(tmp_path):
    path = _prices(tmp_path)
    r = data.portfolio_returns(path, {"A": 0.5, "B": 0.5})
    px = pd.read_csv(path, index_col=0).pct_change().dropna()
    assert len(r) == 999 and np.allclose(r, px.mean(axis=1))


def test_verify_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA", tmp_path)
    path = _prices(tmp_path)
    assert not data.verify(path)
    import hashlib
    (tmp_path / "fingerprints.json").write_text(json.dumps({"p.csv": hashlib.sha256(path.read_bytes()).hexdigest()}))
    assert data.verify(path)
    path.write_text(path.read_text() + "x")
    assert not data.verify(path)


def test_var_campaign_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = data.portfolio_returns(_prices(tmp_path), {"A": 0.5, "B": 0.5})
    out = run_var_campaign({"model": "riskhub-historical"}, r, tmp_path / "campaigns")
    for f in ("facts.json", "findings.json", "conditions_of_use.json", "report.html"):
        assert (out / f).exists()
    reg = yaml.safe_load((tmp_path / "registry" / "models.yaml").read_text())
    assert reg[0]["name"] == "riskhub-historical" and reg[0]["independence"] == "self-review"


def test_cli_requires_fingerprint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "DATA", tmp_path)
    cfg = tmp_path / "c.yaml"
    d = {"tickers": ["A"], "start": "x", "end": "y", "name": "nope", "weights": {"A": 1}}
    cfg.write_text(yaml.safe_dump({"model": "riskhub-historical", "data": d}))
    try:
        cli.main(["run", str(cfg)])
    except SystemExit as e:
        assert "fingerprint" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_kupiec_exact_agrees_in_direction():
    r = bt.bt_kupiec([1] * 12 + [0] * 238, 0.01, n_sims=50).evidence
    assert r["exact_binomial_p"] < 0.01 and r["exact_binomial_p"] <= 2 * binom.sf(11, 250, 0.01)


def test_kupiec_size_and_power_at_5pct_level():
    h = (np.random.default_rng(1).random(250) < 0.05).astype(int)
    res = bt.bt_kupiec(h, 0.05, n_sims=300)
    assert res.evidence["size_at_5pct"] < 0.15 and res.power > 0.5  # the 5 % VaR level has far more power than 1 %


def test_z2_null_centred_and_fat_tail_wider():
    g, t = _z2_null(1000, 0.05, 2000, 1), _z2_null(1000, 0.05, 2000, 1, df=4)
    assert abs(g.mean()) < 0.05 and abs(t.mean()) < 0.1 and t.std() > g.std()
