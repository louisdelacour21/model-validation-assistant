import argparse
from pathlib import Path

import yaml

from . import data
from .campaign import run_pricing_campaign, run_var_campaign


def main(argv=None):
    ap = argparse.ArgumentParser(prog="valkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch").add_argument("config")
    sub.add_parser("run").add_argument("config")
    sub.add_parser("bench")
    a = ap.parse_args(argv)
    if a.cmd == "bench":
        from .bench import run_bench
        print(run_bench(Path("campaigns")))
        return
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if cfg["model"] in ("financepy-heston", "neural-heston"):
        print(run_pricing_campaign(cfg, Path("campaigns")) / "facts.json")
        return
    d = cfg["data"]
    if a.cmd == "fetch":
        print(data.download(d["tickers"], d["start"], d["end"], d["name"]))
        return
    path = data.DATA / f"{d['name']}.csv"
    if not data.verify(path):
        raise SystemExit(f"{path} missing or fingerprint mismatch: run `valkit fetch {a.config}`")
    out = run_var_campaign(cfg, data.portfolio_returns(path, d["weights"]), Path("campaigns"))
    print(out / "facts.json")


if __name__ == "__main__":
    main()
