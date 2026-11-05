"""Frozen portfolio data. Yahoo licence forbids redistribution: the repo ships the download
script and SHA-256 fingerprints, never the CSV."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"


def download(tickers: list[str], start: str, end: str, name: str = "prices") -> Path:
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    # shortcut: 7-day crypto aligned on the NYSE calendar by dropping days the index is closed
    px = px.dropna(subset=[t for t in tickers if t == "^GSPC"]).ffill().dropna()
    path = DATA / f"{name}.csv"
    px.to_csv(path)
    fp = DATA / "fingerprints.json"
    prints = json.loads(fp.read_text()) if fp.exists() else {}
    prints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    fp.write_text(json.dumps(prints, indent=1, sort_keys=True))
    return path


def verify(path: Path) -> bool:
    fp = DATA / "fingerprints.json"
    return fp.exists() and json.loads(fp.read_text()).get(path.name) == hashlib.sha256(path.read_bytes()).hexdigest()


def portfolio_returns(path: Path, weights: dict[str, float]) -> np.ndarray:
    """Daily-rebalanced fixed-weight simple returns."""
    px = pd.read_csv(path, index_col=0)
    r = px.pct_change().dropna()
    return (r[list(weights)] @ pd.Series(weights)).to_numpy()
