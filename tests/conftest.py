import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etfml.backtest import run_backtest  # noqa: E402
from etfml.config import load_config  # noqa: E402
from etfml.data import synthetic_market_data  # noqa: E402
from etfml.features import build_panel  # noqa: E402

TODAY = pd.Timestamp("2026-01-15")


def _run(signal_strength: float, seed: int):
    cfg = load_config()
    market = synthetic_market_data(signal_strength=signal_strength, seed=seed)
    cfg = cfg.with_universe(list(market.close.columns), market.close.columns[0])
    panel = build_panel(market, min_assets=cfg.strategy.min_assets, today=TODAY)
    return cfg, market, panel, run_backtest(panel, cfg, market.source)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def with_signal():
    return _run(1.0, seed=7)


@pytest.fixture(scope="session")
def pure_noise():
    return _run(0.0, seed=3)
