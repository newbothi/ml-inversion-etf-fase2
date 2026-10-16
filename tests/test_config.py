import pytest

from etfml.config import ConfigError, load_config


def test_default_config_loads(cfg):
    assert cfg.universe.benchmark in cfg.universe.tickers
    assert cfg.ib.model == "ensamble"
    assert not cfg.ib.allow_live


def test_invalid_model_is_rejected(tmp_path):
    text = load_config.__globals__["DEFAULT_CONFIG_PATH"].read_text(encoding="utf-8")
    bad = tmp_path / "config.toml"
    bad.write_text(text.replace('modelo = "ensamble"', 'modelo = "red_neuronal"'), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
