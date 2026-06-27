from pathlib import Path

from energy_monitor.config import load_config


def test_config_loads_three_sites() -> None:
    cfg = load_config(Path("config/sites.yaml"))
    assert len(cfg.sites) == 3


def test_site_keys_are_correct() -> None:
    cfg = load_config(Path("config/sites.yaml"))
    keys = [s.key for s in cfg.sites]
    assert "burgos_ph" in keys
    assert "sevilla_es" in keys
    assert "esbjerg_dk" in keys


def test_date_window_is_valid() -> None:
    cfg = load_config(Path("config/sites.yaml"))
    assert cfg.window.start < cfg.window.end
