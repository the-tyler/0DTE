"""
Tests for cache I/O and rate-fetch functions:
_load_cache, _save_cache, fetch_risk_free_rate, fetch_dividend_yield.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

import spy_1dte_vol_pipeline as pipeline


# ── _load_cache / _save_cache ──────────────────────────────────────────────────

def test_load_cache_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "no_such_file.json")
    assert pipeline._load_cache() == {}

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    pipeline._save_cache({"risk_free": 0.045, "risk_free_date": "2026-01-01"})
    result = pipeline._load_cache()
    assert result["risk_free"] == pytest.approx(0.045)
    assert result["risk_free_date"] == "2026-01-01"

def test_save_cache_merges_existing_keys(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    pipeline._save_cache({"risk_free": 0.045})
    pipeline._save_cache({"div_yield": 0.008})
    result = pipeline._load_cache()
    assert "risk_free" in result
    assert "div_yield" in result

def test_save_cache_overwrites_existing_key(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    pipeline._save_cache({"risk_free": 0.045})
    pipeline._save_cache({"risk_free": 0.052})
    assert pipeline._load_cache()["risk_free"] == pytest.approx(0.052)

def test_load_cache_corrupted_json_returns_empty(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("{{not valid json")
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    assert pipeline._load_cache() == {}


# ── fetch_risk_free_rate ───────────────────────────────────────────────────────

def _fred_mock(csv_text: str):
    resp = MagicMock()
    resp.text = csv_text
    resp.raise_for_status = MagicMock()
    return resp

def test_fetch_risk_free_rate_from_fred(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    csv = "DATE,DTB4WK\n2026-07-01,5.10\n"
    with patch.object(pipeline.requests, "get", return_value=_fred_mock(csv)):
        r = pipeline.fetch_risk_free_rate()
    assert abs(r - 0.051) < 1e-6

def test_fetch_risk_free_rate_skips_missing_rows(tmp_path, monkeypatch):
    """FRED sometimes emits '.' for missing observations — must skip them."""
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    csv = "DATE,DTB4WK\n2026-06-28,.\n2026-07-01,4.80\n"
    with patch.object(pipeline.requests, "get", return_value=_fred_mock(csv)):
        r = pipeline.fetch_risk_free_rate()
    assert abs(r - 0.048) < 1e-6

def test_fetch_risk_free_rate_uses_cache_on_network_failure(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"risk_free": 0.041, "risk_free_date": "2026-06-01"}))
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    with patch.object(pipeline.requests, "get", side_effect=Exception("timeout")):
        r = pipeline.fetch_risk_free_rate()
    assert abs(r - 0.041) < 1e-6

def test_fetch_risk_free_rate_hardcoded_fallback_when_cache_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    with patch.object(pipeline.requests, "get", side_effect=Exception("timeout")):
        r = pipeline.fetch_risk_free_rate()
    assert r == pipeline.FALLBACK_R

def test_fetch_risk_free_rate_writes_to_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    csv = "DATE,DTB4WK\n2026-07-01,3.61\n"
    with patch.object(pipeline.requests, "get", return_value=_fred_mock(csv)):
        pipeline.fetch_risk_free_rate()
    saved = json.loads(cache_path.read_text())
    assert "risk_free" in saved
    assert abs(saved["risk_free"] - 0.0361) < 1e-6


# ── fetch_dividend_yield ───────────────────────────────────────────────────────

def _yf_mock(yield_value):
    ticker = MagicMock()
    ticker.info = {"trailingAnnualDividendYield": yield_value}
    return ticker

def test_fetch_dividend_yield_from_yfinance(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    with patch.object(pipeline.yf, "Ticker", return_value=_yf_mock(0.0090)):
        q = pipeline.fetch_dividend_yield()
    assert abs(q - 0.009) < 1e-6

def test_fetch_dividend_yield_rejects_implausible_value(tmp_path, monkeypatch):
    """Yield ≥ 20% is implausible for SPY; must fall through to cache/fallback."""
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    with patch.object(pipeline.yf, "Ticker", return_value=_yf_mock(0.50)):
        q = pipeline.fetch_dividend_yield()
    assert q == pipeline.FALLBACK_Q

def test_fetch_dividend_yield_uses_cache_on_empty_info(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"div_yield": 0.0076, "div_yield_date": "2026-06-01"}))
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    ticker = MagicMock()
    ticker.info = {}
    with patch.object(pipeline.yf, "Ticker", return_value=ticker):
        q = pipeline.fetch_dividend_yield()
    assert abs(q - 0.0076) < 1e-6

def test_fetch_dividend_yield_uses_cache_on_exception(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"div_yield": 0.0082}))
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    with patch.object(pipeline.yf, "Ticker", side_effect=Exception("network")):
        q = pipeline.fetch_dividend_yield()
    assert abs(q - 0.0082) < 1e-6

def test_fetch_dividend_yield_hardcoded_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "CACHE_FILE", tmp_path / "cache.json")
    ticker = MagicMock()
    ticker.info = {}
    with patch.object(pipeline.yf, "Ticker", return_value=ticker):
        q = pipeline.fetch_dividend_yield()
    assert q == pipeline.FALLBACK_Q

def test_fetch_dividend_yield_writes_to_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(pipeline, "CACHE_FILE", cache_path)
    with patch.object(pipeline.yf, "Ticker", return_value=_yf_mock(0.0085)):
        pipeline.fetch_dividend_yield()
    saved = json.loads(cache_path.read_text())
    assert "div_yield" in saved
    assert abs(saved["div_yield"] - 0.0085) < 1e-6
