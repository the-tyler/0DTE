"""
Tests for expiry logic: _next_trading_day, find_1dte_expiry.
"""
from datetime import date
from unittest.mock import MagicMock

import pytest

import spy_1dte_vol_pipeline as pipeline


# ── _next_trading_day ──────────────────────────────────────────────────────────
# Reference calendar (2026):
#   Mon Jul 6, Tue Jul 7, Wed Jul 8, Thu Jul 9, Fri Jul 10
#   Mon Jul 13 ...

def test_next_trading_day_monday(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 6))   # Monday
    assert pipeline._next_trading_day() == date(2026, 7, 7)    # Tuesday

def test_next_trading_day_tuesday(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 7))   # Tuesday
    assert pipeline._next_trading_day() == date(2026, 7, 8)    # Wednesday

def test_next_trading_day_thursday(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 9))   # Thursday
    assert pipeline._next_trading_day() == date(2026, 7, 10)   # Friday

def test_next_trading_day_friday_skips_weekend(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 10))  # Friday
    assert pipeline._next_trading_day() == date(2026, 7, 13)   # Monday

def test_next_trading_day_saturday_returns_monday(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 11))  # Saturday
    assert pipeline._next_trading_day() == date(2026, 7, 13)   # Monday

def test_next_trading_day_sunday_returns_monday(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 12))  # Sunday
    assert pipeline._next_trading_day() == date(2026, 7, 13)   # Monday


# ── find_1dte_expiry ───────────────────────────────────────────────────────────

def _mock_spy(options):
    spy = MagicMock()
    spy.options = tuple(options)
    return spy

def test_find_1dte_expiry_weekday_hit(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 6))   # Monday
    spy = _mock_spy(["2026-07-07", "2026-07-14", "2026-07-21"])
    result = pipeline.find_1dte_expiry(spy)
    assert result == ("2026-07-07", 1)

def test_find_1dte_expiry_friday_returns_3_cal_days(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 10))  # Friday
    spy = _mock_spy(["2026-07-13", "2026-07-14", "2026-07-17"])
    result = pipeline.find_1dte_expiry(spy)
    assert result == ("2026-07-13", 3)                          # Mon is 3 cal-days away

def test_find_1dte_expiry_not_listed_returns_none(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 6))   # Monday
    # Next business day (Tue Jul 7) is not in options (holiday, etc.)
    spy = _mock_spy(["2026-07-14", "2026-07-21"])
    assert pipeline.find_1dte_expiry(spy) is None

def test_find_1dte_expiry_empty_options_returns_none(monkeypatch):
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 6))
    assert pipeline.find_1dte_expiry(_mock_spy([])) is None

def test_find_1dte_expiry_ignores_further_expirations(monkeypatch):
    """Only the exact next business day qualifies — not any nearby date."""
    monkeypatch.setattr(pipeline, "TODAY", date(2026, 7, 6))   # Monday
    # Options listed from Wed onward; Tue (the target) is absent
    spy = _mock_spy(["2026-07-08", "2026-07-09", "2026-07-14"])
    assert pipeline.find_1dte_expiry(spy) is None
