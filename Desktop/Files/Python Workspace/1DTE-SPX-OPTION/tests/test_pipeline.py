"""
Tests for pipeline orchestration: _parse_args, _print_pillars, _run_historical.
"""
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

import spy_1dte_vol_pipeline as pipeline


# ── _parse_args ────────────────────────────────────────────────────────────────

def test_parse_args_no_flag_returns_today(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spy_1dte_vol_pipeline.py"])
    assert pipeline._parse_args() == date.today()

def test_parse_args_valid_date(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spy_1dte_vol_pipeline.py", "--date", "2026-06-30"])
    assert pipeline._parse_args() == date(2026, 6, 30)

def test_parse_args_invalid_format_raises_system_exit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spy_1dte_vol_pipeline.py", "--date", "30-06-2026"])
    with pytest.raises(SystemExit):
        pipeline._parse_args()

def test_parse_args_garbage_string_raises_system_exit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spy_1dte_vol_pipeline.py", "--date", "not-a-date"])
    with pytest.raises(SystemExit):
        pipeline._parse_args()


# ── _print_pillars ─────────────────────────────────────────────────────────────

def test_print_pillars_shows_all_labels(capsys):
    pillars = [0.20, 0.16, 0.13, 0.16, 0.21]
    pipeline._print_pillars(pillars)
    out = capsys.readouterr().out
    for label in pipeline.PILLAR_LABELS:
        assert label in out

def test_print_pillars_nan_shows_na(capsys):
    pipeline._print_pillars([np.nan] * 5)
    assert "N/A" in capsys.readouterr().out

def test_print_pillars_shows_pillar_deltas(capsys):
    pipeline._print_pillars([0.20, 0.16, 0.13, 0.16, 0.21])
    out = capsys.readouterr().out
    for d in pipeline.PILLARS:
        assert f"{d:.2f}" in out


# ── _run_historical ────────────────────────────────────────────────────────────

def _minimal_snapshot(run_date, tmp_path):
    """Write a minimal parquet snapshot to tmp_path and return the path."""
    S, F, T, r, q = 500.0, 500.5, 1 / 365, 0.05, 0.008
    df = pd.DataFrame({
        "strike":        [490.0, 495.0, 500.0, 505.0, 510.0],
        "kind":          ["put", "put", "put", "call", "call"],
        "bid":           [1.0] * 5,
        "ask":           [1.5] * 5,
        "mid":           [1.25] * 5,
        "iv":            [0.21, 0.17, 0.14, 0.17, 0.22],
        "call_delta":    [0.10, 0.25, 0.50, 0.75, 0.90],
        "log_moneyness": [-0.20, -0.10, 0.0, 0.10, 0.20],
        "as_of":         [str(run_date)] * 5,
        "expiry":        ["2026-07-01"] * 5,
        "spot":          [S] * 5,
        "forward":       [F] * 5,
        "T":             [T] * 5,
        "r":             [r] * 5,
        "q":             [q] * 5,
    })
    path = tmp_path / f"{run_date}.parquet"
    df.to_parquet(path, index=False)
    return path


def test_run_historical_missing_snapshot_prints_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "SNAP_DIR", tmp_path)
    pipeline._run_historical(date(2020, 1, 2))
    out = capsys.readouterr().out
    assert "No snapshot found" in out

def test_run_historical_missing_snapshot_lists_available(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "SNAP_DIR", tmp_path)
    # Put a real snapshot in the dir so the "available" list is populated
    _minimal_snapshot(date(2026, 6, 30), tmp_path)
    pipeline._run_historical(date(2020, 1, 2))
    out = capsys.readouterr().out
    assert "2026-06-30" in out

def test_run_historical_existing_snapshot_loads(tmp_path, monkeypatch, capsys):
    run_date = date(2026, 6, 30)
    monkeypatch.setattr(pipeline, "ROOT",     tmp_path)
    monkeypatch.setattr(pipeline, "SNAP_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PLOT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    # Stub plots so we don't need a display
    monkeypatch.setattr(pipeline, "plot_smile",          lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "plot_pillar_history", lambda *a, **kw: None)

    _minimal_snapshot(run_date, tmp_path)

    pipeline._run_historical(run_date)
    out = capsys.readouterr().out
    assert "Loaded" in out

def test_run_historical_existing_snapshot_prints_pillars(tmp_path, monkeypatch, capsys):
    run_date = date(2026, 6, 30)
    monkeypatch.setattr(pipeline, "ROOT",     tmp_path)
    monkeypatch.setattr(pipeline, "SNAP_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PLOT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "plot_smile",          lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "plot_pillar_history", lambda *a, **kw: None)

    _minimal_snapshot(run_date, tmp_path)

    pipeline._run_historical(run_date)
    out = capsys.readouterr().out
    # _print_pillars outputs the table header
    assert "Pillar" in out
    assert "Call" in out

def test_run_historical_existing_snapshot_all_done(tmp_path, monkeypatch, capsys):
    run_date = date(2026, 6, 30)
    monkeypatch.setattr(pipeline, "ROOT",     tmp_path)
    monkeypatch.setattr(pipeline, "SNAP_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "PLOT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "plot_smile",          lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "plot_pillar_history", lambda *a, **kw: None)

    _minimal_snapshot(run_date, tmp_path)

    pipeline._run_historical(run_date)
    out = capsys.readouterr().out
    assert "All done" in out
