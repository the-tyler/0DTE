"""
Tests for interpolate_pillars: cubic spline, linear fallback, edge cases.
"""
import numpy as np
import pandas as pd
import pytest

import spy_1dte_vol_pipeline as pipeline


def _df(deltas, ivs):
    return pd.DataFrame({"call_delta": list(deltas), "iv": list(ivs)})


# ── Basic contract ─────────────────────────────────────────────────────────────

def test_interpolate_returns_five_values():
    df = _df([0.10, 0.25, 0.50, 0.75, 0.90],
             [0.20, 0.16, 0.13, 0.16, 0.21])
    result = pipeline.interpolate_pillars(df)
    assert len(result) == len(pipeline.PILLARS)

def test_interpolate_values_are_floats():
    df = _df([0.10, 0.25, 0.50, 0.75, 0.90],
             [0.20, 0.16, 0.13, 0.16, 0.21])
    result = pipeline.interpolate_pillars(df)
    assert all(isinstance(v, float) for v in result)


# ── Empty / insufficient data ──────────────────────────────────────────────────

def test_interpolate_empty_df_all_nan():
    result = pipeline.interpolate_pillars(pd.DataFrame(columns=["call_delta", "iv"]))
    assert all(np.isnan(v) for v in result)

def test_interpolate_single_point_all_nan():
    result = pipeline.interpolate_pillars(_df([0.50], [0.13]))
    assert all(np.isnan(v) for v in result)

def test_interpolate_all_nan_iv_returns_nan():
    df = _df([0.10, 0.25, 0.50, 0.75, 0.90],
             [np.nan] * 5)
    result = pipeline.interpolate_pillars(df)
    assert all(np.isnan(v) for v in result)


# ── Linear fallback (< 4 points) ──────────────────────────────────────────────

def test_interpolate_three_points_uses_linear():
    df = _df([0.25, 0.50, 0.75], [0.15, 0.12, 0.15])
    result = pipeline.interpolate_pillars(df)
    # ATM pillar should be recovered exactly (data point at 0.50)
    assert abs(result[2] - 0.12) < 1e-6

def test_interpolate_two_points_linear():
    df = _df([0.40, 0.60], [0.14, 0.14])
    result = pipeline.interpolate_pillars(df)
    assert len(result) == 5
    # All five pillars: only 0.50 is in [0.40, 0.60]; rest → NaN
    assert not np.isnan(result[2])   # ATM is in range
    assert np.isnan(result[0])       # 10Δ out of range
    assert np.isnan(result[4])       # 10Δ-Put out of range


# ── Cubic spline (≥ 4 points) ─────────────────────────────────────────────────

def test_interpolate_four_points_cubic_no_error():
    df = _df([0.20, 0.40, 0.60, 0.80],
             [0.18, 0.14, 0.14, 0.18])
    result = pipeline.interpolate_pillars(df)
    assert len(result) == 5
    # ATM (0.50) is in [0.20, 0.80] → should not be NaN
    assert not np.isnan(result[2])

def test_interpolate_five_points_atm_exact():
    """When the data has a point at exactly 0.50, ATM pillar == that IV."""
    df = _df([0.10, 0.25, 0.50, 0.75, 0.90],
             [0.21, 0.16, 0.13, 0.16, 0.22])
    result = pipeline.interpolate_pillars(df)
    assert abs(result[2] - 0.13) < 1e-4   # ATM index = 2

def test_interpolate_cubic_gives_smooth_result():
    """Cubic spline values should be positive and monotonically reasonable."""
    df = _df([0.10, 0.25, 0.50, 0.75, 0.90],
             [0.25, 0.18, 0.14, 0.18, 0.26])
    result = pipeline.interpolate_pillars(df)
    # All five pillars are in range → none should be NaN
    assert not any(np.isnan(v) for v in result)
    # IV values should be positive
    assert all(v > 0 for v in result if not np.isnan(v))


# ── Out-of-range pillars ───────────────────────────────────────────────────────

def test_interpolate_out_of_range_pillars_are_nan():
    """Pillars outside the observed delta range get NaN (bounds_error=False)."""
    df = _df([0.40, 0.45, 0.50, 0.55, 0.60],
             [0.14, 0.13, 0.12, 0.13, 0.14])
    result = pipeline.interpolate_pillars(df)
    assert np.isnan(result[0])   # 10Δ-Call  (0.10) below min 0.40
    assert np.isnan(result[1])   # 25Δ-Call  (0.25) below min 0.40
    assert np.isnan(result[3])   # 25Δ-Put   (0.75) above max 0.60
    assert np.isnan(result[4])   # 10Δ-Put   (0.90) above max 0.60
    assert not np.isnan(result[2])  # ATM (0.50) in range


# ── Deduplication ──────────────────────────────────────────────────────────────

def test_interpolate_duplicate_deltas_handled():
    """Duplicate call_delta rows must not crash the interpolation."""
    df = _df([0.10, 0.25, 0.50, 0.50, 0.75, 0.90],
             [0.20, 0.16, 0.13, 0.13, 0.16, 0.21])
    result = pipeline.interpolate_pillars(df)
    assert len(result) == 5
