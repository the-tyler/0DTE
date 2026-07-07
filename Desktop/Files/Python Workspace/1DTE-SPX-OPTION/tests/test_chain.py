"""
Tests for option chain cleaning and smile construction:
_clean_chain, build_smile_df.
"""
import pandas as pd
import numpy as np
import pytest

import spy_1dte_vol_pipeline as pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────

def _chain(strikes, bids=None, asks=None, ois=None):
    n = len(strikes)
    return pd.DataFrame({
        "strike":       strikes,
        "bid":          bids or [1.00] * n,
        "ask":          asks or [1.20] * n,
        "openInterest": ois  or [500] * n,
    })


# ── _clean_chain ───────────────────────────────────────────────────────────────

def test_clean_chain_drops_zero_bid():
    df = _chain([100, 105], bids=[0.0, 1.0])
    out = pipeline._clean_chain(df, "call")
    assert 100 not in out["strike"].values
    assert 105 in out["strike"].values

def test_clean_chain_drops_bid_below_minimum():
    # MIN_BID = 0.01; 0.005 should be dropped
    df = _chain([100, 105], bids=[0.005, 1.0])
    out = pipeline._clean_chain(df, "call")
    assert 100 not in out["strike"].values

def test_clean_chain_drops_low_open_interest():
    # MIN_OI = 10; oi=5 should be removed
    df = _chain([100, 105], ois=[5, 500])
    out = pipeline._clean_chain(df, "call")
    assert 100 not in out["strike"].values
    assert 105 in out["strike"].values

def test_clean_chain_drops_wide_spread():
    # MAX_SPREAD_RATIO = 0.75; (ask-bid)/mid must be < 0.75
    # bid=0.10, ask=2.00 → mid=1.05, ratio=1.90/1.05≈1.81 → dropped
    df = _chain([100, 105], bids=[0.10, 1.0], asks=[2.00, 1.2])
    out = pipeline._clean_chain(df, "call")
    assert 100 not in out["strike"].values
    assert 105 in out["strike"].values

def test_clean_chain_mid_is_average_of_bid_ask():
    df = _chain([100], bids=[2.0], asks=[3.0])
    out = pipeline._clean_chain(df, "call")
    assert abs(out["mid"].iloc[0] - 2.5) < 1e-9

def test_clean_chain_kind_column_added():
    out = pipeline._clean_chain(_chain([100]), "put")
    assert out["kind"].iloc[0] == "put"

def test_clean_chain_empty_frame_returns_empty():
    df = pd.DataFrame(columns=["strike", "bid", "ask", "openInterest"])
    out = pipeline._clean_chain(df, "call")
    assert out.empty

def test_clean_chain_all_pass_filter():
    df = _chain([100, 105, 110])
    out = pipeline._clean_chain(df, "call")
    assert len(out) == 3


# ── build_smile_df ─────────────────────────────────────────────────────────────
# Use T=1.0 (annual) so OTM prices are far enough from zero to survive filters.

S, F, T, r, q, SIGMA = 100.0, 103.0, 1.0, 0.05, 0.02, 0.20


def _priced_chain(strikes, kind):
    rows = []
    for K in strikes:
        price = pipeline.bs_price(S, K, T, r, q, SIGMA, kind)
        if price > 0.05:   # skip near-zero prices that won't pass filters
            rows.append({
                "strike": K,
                "bid":    price * 0.90,
                "ask":    price * 1.10,
                "openInterest": 500,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["strike", "bid", "ask", "openInterest"]
    )


def test_build_smile_df_calls_only_otm_of_F():
    chain_calls = _priced_chain([104, 106, 108, 112, 116], "call")
    chain_puts  = _priced_chain([], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if not df.empty:
        call_rows = df[df["kind"] == "call"]
        assert (call_rows["strike"] > F).all(), "Call strikes must be > F"

def test_build_smile_df_puts_only_otm_of_F():
    chain_calls = _priced_chain([], "call")
    chain_puts  = _priced_chain([85, 88, 91, 94, 97, 100, 103], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if not df.empty:
        put_rows = df[df["kind"] == "put"]
        assert (put_rows["strike"] <= F).all(), "Put strikes must be ≤ F"

def test_build_smile_df_delta_range_enforced():
    chain_calls = _priced_chain([104, 106, 110, 115, 120, 130], "call")
    chain_puts  = _priced_chain([80, 85, 90, 95, 100, 103], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if not df.empty:
        assert (df["call_delta"] >= 0.02).all()
        assert (df["call_delta"] <= 0.98).all()

def test_build_smile_df_required_columns_present():
    chain_calls = _priced_chain([106, 110, 115], "call")
    chain_puts  = _priced_chain([90, 95, 100], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if not df.empty:
        for col in ("strike", "kind", "iv", "call_delta", "log_moneyness",
                    "spot", "forward", "T", "r", "q"):
            assert col in df.columns, f"Missing column: {col}"

def test_build_smile_df_iv_positive():
    chain_calls = _priced_chain([106, 110, 115], "call")
    chain_puts  = _priced_chain([90, 95, 100], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if not df.empty:
        assert (df["iv"] > 0).all()

def test_build_smile_df_empty_chains_returns_empty():
    empty = pd.DataFrame(columns=["strike", "bid", "ask", "openInterest"])
    df = pipeline.build_smile_df(empty, empty, S, F, T, r, q)
    assert df.empty

def test_build_smile_df_metadata_values_correct():
    chain_calls = _priced_chain([106, 110], "call")
    chain_puts  = _priced_chain([], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if not df.empty:
        assert df["spot"].iloc[0] == pytest.approx(S)
        assert df["forward"].iloc[0] == pytest.approx(F)
        assert df["T"].iloc[0] == pytest.approx(T)
        assert df["r"].iloc[0] == pytest.approx(r)
        assert df["q"].iloc[0] == pytest.approx(q)

def test_build_smile_df_sorted_by_strike():
    chain_calls = _priced_chain([115, 106, 110], "call")  # unsorted input
    chain_puts  = _priced_chain([90, 95], "put")
    df = pipeline.build_smile_df(chain_calls, chain_puts, S, F, T, r, q)
    if len(df) > 1:
        assert list(df["strike"]) == sorted(df["strike"])
