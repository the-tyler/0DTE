"""
Tests for the Black-Scholes core: _d1d2, bs_price, bs_call_delta, implied_vol.
"""
import math
import numpy as np
import pytest

import spy_1dte_vol_pipeline as pipeline

# ── Shared fixtures ────────────────────────────────────────────────────────────
S, K, r, q, T, SIGMA = 100.0, 100.0, 0.05, 0.02, 1.0, 0.20


# ── _d1d2 ──────────────────────────────────────────────────────────────────────

def test_d1d2_values():
    d1, d2 = pipeline._d1d2(S, K, T, r, q, SIGMA)
    # With S=K: ln(S/K)=0 → d1 = (r-q+0.5σ²)/σ = (0.05-0.02+0.02)/0.20 = 0.25
    assert abs(d1 - 0.25) < 1e-10
    assert abs(d2 - 0.05) < 1e-10

def test_d1d2_d2_equals_d1_minus_sigma_sqrtT():
    d1, d2 = pipeline._d1d2(S, K * 1.1, T, r, q, SIGMA)
    assert abs(d2 - (d1 - SIGMA * math.sqrt(T))) < 1e-12


# ── bs_price ───────────────────────────────────────────────────────────────────

def test_put_call_parity():
    """C − P = S·e^(−qT) − K·e^(−rT) for all sigma."""
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    for sigma in [0.05, 0.10, 0.20, 0.50, 1.0, 2.0]:
        call = pipeline.bs_price(S, K, T, r, q, sigma, "call")
        put  = pipeline.bs_price(S, K, T, r, q, sigma, "put")
        assert abs((call - put) - rhs) < 1e-8, f"PCP failed at sigma={sigma}"

def test_bs_price_intrinsic_call_T0():
    assert abs(pipeline.bs_price(105, 100, 0, r, q, SIGMA, "call") - 5.0) < 1e-9
    assert pipeline.bs_price(95, 100, 0, r, q, SIGMA, "call") == 0.0

def test_bs_price_intrinsic_put_T0():
    assert abs(pipeline.bs_price(95, 100, 0, r, q, SIGMA, "put") - 5.0) < 1e-9
    assert pipeline.bs_price(105, 100, 0, r, q, SIGMA, "put") == 0.0

def test_bs_price_non_negative():
    for sigma in [0.05, 0.20, 1.0, 2.0]:
        assert pipeline.bs_price(S, K, T, r, q, sigma, "call") > 0
        assert pipeline.bs_price(S, K, T, r, q, sigma, "put") > 0

def test_bs_price_higher_vol_means_higher_price():
    p_lo = pipeline.bs_price(S, K, T, r, q, 0.10, "call")
    p_hi = pipeline.bs_price(S, K, T, r, q, 0.40, "call")
    assert p_hi > p_lo

def test_bs_price_sigma_zero_returns_intrinsic():
    # sigma=0 hits the T<=0 or sigma<=0 branch → intrinsic
    assert pipeline.bs_price(105, 100, T, r, q, 0, "call") == 5.0
    assert pipeline.bs_price(95, 100, T, r, q, 0, "put") == 5.0


# ── bs_call_delta ──────────────────────────────────────────────────────────────

def test_call_delta_in_unit_interval():
    for sigma in [0.10, 0.20, 0.50, 1.0]:
        d = pipeline.bs_call_delta(S, K, T, r, q, sigma)
        assert 0.0 < d < 1.0, f"Delta out of (0,1) for sigma={sigma}"

def test_call_delta_deep_itm_near_one():
    d = pipeline.bs_call_delta(200, 100, T, r, q, 0.20)
    assert d > 0.95

def test_call_delta_deep_otm_near_zero():
    d = pipeline.bs_call_delta(50, 200, T, r, q, 0.20)
    assert d < 0.05

def test_call_delta_atm_near_half():
    # ATM 1DTE with zero dividends: Δ ≈ N(d1) ≈ 0.5
    d = pipeline.bs_call_delta(100, 100, 1 / 365, 0.05, 0.0, 0.20)
    assert abs(d - 0.5) < 0.10

def test_call_delta_nan_on_T0():
    assert np.isnan(pipeline.bs_call_delta(S, K, 0, r, q, SIGMA))

def test_call_delta_nan_on_sigma0():
    assert np.isnan(pipeline.bs_call_delta(S, K, T, r, q, 0.0))

def test_call_delta_nan_on_sigma_nan():
    assert np.isnan(pipeline.bs_call_delta(S, K, T, r, q, np.nan))


# ── implied_vol ────────────────────────────────────────────────────────────────

def test_implied_vol_call_roundtrip():
    """Price a call then recover sigma; must match to 1e-5 across wide range."""
    for sigma in [0.05, 0.10, 0.20, 0.50, 1.0, 2.0]:
        price = pipeline.bs_price(S, K, T, r, q, sigma, "call")
        iv = pipeline.implied_vol(price, S, K, T, r, q, "call")
        assert abs(iv - sigma) < 1e-5, f"Call roundtrip failed at sigma={sigma}"

def test_implied_vol_put_roundtrip():
    for sigma in [0.10, 0.20, 0.50, 1.0]:
        price = pipeline.bs_price(S, K, T, r, q, sigma, "put")
        iv = pipeline.implied_vol(price, S, K, T, r, q, "put")
        assert abs(iv - sigma) < 1e-5, f"Put roundtrip failed at sigma={sigma}"

def test_implied_vol_1dte_realistic():
    """Roundtrip under conditions the pipeline actually encounters."""
    S_spy, K_spy, T_1 = 750.0, 748.0, 1 / 365
    r_s, q_s, sigma_s = 0.036, 0.0076, 0.15
    price = pipeline.bs_price(S_spy, K_spy, T_1, r_s, q_s, sigma_s, "call")
    iv = pipeline.implied_vol(price, S_spy, K_spy, T_1, r_s, q_s, "call")
    assert abs(iv - sigma_s) < 1e-4

def test_implied_vol_nan_price_zero():
    assert np.isnan(pipeline.implied_vol(0.0, S, K, T, r, q, "call"))

def test_implied_vol_nan_negative_price():
    assert np.isnan(pipeline.implied_vol(-1.0, S, K, T, r, q, "call"))

def test_implied_vol_nan_T_zero():
    assert np.isnan(pipeline.implied_vol(5.0, S, K, 0.0, r, q, "call"))

def test_implied_vol_nan_below_intrinsic():
    # Deep ITM call: intrinsic = 50; price = 0.01 ≪ intrinsic → NaN
    assert np.isnan(pipeline.implied_vol(0.01, 100, 50, T, r, q, "call"))
