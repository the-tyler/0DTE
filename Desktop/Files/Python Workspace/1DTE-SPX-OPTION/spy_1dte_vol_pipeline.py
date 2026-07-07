#!/usr/bin/env python3
"""
SPY 1-Day-To-Maturity Implied Volatility Pipeline

Pulls the SPY option chain, computes Black-Scholes implied volatility,
call delta, and standardised log-moneyness for each OTM strike.
Saves a daily parquet snapshot, interpolates 10/25/ATM/25/10-delta
pillars, and produces a two-panel smile plot plus a pillar-history chart.
"""

import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; plots saved to disk
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
SNAP_DIR   = DATA_DIR / "snapshots"   # one .parquet per run-date
PLOT_DIR   = DATA_DIR / "plots"       # smile_<date>.png, pillar_history_<date>.png
CACHE_FILE = DATA_DIR / "rates_cache.json"

for _d in (SNAP_DIR, PLOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
TICKER = "SPY"
TODAY  = date.today()

MIN_OI           = 10   # minimum open interest per strike
MIN_BID          = 0.01
MAX_SPREAD_RATIO = 0.75  # (ask-bid)/mid — drop illiquid strikes

IV_LO, IV_HI = 1e-4, 30.0  # brentq search bounds (annualised)

# Delta pillars expressed as CALL delta
# Sorted low→high call-Δ  →  high→low strike (put wing left, call wing right)
#   0.10 = deep-OTM call (10Δ-call)
#   0.25 = OTM call      (25Δ-call)
#   0.50 = ATM
#   0.75 = same strike as 25Δ-put  (|put Δ|=0.25)
#   0.90 = same strike as 10Δ-put  (|put Δ|=0.10)
PILLARS       = [0.10, 0.25, 0.50, 0.75, 0.90]
PILLAR_LABELS = ["10Δ-Call", "25Δ-Call", "ATM", "25Δ-Put", "10Δ-Put"]
PILLAR_COLORS = ["#1565C0", "#42A5F5", "#616161", "#EF5350", "#B71C1C"]

FALLBACK_R = 0.0525   # 5.25% — last-resort if FRED + cache fail
FALLBACK_Q = 0.008    # 0.80% — last-resort if yfinance + cache fail


# ═══════════════════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES CORE
# ═══════════════════════════════════════════════════════════════════════════════

def _d1d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return d1, d1 - sigma * sqrtT


def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, kind: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if kind == "call" else max(K - S, 0)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if kind == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def bs_call_delta(S: float, K: float, T: float, r: float, q: float,
                  sigma: float) -> float:
    """Return call delta; put delta = call delta − exp(−qT)."""
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return np.nan
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    return float(np.exp(-q * T) * norm.cdf(d1))


def implied_vol(price: float, S: float, K: float, T: float,
                r: float, q: float, kind: str = "call") -> float:
    """Brent-method implied vol; returns NaN on any failure."""
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if kind == "call" else max(K - S, 0)
    if price < intrinsic - 0.01:   # below intrinsic (allow tiny rounding)
        return np.nan
    try:
        f = lambda v: bs_price(S, K, T, r, q, v, kind) - price
        lo_val, hi_val = f(IV_LO), f(IV_HI)
        if lo_val * hi_val > 0:
            return np.nan
        return float(brentq(f, IV_LO, IV_HI, xtol=1e-8, maxiter=300))
    except Exception:
        return np.nan


# ═══════════════════════════════════════════════════════════════════════════════
#  RATES & DIVIDEND YIELD — with local JSON cache and hardcoded fallbacks
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_cache(updates: dict) -> None:
    cache = _load_cache()
    cache.update(updates)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def fetch_risk_free_rate() -> float:
    """
    4-week T-bill rate (DTB4WK) from the FRED public CSV endpoint.
    No API key required.  Returns a decimal (e.g. 0.0525 for 5.25%).
    Falls back to the last cached value, then to FALLBACK_R.
    """
    try:
        url  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB4WK"
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        for line in reversed(resp.text.strip().splitlines()[1:]):
            parts = line.split(",")
            if len(parts) == 2 and parts[1].strip() not in (".", ""):
                r    = float(parts[1]) / 100.0
                date_str = parts[0].strip()
                _save_cache({"risk_free": r, "risk_free_date": date_str})
                print(f"  Risk-free rate  (FRED DTB4WK):  {r:.4%}  [{date_str}]")
                return r
    except Exception as exc:
        print(f"  FRED unavailable ({exc})")

    cache = _load_cache()
    if "risk_free" in cache:
        r = float(cache["risk_free"])
        print(f"  Risk-free rate  (cache):        {r:.4%}  [{cache.get('risk_free_date', '?')}]")
        return r

    print(f"  Risk-free rate  (hardcoded):    {FALLBACK_R:.4%}")
    return FALLBACK_R


def fetch_dividend_yield() -> float:
    """
    Trailing annual dividend yield for SPY from yfinance.
    Falls back to last cached value, then to FALLBACK_Q.
    """
    try:
        info = yf.Ticker(TICKER).info
        q = info.get("trailingAnnualDividendYield") or info.get("dividendYield")
        if q and 0 < float(q) < 0.20:
            q = float(q)
            _save_cache({"div_yield": q, "div_yield_date": str(TODAY)})
            print(f"  Dividend yield  (yfinance):     {q:.4%}")
            return q
    except Exception as exc:
        print(f"  Div-yield fetch failed ({exc})")

    cache = _load_cache()
    if "div_yield" in cache:
        q = float(cache["div_yield"])
        print(f"  Dividend yield  (cache):        {q:.4%}  [{cache.get('div_yield_date', '?')}]")
        return q

    print(f"  Dividend yield  (hardcoded):    {FALLBACK_Q:.4%}")
    return FALLBACK_Q


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION CHAIN — fetch, clean, build smile
# ═══════════════════════════════════════════════════════════════════════════════

def _next_trading_day() -> date:
    """
    Return the next exchange business day after TODAY.
    pd.bdate_range handles Mon–Fri; if that day is a market holiday,
    spy.options won't list it and find_1dte_expiry returns None — the
    option chain is the authoritative source of truth on closure.
    """
    return pd.bdate_range(start=TODAY + timedelta(days=1), periods=1)[0].date()


def find_1dte_expiry(spy: yf.Ticker) -> Optional[Tuple[str, int]]:
    """
    Target exactly the next trading day — nothing further.
    On Mon–Thu that is tomorrow (1 cal-day); on Fri it is Monday (3 cal-days).
    Returns (expiry_str, calendar_days) or None if that expiry is not listed
    (market holiday, chain not yet published, etc.).
    """
    target     = _next_trading_day()
    target_str = target.strftime("%Y-%m-%d")
    if target_str in spy.options:
        return target_str, (target - TODAY).days
    return None


def _clean_chain(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Filter option chain rows; add mid and spread_ratio columns."""
    df = df.copy()
    df["kind"] = kind
    df["mid"]  = (df["bid"] + df["ask"]) / 2.0
    mask = (df["bid"] >= MIN_BID) & (df["ask"] > df["bid"]) & (df["mid"] > 0)
    if "openInterest" in df.columns:
        mask &= df["openInterest"].fillna(0) >= MIN_OI
    df = df[mask].copy()
    df["spread_ratio"] = (df["ask"] - df["bid"]) / df["mid"]
    df = df[df["spread_ratio"] < MAX_SPREAD_RATIO]
    keep = [c for c in ("strike", "bid", "ask", "mid", "openInterest", "volume", "kind")
            if c in df.columns]
    return df[keep].reset_index(drop=True)


def build_smile_df(chain_calls: pd.DataFrame, chain_puts: pd.DataFrame,
                   S: float, F: float, T: float,
                   r: float, q: float) -> pd.DataFrame:
    """
    Build the combined OTM vol surface:
      • OTM puts  (K ≤ F) → price with put formula
      • OTM calls (K > F) → price with call formula
    Each row gets: iv, call_delta, log_moneyness = ln(K/F) / (σ√T).
    """
    calls = _clean_chain(chain_calls, "call")
    puts  = _clean_chain(chain_puts,  "put")

    otm_calls = calls[calls["strike"] >  F].copy()
    otm_puts  = puts[puts["strike"]  <= F].copy()

    records = []
    for leg_df, kind in [(otm_puts, "put"), (otm_calls, "call")]:
        for _, row in leg_df.iterrows():
            K  = float(row["strike"])
            iv = implied_vol(float(row["mid"]), S, K, T, r, q, kind)
            if np.isnan(iv) or iv <= 0:
                continue
            delta = bs_call_delta(S, K, T, r, q, iv)
            if np.isnan(delta) or not (0.02 <= delta <= 0.98):
                continue
            lm    = np.log(K / F) / (iv * np.sqrt(T)) if iv > 0 else np.nan
            records.append({
                "strike":        K,
                "kind":          kind,
                "bid":           float(row["bid"]),
                "ask":           float(row["ask"]),
                "mid":           float(row["mid"]),
                "iv":            iv,
                "call_delta":    delta,
                "log_moneyness": lm,
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("strike").reset_index(drop=True)
    # Metadata columns (written to parquet)
    df["as_of"]   = str(TODAY)
    df["expiry"]  = ""
    df["spot"]    = S
    df["forward"] = F
    df["T"]       = T
    df["r"]       = r
    df["q"]       = q
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PILLAR INTERPOLATION
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_pillars(df: pd.DataFrame) -> list[float]:
    """
    Cubic-spline interpolation of IV at each target call-delta in PILLARS.
    Cubic spline respects the convex, curved shape of the vol smile and avoids
    the kinks that linear interpolation introduces between observed strikes.
    Falls back to linear when fewer than 4 points are available (cubic requires
    at least 4 to be well-conditioned).
    Returns NaN for pillars outside the observed delta range.
    """
    pts = (df[["call_delta", "iv"]]
           .dropna()
           .sort_values("call_delta")
           .drop_duplicates("call_delta"))
    pts = pts[(pts["call_delta"] > 0.02) & (pts["call_delta"] < 0.98)]

    if len(pts) < 2:
        return [np.nan] * len(PILLARS)

    kind = "cubic" if len(pts) >= 4 else "linear"
    try:
        fn = interp1d(pts["call_delta"], pts["iv"],
                      kind=kind, bounds_error=False, fill_value=np.nan)
        return [float(fn(d)) for d in PILLARS]
    except Exception:
        return [np.nan] * len(PILLARS)


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_smile(df: pd.DataFrame, pillars: list[float],
               S: float, F: float, expiry: str, days: int) -> None:
    """Two-panel smile: log-moneyness (left) and call-delta with pillars (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"SPY 1DTE Vol Smile  —  {TODAY}   "
        f"(exp {expiry}, {days}d  |  S={S:.2f}  F={F:.2f})",
        fontsize=11, fontweight="bold",
    )

    kind_color = {"call": "#1976D2", "put": "#D32F2F"}

    # ── Panel 1: IV vs standardised log-moneyness ─────────────────────────────
    for kind, grp in df.groupby("kind"):
        sub = grp.dropna(subset=["log_moneyness", "iv"]).sort_values("log_moneyness")
        ax1.scatter(sub["log_moneyness"], sub["iv"] * 100,
                    s=22, alpha=0.7, color=kind_color[kind], label=kind.title(), zorder=3)
        if len(sub) > 2:
            ax1.plot(sub["log_moneyness"], sub["iv"] * 100,
                     lw=0.9, alpha=0.35, color=kind_color[kind])

    ax1.axvline(0, lw=1, ls=":", color="black", alpha=0.5, label="ATM (ln(K/F)=0)")
    ax1.set_xlabel("Standardised Log-Moneyness   ln(K/F) / (σ√T)", fontsize=9)
    ax1.set_ylabel("Implied Volatility (%)", fontsize=9)
    ax1.set_title("Log-Moneyness Smile", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: IV vs call delta with pillar markers ─────────────────────────
    for kind, grp in df.groupby("kind"):
        sub = grp.dropna(subset=["call_delta", "iv"]).sort_values("call_delta")
        ax2.scatter(sub["call_delta"], sub["iv"] * 100,
                    s=22, alpha=0.7, color=kind_color[kind], label=kind.title(), zorder=3)
        if len(sub) > 2:
            ax2.plot(sub["call_delta"], sub["iv"] * 100,
                     lw=0.9, alpha=0.35, color=kind_color[kind])

    for lbl, d_tgt, iv_val, col in zip(PILLAR_LABELS, PILLARS, pillars, PILLAR_COLORS):
        if not np.isnan(iv_val):
            ax2.axvline(d_tgt, lw=0.7, ls="--", color=col, alpha=0.5)
            ax2.scatter([d_tgt], [iv_val * 100], marker="D", s=55,
                        color=col, zorder=5, linewidths=0.6, edgecolors="k")
            ax2.annotate(
                f"{lbl}\n{iv_val:.1%}",
                xy=(d_tgt, iv_val * 100),
                xytext=(5, 5), textcoords="offset points", fontsize=7.5,
            )

    ax2.axvline(0.5, lw=1, ls=":", color="black", alpha=0.5, label="ATM (Δ=0.5)")
    ax2.set_xlabel("Call Delta  Δ", fontsize=9)
    ax2.set_ylabel("Implied Volatility (%)", fontsize=9)
    ax2.set_title("Delta Smile with Pillar Markers", fontsize=10)
    ax2.set_xlim(0.02, 0.98)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = PLOT_DIR / f"smile_{TODAY}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Smile plot    → {path.relative_to(ROOT)}")


def plot_pillar_history() -> None:
    """
    Load every snapshot in SNAP_DIR, recompute pillars, and plot the
    IV time series for each delta pillar.
    """
    files = sorted(SNAP_DIR.glob("*.parquet"))
    if not files:
        return

    rows = []
    for fpath in files:
        try:
            snap    = pd.read_parquet(fpath)
            pillars = interpolate_pillars(snap)
            as_of   = snap["as_of"].iloc[0] if "as_of" in snap.columns else fpath.stem
            rows.append({"date": pd.to_datetime(as_of),
                         **dict(zip(PILLAR_LABELS, pillars))})
        except Exception:
            continue

    if not rows:
        return

    hist = pd.DataFrame(rows).set_index("date").sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("SPY 1DTE Delta-Pillar IV History", fontsize=12, fontweight="bold")

    for lbl, col in zip(PILLAR_LABELS, PILLAR_COLORS):
        if lbl in hist.columns:
            ax.plot(hist.index, hist[lbl] * 100,
                    marker="o", ms=4, lw=1.5, label=lbl, color=col)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.set_ylabel("Implied Volatility (%)", fontsize=9)
    ax.set_xlabel("Date", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = PLOT_DIR / f"pillar_history_{TODAY}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  History plot  → {path.relative_to(ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline() -> None:
    sep = "═" * 62
    print(f"\n{sep}")
    print(f"  SPY 1DTE IV Pipeline  —  {TODAY}")
    print(f"{sep}\n")

    # 1 ── Spot price (multiple fallbacks for yfinance version quirks) ──────────
    spy = yf.Ticker(TICKER)
    S   = None

    # 1a. yf.download — separate code path from .history()
    try:
        dl = yf.download(TICKER, period="5d", progress=False, auto_adjust=True)
        if not dl.empty:
            S = float(dl["Close"].dropna().iloc[-1])
    except Exception:
        pass

    # 1b. fast_info attributes (no history fetch required)
    if S is None:
        fi = spy.fast_info
        for key in ("lastPrice", "previousClose", "regularMarketPreviousClose", "open"):
            try:
                val = fi[key]
                if val and float(val) > 0:
                    S = float(val)
                    break
            except Exception:
                continue

    # 1c. info dict
    if S is None:
        try:
            info = spy.info
            for key in ("regularMarketPrice", "currentPrice", "previousClose", "open"):
                val = info.get(key)
                if val and float(val) > 0:
                    S = float(val)
                    break
        except Exception:
            pass

    if S is None:
        print("  ERROR: Could not fetch SPY price from any source.")
        return
    print(f"  SPY spot: ${S:.2f}\n")

    # 2 ── Rates & yield ───────────────────────────────────────────────────────
    r = fetch_risk_free_rate()
    q = fetch_dividend_yield()

    # 3 ── Nearest 1DTE expiry ─────────────────────────────────────────────────
    result = find_1dte_expiry(spy)
    if result is None:
        nxt = _next_trading_day()
        print(f"\n  No expiry found for next trading day ({nxt}).")
        print(f"  Market may be closed (holiday) or chain not yet published.")
        print(f"  Available expirations: {spy.options[:8]}")
        return
    expiry, days = result
    bday_label = "1 business day" if days == 1 else f"1 business day ({days} cal-days)"
    T = days / 365.0
    F = S * np.exp((r - q) * T)
    print(f"\n  Expiry : {expiry}  ({bday_label},  T = {T:.6f} yr)")
    print(f"  Forward: ${F:.2f}\n")

    # 4 ── Build smile DataFrame ───────────────────────────────────────────────
    chain = spy.option_chain(expiry)
    df    = build_smile_df(chain.calls, chain.puts, S, F, T, r, q)
    if df.empty:
        print("  ERROR: No usable strikes after filtering.  Check expiry liquidity.")
        return
    df["expiry"] = expiry

    n_calls = int((df["kind"] == "call").sum())
    n_puts  = int((df["kind"] == "put").sum())
    print(f"  OTM strikes retained — calls: {n_calls}   puts: {n_puts}")

    # 5 ── Parquet snapshot ────────────────────────────────────────────────────
    snap_path = SNAP_DIR / f"{TODAY}.parquet"
    df.to_parquet(snap_path, index=False)
    print(f"  Snapshot saved  → {snap_path.relative_to(ROOT)}")

    # 6 ── Delta pillars ───────────────────────────────────────────────────────
    pillars = interpolate_pillars(df)
    print()
    print("  ┌──────────────────────────────────────────┐")
    print("  │  Pillar          Call Δ       IV          │")
    print("  ├──────────────────────────────────────────┤")
    for lbl, d_tgt, iv_val in zip(PILLAR_LABELS, PILLARS, pillars):
        iv_str = f"{iv_val:.3%}" if not np.isnan(iv_val) else "   N/A   "
        print(f"  │  {lbl:<14s}  {d_tgt:.2f}      {iv_str:>9s}   │")
    print("  └──────────────────────────────────────────┘")

    # 7 ── Plots ───────────────────────────────────────────────────────────────
    print()
    plot_smile(df, pillars, S, F, expiry, days)
    plot_pillar_history()

    print(f"\n  All done.  Outputs in: {DATA_DIR.relative_to(ROOT)}/\n")


if __name__ == "__main__":
    run_pipeline()
