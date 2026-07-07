# SPY 1DTE Implied Volatility Pipeline

Pulls the SPY option chain, computes Black-Scholes implied volatility and
call delta for every OTM strike, saves a daily parquet snapshot, interpolates
10/25/ATM/25/10-delta pillars, and produces two plots per run.

---

## Project layout

```
1DTE-SPX-OPTION/
├── spy_1dte_vol_pipeline.py   ← main pipeline
├── requirements.txt
├── run_pipeline_mac.command   ← double-click on macOS
├── run_pipeline_windows.bat   ← double-click on Windows
└── data/                      ← auto-created on first run
    ├── snapshots/             ← YYYY-MM-DD.parquet (one per run-date)
    ├── plots/                 ← smile_<date>.png, pillar_history_<date>.png
    └── rates_cache.json       ← last-good FRED rate + SPY div yield
```

---

## Quick start

### macOS
1. Open Terminal, `cd` into this folder.
2. `pip install -r requirements.txt`
3. `python3 spy_1dte_vol_pipeline.py`

Or double-click **run_pipeline_mac.command** in Finder (right-click → Open
on first use to approve the script).

### Windows
Double-click **run_pipeline_windows.bat**.  Python must be installed and
on `PATH`.

---

## What the pipeline does

| Step | Detail |
|------|--------|
| **Spot** | Last close for SPY via `yfinance` |
| **Risk-free rate** | 4-week T-bill (DTB4WK) from FRED's public CSV — no API key needed; falls back to cached value, then 5.25% |
| **Dividend yield** | SPY trailing annual yield via `yfinance`; falls back to cached value, then 1.30% |
| **Expiry** | Nearest expiry 1–5 calendar days out (skips same-day 0DTE) |
| **Forward** | `F = S · exp((r − q) · T)` |
| **OTM filtering** | Calls for K > F, puts for K ≤ F; drops strikes with zero bid, open interest < 10, or bid-ask spread > 75% of mid |
| **Implied vol** | Brent-method solve on Black-Scholes price formula; bounds 0.01% – 3000% annualised |
| **Call delta** | `exp(−qT) · N(d1)` |
| **Log-moneyness** | `ln(K/F) / (σ√T)` — standardised so a standard normal maps to ±1σ |
| **Parquet snapshot** | `data/snapshots/YYYY-MM-DD.parquet` |
| **Pillar interpolation** | Linear interp of (call_delta, iv) at Δ = 0.10, 0.25, 0.50, 0.75, 0.90 |
| **Plots** | Saved as 150 dpi PNG to `data/plots/` |

---

## Delta pillar convention

Pillars are expressed in **call-delta** space (0 = deep OTM call, 1 = deep ITM call):

| Label | Call Δ | Interpretation |
|-------|--------|----------------|
| 10Δ-Call | 0.10 | OTM call wing |
| 25Δ-Call | 0.25 | Near-OTM call |
| ATM | 0.50 | At-the-money |
| 25Δ-Put | 0.75 | Same strike as put with \|Δ\|=0.25 |
| 10Δ-Put | 0.90 | Same strike as put with \|Δ\|=0.10 |

For equity options, put-call parity gives `call_Δ + |put_Δ| ≈ 1`, so
the 25-delta put has the same strike as a 75-delta call.

---

## Parquet schema

| Column | Type | Description |
|--------|------|-------------|
| `strike` | float | Option strike |
| `kind` | str | `"call"` or `"put"` |
| `bid`, `ask`, `mid` | float | Market prices |
| `iv` | float | Annualised BS implied vol (decimal) |
| `call_delta` | float | BS call delta |
| `log_moneyness` | float | Standardised log-moneyness |
| `as_of` | str | Run date (YYYY-MM-DD) |
| `expiry` | str | Option expiry (YYYY-MM-DD) |
| `spot` | float | SPY last close |
| `forward` | float | Risk-neutral 1-day forward |
| `T` | float | Time to expiry (years) |
| `r` | float | Risk-free rate (decimal) |
| `q` | float | Dividend yield (decimal) |

---

## Requirements

- Python 3.10+
- See `requirements.txt` — all packages are available on PyPI
- Internet access for live data (FRED + Yahoo Finance); offline runs fall back to cached rates

---

## Notes

- **Weekend / holiday runs**: if no 1–5 day expiry is found, the pipeline
  prints the available expirations and exits cleanly.
- **Parquet storage**: uses `pyarrow`.  Install `fastparquet` instead if you
  prefer — both work with `pd.to_parquet`.
- **Smile interpolation**: linear interpolation is used for simplicity.
  For production use, consider SVI or SABR parameterisations.
- **IV bounds**: the solver searches 0.01% – 3000% annualised.  Deep OTM
  1DTE options can carry extreme implied vols; filter as needed.
