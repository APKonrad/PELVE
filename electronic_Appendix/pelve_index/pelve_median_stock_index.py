"""
Script: pelve_median_stock_index.py

Purpose:
    Computes rolling empirical PELVE estimates for selected stock indices or
    individual stocks using price data downloaded from Yahoo Finance.

Thesis reference:
    Chapter 4:
        PELVE comparison plots for S&P 500, DAX, NVIDIA and Siemens AG.
        Table 4.1: PELVE medians for selected DAX constituents and the DAX index.

Description:
    The script downloads adjusted price data for a selected Yahoo Finance ticker,
    converts prices to log-losses, and computes rolling empirical VaR, ES and
    PELVE estimates. The resulting PELVE time series is plotted together with
    optional horizontal reference lines for the sample median and Euler's number e.

    The sample median of the rolling PELVE series is printed and can be used for
    the values reported in the Table of Chapter 4.

Usage:
    Select the asset or index by setting

        TICKER
        INDEX_NAME

    in the configuration block. The parameters BEGIN_DATE, END_DATE,
    ROLLING_WINDOW and EPS control the sample period, rolling window size and
    PELVE level.

Notes:
    The script requires an internet connection because the data are downloaded
    from Yahoo Finance via yfinance.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

# "^GSPC"   -> S&P 500
# "^IXIC"   -> NASDAQ Composite
# "^NDX"    -> NASDAQ-100
# "^DJI"    -> Dow Jones Industrial Average
# "^RUT"    -> Russell 2000
# "^GDAXI"  -> DAX
# "^STOXX50E" -> EURO STOXX 50
# "^FTSE"   -> FTSE 100
# "^N225"   -> Nikkei 225
# "^HSI"    -> Hang Seng

# "SIE.DE"  -> Siemens AG
# "NVDA"    -> NVIDIA Corporation

# DAX constituents used for Table in Chapter 4:
# "ADS.DE"   -> Adidas
# "HEN3.DE"  -> Henkel
# "MUV2.DE"  -> Münchener Rück
# "SIE.DE"   -> Siemens
# "RWE.DE"   -> RWE
# "BAYN.DE"  -> Bayer
# "SAP.DE"   -> SAP
# "DBK.DE"   -> Deutsche Bank
# "DTE.DE"   -> Deutsche Telekom
# "VOW3.DE"  -> Volkswagen
# "ALV.DE"   -> Allianz
# "BMW.DE"   -> BMW
# "BAS.DE"   -> BASF


TICKER = "^GDAXI" #"^GSPC"
INDEX_NAME = "DAX" #S&P 500"

BEGIN_DATE = "1999-01-01"
END_DATE = "2024-10-09"

ROLLING_WINDOW = 500
EPS = 0.05


# ============================================================
# DATA
# ============================================================

def load_log_losses(
    ticker: str = TICKER,
    begin: str = BEGIN_DATE,
    end: str = END_DATE,
) -> pd.Series:
    df = yf.download(
        ticker,
        start=begin,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"Keine Daten von Yahoo Finance für Ticker {ticker} gefunden."
        )

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            px = df[("Close", ticker)].dropna()
        elif ("Adj Close", ticker) in df.columns:
            px = df[("Adj Close", ticker)].dropna()
        else:
            px = df.iloc[:, 0].dropna()
    else:
        if "Close" in df.columns:
            px = df["Close"].dropna()
        elif "Adj Close" in df.columns:
            px = df["Adj Close"].dropna()
        else:
            raise ValueError(
                f"Keine passende Close-Spalte für Ticker {ticker} gefunden."
            )

    r = np.log(px).diff().dropna()

    losses = (-r).astype(float)
    losses.name = "loss"

    return losses


# ============================================================
# EMPIRICAL VAR / ES
# ============================================================

def empirical_var(x, p: float) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.sort(x)

    n = x.size
    if n == 0:
        return np.nan

    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return float(x[k - 1])


def empirical_es(x, p: float) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.sort(x)

    n = x.size
    if n == 0:
        return np.nan

    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return float(x[k - 1:].mean())


# ============================================================
# PELVE
# ============================================================

def pelve(
    window_losses,
    var_1meps: float,
    es_1meps: float,
    eps: float = EPS,
) -> float:

    if np.isnan(var_1meps) or np.isnan(es_1meps):
        return np.nan

    if es_1meps < var_1meps:
        return np.nan

    def f(c):
        level = 1.0 - c * eps
        return empirical_es(window_losses, level) - var_1meps

    c_low = 1.0
    c_high = 1.0 / eps - 1e-12

    try:
        f_low = f(c_low)
        f_high = f(c_high)
    except Exception:
        return np.nan

    if np.isnan(f_low) or np.isnan(f_high):
        return np.nan
    if f_low * f_high > 0:
        return np.nan

    try:
        return float(brentq(f, c_low, c_high))
    except Exception:
        return np.nan


# ============================================================
# ROLLING ESTIMATION
# ============================================================

def rolling_var_es(
    ticker: str = TICKER,
    begin: str = BEGIN_DATE,
    end: str = END_DATE,
    window: int = ROLLING_WINDOW,
    eps: float = EPS,
) -> pd.DataFrame:
    losses = load_log_losses(ticker=ticker, begin=begin, end=end)
    stream = losses.values
    indx = losses.index

    p = 1.0 - eps

    vars_, ess, pelves, dates = [], [], [], []

    for i in range(window - 1, len(stream)):
        wdw = stream[i - (window - 1): i + 1]

        var_i = empirical_var(wdw, p)
        es_i = empirical_es(wdw, p)
        pel_i = pelve(wdw, var_i, es_i, eps=eps)

        vars_.append(var_i)
        ess.append(es_i)
        pelves.append(pel_i)
        dates.append(indx[i])

    return pd.DataFrame(
        {
            "VaR": vars_,
            "ES": ess,
            "PELVE": pelves,
        },
        index=pd.Index(dates, name="Date"),
    )


# ============================================================
# PLOT
# ============================================================

def plot_pelve(
    df: pd.DataFrame,
    title: str = "Raw PELVE (rolling window)",
    show_e: bool = False, show_median: bool = False
) -> None:
    d = df.copy()
    d.index = pd.to_datetime(d.index)
    d["PELVE"] = pd.to_numeric(d["PELVE"], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(d.index, d["PELVE"], linewidth=1, label="Raw PELVE")

    if show_median:
        median_pelve = d["PELVE"].median()
        ax.axhline(
            median_pelve,
            linestyle="--",
            linewidth=1,
            label=f"median = {median_pelve:.3f}")

    if show_e:
        ax.axhline(np.e, linestyle=":", linewidth=1, label=f"e = {np.e:.3f}")

    ax.set_ylabel("PELVE")
    ax.set_xlabel("Year")
    plt.ylim(2, 4)
    #ax.set_title(title)
    #ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    res = rolling_var_es(
        ticker=TICKER,
        begin=BEGIN_DATE,
        end=END_DATE,
        window=ROLLING_WINDOW,
        eps=EPS,
    )

    print(res.head())
    print(res.tail())
    print("PELVE median:", res["PELVE"].median(skipna=True))

    plot_pelve(
        res,
        title=f"{INDEX_NAME}: {int(EPS * 100)}% PELVE ({ROLLING_WINDOW}-day rolling)",
        show_e=True, show_median=True
    )