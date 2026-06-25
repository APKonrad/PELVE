"""
Script: historical_test_1.py

Purpose:
    Computes rolling empirical PELVE estimates for changes in 3-month U.S.
    Treasury yields and provides plots for historical subperiods.

Thesis reference:
    Chapter 5, Figure 5.7:
        Examples of rolling empirical PELVE estimates for changes in 3-month
        U.S. Treasury yields.

Description:
    The script loads 3-month U.S. Treasury yield data from a local CSV file,
    computes daily yield changes, and converts them into losses by changing
    the sign.

    Based on the resulting loss series, rolling empirical VaR, ES and PELVE
    estimates are computed for a chosen window size and tail probability epsilon.

    The script additionally provides diagnostic output for selected rolling
    windows and can plot

        yield changes,
        losses,
        rolling PELVE,
        rolling VaR,
        rolling ES.


Usage:
    Place the data file

        3MY.csv

    in the folder

        data/

    next to this script. Then choose the historical period, rolling window size
    and tail probability by setting

        BEGIN
        END
        W
        eps

    in the main block.

Notes:
    Figure 5.7 is obtained by running the script separately for selected
    subperiods and tail probabilities, in particular

        eps = 0.05,
        eps = 0.10.

    The rolling window size used in the historical analysis is

        W = 500.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1) Data: 3M Treasury yields -> yield changes -> losses
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
def load_losses_3m(
    csv_path=DATA_DIR / "3MY.csv",
    begin="1999-01-04",
    end="2024-12-31",
):
    px = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze()
    px = px.loc[begin:end].astype(float)

    # yield changes (not log returns)
    changes = px.diff().dropna().astype(float)

    # loss convention
    losses = (-changes).astype(float)
    losses.name = "loss"
    return losses, changes



# ============================================================
# 2) Empirical VaR / ES
# ============================================================
def empirical_var(x, p):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1).")

    x = np.sort(x)
    n = x.size
    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return float(x[k - 1])


def empirical_es(x, p):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1).")

    x = np.sort(x)
    n = x.size
    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return float(x[k - 1:].mean())


# ============================================================
# 3) PELVE
# ============================================================
def pelve(window_losses, eps=0.05):
    window_losses = np.asarray(window_losses, dtype=float).reshape(-1)
    window_losses = window_losses[np.isfinite(window_losses)]

    if window_losses.size == 0:
        return np.nan
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must lie in (0, 1).")

    p_var = 1.0 - eps
    var_1meps = empirical_var(window_losses, p_var)
    es_1meps = empirical_es(window_losses, p_var)

    if not np.isfinite(var_1meps) or not np.isfinite(es_1meps):
        return np.nan
    if es_1meps < var_1meps:
        return np.nan

    def f(c):
        level = 1.0 - c * eps
        level = min(max(level, 1e-12), 1.0 - 1e-12)
        return empirical_es(window_losses, level) - var_1meps

    c_low = 1.0
    c_high = 1.0 / eps - 1e-12

    f_low = f(c_low)
    f_high = f(c_high)

    if (not np.isfinite(f_low)) or (not np.isfinite(f_high)):
        return np.nan
    if f_low * f_high > 0:
        return np.nan

    return float(brentq(f, c_low, c_high))


# ============================================================
# 4) Rolling series
# ============================================================
def rolling_var_es_pelve(
    losses: pd.Series,
    window: int = 500,
    eps: float = 0.05
):
    if window < 2:
        raise ValueError("window must be at least 2.")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must lie in (0, 1).")

    losses = losses.dropna().sort_index()
    x = losses.to_numpy(dtype=float)
    idx = losses.index

    p = 1.0 - eps
    vars_, ess_, pelves_, dates = [], [], [], []

    for i in range(window - 1, len(x)):
        wdw = x[i - window + 1:i + 1]

        vars_.append(empirical_var(wdw, p))
        ess_.append(empirical_es(wdw, p))
        pelves_.append(pelve(wdw, eps=eps))
        dates.append(idx[i])

    return pd.DataFrame(
        {"VaR": vars_, "ES": ess_, "PELVE": pelves_},
        index=pd.Index(dates, name="Date")
    )


# ============================================================
# 5) Diagnostics
# ============================================================
def debug_windows(losses: pd.Series, window=500, p=0.95, n_windows=6):
    losses = losses.dropna().sort_index()
    x = losses.to_numpy(dtype=float)
    idx = losses.index

    if len(x) < window:
        print("Not enough data for the requested window length.")
        return

    ts = np.linspace(window - 1, len(x) - 1, n_windows, dtype=int)

    for t in ts:
        w = x[t - window + 1:t + 1]
        w_sorted = np.sort(w)

        var_p = empirical_var(w_sorted, p)
        es_p = empirical_es(w_sorted, p)

        vc = pd.Series(w_sorted).value_counts()
        unique_share = vc.size / len(w_sorted)

        print("\n--- window ending:", idx[t], "---")
        print(f"VaR({p:.3f}) = {var_p:.6f}")
        print(f"ES({p:.3f})  = {es_p:.6f}")
        print(f"unique share = {unique_share:.4f}")
        print("top 6 value counts:")
        print(vc.head(6).to_string())


# ============================================================
# 6) Plotting
# ============================================================
def plot_four(
    changes: pd.Series,
    losses: pd.Series,
    results: pd.DataFrame,
    title_prefix="",
    show_median=True,
    ylim_pelve=None
):
    changes = changes.dropna().copy()
    losses = losses.dropna().copy()
    results = results.copy()

    changes.index = pd.to_datetime(changes.index)
    losses.index = pd.to_datetime(losses.index)
    results.index = pd.to_datetime(results.index)

    # Changes
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(changes.index, changes.values, linewidth=0.8, label="Yield changes")
    ax.axhline(0.0, linestyle=":", linewidth=1)
    ax.set_ylabel("Change")
    ax.set_xlabel("Year")
    ax.set_title(f"{title_prefix} Yield changes")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # Optional: losses instead of changes
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(losses.index, losses.values, linewidth=0.8, label="Losses = -changes")
    ax.axhline(0.0, linestyle=":", linewidth=1)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Year")
    ax.set_title(f"{title_prefix} Losses")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # PELVE
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results.index, results["PELVE"], linewidth=1, label="PELVE")
    #ax.set_ylim(1, 7)
    if show_median:
        med = results["PELVE"].median(skipna=True)
        ax.axhline(med, linestyle=":", linewidth=1, label=f"median = {med:.1f}")
    if ylim_pelve is not None:
        ax.set_ylim(*ylim_pelve)
    ax.set_ylabel("PELVE")
    ax.set_xlabel("Year")
    #ax.set_title(f"{title_prefix} PELVE")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # VaR
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(results.index, results["VaR"], linewidth=1, label="VaR")
    if show_median:
        med = results["VaR"].median(skipna=True)
        ax.axhline(med, linestyle=":", linewidth=1, label=f"median = {med:.4f}")
    ax.set_ylabel("VaR")
    ax.set_xlabel("Year")
    ax.set_title(f"{title_prefix} VaR")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()

    # ES
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(results.index, results["ES"], linewidth=1, label="ES")
    if show_median:
        med = results["ES"].median(skipna=True)
        ax.axhline(med, linestyle=":", linewidth=1, label=f"median = {med:.4f}")
    ax.set_ylabel("ES")
    ax.set_xlabel("Year")
    ax.set_title(f"{title_prefix} ES")
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


# ============================================================
# 7) Main
# ============================================================
if __name__ == "__main__":
    CSV = r"/Users/annakonrad/Desktop/Masterarbeit/3MY.csv"
    BEGIN, END = "2007-10-01", "2021-11-15"
    W = 500
    eps = 0.1

    losses, changes = load_losses_3m(
        csv_path=CSV,
        begin=BEGIN,
        end=END,
    )

    debug_windows(losses, window=W, p=1.0 - eps, n_windows=6)

    res = rolling_var_es_pelve(losses, window=W, eps=eps)

    title = f"3M Treasury (changes->losses), eps={eps}, W={W}:"
    plot_four(
        changes=changes,
        losses=losses,
        results=res,
        title_prefix=title,
        show_median=True,
        ylim_pelve=None
    )