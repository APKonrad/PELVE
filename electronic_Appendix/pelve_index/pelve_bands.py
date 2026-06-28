"""
Script: pelve_bands.py

Purpose:
    Computes rolling empirical PELVE estimates and constructs pointwise
    confidence bands using an i.i.d. bootstrap procedure.

Thesis reference:
    Chapter 4:
        Figure 4.1: Empirical PELVE for S&P 500 log-losses,
                    epsilon = 0.05, rolling window size = 500.
        Figure 4.2: Empirical PELVE for 3-month U.S. Treasury yield changes,
                    epsilon = 0.1, rolling window size = 500.

Description:
    The script computes rolling empirical VaR, ES and PELVE estimates. For
    selected rolling windows, bootstrap samples are drawn from the observations
    in the window and used to construct pointwise confidence bands for the
    PELVE estimates.

    The active data-loading function determines whether the script is run for
    S&P 500 log-losses or for 3-month Treasury yield changes. The alternative
    data-loading block is included in the script but may be commented out
    depending on which figure is reproduced.

Usage:
    The parameters BEGIN, END, WINDOW, EPS, B, ALPHA and STEP control the
    sample period, rolling window size, PELVE level, number of bootstrap
    repetitions, confidence band level and bootstrap frequency.

IMPORTANT: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    The script uses a relative data path. It expects a folder named

        data

    in the same directory as this script. The required CSV file must be stored
    in this folder.

    Example structure:

        electronic_appendix/
        ├── Pelve_Bands.py
        └── data/
            ├── sp500_prices.csv
            └── 3MY.csv

Notes:
    Final settings:
        Figure 4.1:
            file = sp500_prices.csv,
            EPS = 0.05,
            WINDOW = 500.

        Figure 4.2:
            file = 3MY.csv,
            EPS = 0.1,
            WINDOW = 500.

    In both cases, the bootstrap settings used for the final figures are

        B = 1000,
        ALPHA = 0.10,
        STEP = 5.

    With B = 1000 and the full sample period, the script may take a long time to
    run. For a quicker test run, one may either reduce the sample period by
    changing BEGIN and END or set B = 100.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path

# ============================================================
# 1) Data
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_log_losses(begin="1999-01-01", end="2024-10-09"):
    csv_path = DATA_DIR / "sp500_prices.csv"

    px = pd.read_csv(
        csv_path,
        index_col=0,
        parse_dates=True
    ).squeeze()

    px = px.loc[begin:end]
    r = np.log(px).diff().dropna()
    losses = (-r).astype(float)

    return losses

# def load_log_losses(
#     csv_path= DATA_DIR / "3MY.csv",
#     begin="1999-01-04",
#     end="2024-12-31",
# ):
#     px = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze()
#     px = px.loc[begin:end].astype(float)
#
#     changes = px.diff().dropna().astype(float)
#
#     losses = (-changes).astype(float)
#     losses.name = "loss"
#
#     return losses

# ============================================================
# 2) Empirical VaR / ES / PELVE
# ============================================================
def empirical_var(x, p):
    x = np.asarray(x, float).reshape(-1)
    x = np.sort(x)
    n = x.size
    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)
    return float(x[k - 1])


def empirical_es(x, p):
    x = np.asarray(x, float).reshape(-1)
    x = np.sort(x)
    n = x.size
    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)
    return float(x[k - 1:].mean())


def pelve(window_losses, var_1meps, es_1meps, eps=0.05):
    if es_1meps < var_1meps:
        return np.nan

    def f(c):
        level = 1.0 - c * eps
        return empirical_es(window_losses, level) - var_1meps

    c_low = 1.0
    c_high = 1.0 / eps - 1e-12

    f_low = f(c_low)
    f_high = f(c_high)

    if f_low * f_high > 0:
        return np.nan

    return float(brentq(f, c_low, c_high))


# ============================================================
# 3) IID bootstrap helper
# ============================================================
def iid_bootstrap_sample(window_losses, rng):
    w = np.asarray(window_losses, float).reshape(-1)
    n = w.size
    return rng.choice(w, size=n, replace=True)


# ============================================================
# 4) Bootstrap for one window: PELVE
# ============================================================
def bootstrap_pelve(
    window_losses,
    eps=0.05,
    B=300,
    alpha=0.10,
    seed=None,
    min_valid=50
):

    rng = np.random.default_rng(seed)
    w = np.asarray(window_losses, float).reshape(-1)
    p = 1.0 - eps

    var_hat = empirical_var(w, p)
    es_hat = empirical_es(w, p)
    pelve_hat = pelve(w, var_hat, es_hat, eps=eps)

    pelve_stars = []

    for _ in range(B):
        w_star = iid_bootstrap_sample(w, rng)

        var_star = empirical_var(w_star, p)
        es_star = empirical_es(w_star, p)
        pelve_star = pelve(w_star, var_star, es_star, eps=eps)

        if np.isfinite(pelve_star):
            pelve_stars.append(pelve_star)

    pelve_stars = np.asarray(pelve_stars, float)

    if pelve_stars.size < min_valid:
        return {
            "VaR_hat": var_hat,
            "ES_hat": es_hat,
            "PELVE_hat": pelve_hat,
            "PELVE_lo": np.nan,
            "PELVE_hi": np.nan,
            "n_valid": int(pelve_stars.size),
        }

    pelve_lo = float(np.quantile(pelve_stars, alpha / 2))
    pelve_hi = float(np.quantile(pelve_stars, 1 - alpha / 2))

    return {
        "VaR_hat": var_hat,
        "ES_hat": es_hat,
        "PELVE_hat": pelve_hat,
        "PELVE_lo": pelve_lo,
        "PELVE_hi": pelve_hi,
        "n_valid": int(pelve_stars.size),
    }


# ============================================================
# 5) Rolling empirical PELVE + IID bootstrap bands
# ============================================================
def rolling_pelve_iid_bands(
    data,
    window=500,
    eps=0.05,
    B=300,
    alpha=0.10,
    step=1,
    seed=123,
    min_valid=50
):
    stream = data.to_numpy(dtype=float)
    indx = data.index
    p = 1.0 - eps

    dates_all = []
    var_emp = []
    es_emp = []
    pelve_emp = []

    band_dates = []
    pelve_lo = []
    pelve_hi = []
    n_valid_list = []

    for i in range(window - 1, len(stream)):
        print(i)
        wdw = stream[i - (window - 1): i + 1]


        var_i = empirical_var(wdw, p)
        es_i = empirical_es(wdw, p)
        pelve_i = pelve(wdw, var_i, es_i, eps=eps)

        dates_all.append(indx[i])
        var_emp.append(var_i)
        es_emp.append(es_i)
        pelve_emp.append(pelve_i)


        if ((i - (window - 1)) % step) != 0:
            continue

        out = bootstrap_pelve(
            wdw,
            eps=eps,
            B=B,
            alpha=alpha,
            seed=seed + i,
            min_valid=min_valid
        )

        band_dates.append(indx[i])
        pelve_lo.append(out["PELVE_lo"])
        pelve_hi.append(out["PELVE_hi"])
        n_valid_list.append(out["n_valid"])

    df_emp = pd.DataFrame(
        {
            "VaR": var_emp,
            "ES": es_emp,
            "PELVE": pelve_emp,
        },
        index=pd.Index(dates_all, name="Date")
    )

    df_bands = pd.DataFrame(
        {
            "PELVE_lo": pelve_lo,
            "PELVE_hi": pelve_hi,
            "n_valid": n_valid_list,
        },
        index=pd.Index(band_dates, name="Date")
    )

    df_bands = df_bands.reindex(df_emp.index)

    if step > 1:
        df_bands[["PELVE_lo", "PELVE_hi"]] = (
            df_bands[["PELVE_lo", "PELVE_hi"]].interpolate(method="time")
        )

    return df_emp.join(df_bands)


# ============================================================
# 6) Plot
# ============================================================
def plot_pelve(df, title="Rolling empirical PELVE with IID bootstrap bands", show_e=True):
    d = df.copy()
    d.index = pd.to_datetime(d.index)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(d.index, d["PELVE"], linewidth=1.5, label="Empirical PELVE", color="#1f77b4")
    ax.plot(d.index, d["PELVE_lo"], linewidth=1.5, label="Lower band", color="#aec7e8")
    ax.plot(d.index, d["PELVE_hi"], linewidth=1.5, label="Upper band", color="#aec7e8")

    if show_e:
        ax.axhline(np.e, linestyle=":", linewidth=1, label=f"e = {np.e:.3f}")

    ax.set_xlabel("Date")
    ax.set_ylabel("PELVE")
    #ax.set_title(title)
    #ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()


# ============================================================
# 7) Run
# ============================================================
BEGIN = "1999-04-01"
END = "2020-10-09"

WINDOW = 500
EPS = 0.05
B = 1000
ALPHA = 0.10
STEP = 5
SEED = 123
MIN_VALID = B * 0.05

losses = load_log_losses(begin=BEGIN, end=END)

df_pelve = rolling_pelve_iid_bands(
    data=losses,
    window=WINDOW,
    eps=EPS,
    B=B,
    alpha=ALPHA,
    step=STEP,
    seed=SEED,
    min_valid=MIN_VALID
)

print(df_pelve.head(20))
print(df_pelve[["PELVE", "PELVE_lo", "PELVE_hi", "n_valid"]].tail(20))

plot_pelve(
    df_pelve,
    title="S&P 500: 5% rolling empirical PELVE with IID bootstrap bands",
    show_e=False
)