"""
Script: garch_historical_backtesting.py

Purpose:
    Performs a historical rolling backtesting study for GARCH-based VaR and
    Expected Shortfall forecasts, combined with an empirically calibrated
    PELVE level.

Thesis reference:
    Chapter 5, Tables 5.1 and 5.2:
        Historical GARCH backtesting results for 3-month US Treasury yield data.


Description:
    The historical forecast performance is evaluated using
        - a Kupiec-type hit test at the PELVE-implied VaR level,
        - a Ziegel-type joint VaR/ES backtest at the target level beta.

    Empirical critical values are obtained by an iid bootstrap based on the
    historical loss sample.
Usage:
    Set the parameters in the Run section, choose
    garch_dist = "t" or "normal", and execute the script directly.

"""


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch import arch_model
from scipy.optimize import brentq
from scipy.stats import chi2, norm, t
from pathlib import Path


# ============================================================
# Data
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV = DATA_DIR / "3MY.csv"
def load_losses_3m(csv_path, begin="1999-01-04", end="2024-12-31"):
    px = pd.read_csv(csv_path, index_col=0, parse_dates=True).squeeze()
    px = px.loc[begin:end].astype(float)

    changes = px.diff().dropna().astype(float)
    losses = (-changes).astype(float)
    losses.name = "loss"

    return losses, changes


# ============================================================
# Empirical VaR/ES
# ============================================================

def empirical_var(x, p):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0,1).")

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
        raise ValueError("p must lie in (0,1).")

    x = np.sort(x)
    n = x.size

    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return float(x[k - 1:].mean())


# ============================================================
# Test statistics
# ============================================================

def kupiec_stat(hit_count, n, p):
    phat = hit_count / n

    eps = 1e-12
    phat = np.clip(phat, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    ll_null = (n - hit_count) * np.log(1 - p) + hit_count * np.log(p)
    ll_alt = (n - hit_count) * np.log(1 - phat) + hit_count * np.log(phat)

    return float(-2.0 * (ll_null - ll_alt))


def ziegel_stat_from_series(losses_realized, q_series, e_series, beta):
    losses_realized = np.asarray(losses_realized, dtype=float).reshape(-1)
    q_series = np.asarray(q_series, dtype=float).reshape(-1)
    e_series = np.asarray(e_series, dtype=float).reshape(-1)

    n = len(losses_realized)

    if not (len(q_series) == n and len(e_series) == n):
        raise ValueError("losses_realized, q_series, e_series must have same length.")

    V = np.empty((n, 2), dtype=float)

    for i in range(n):
        x = losses_realized[i]
        q_val = q_series[i]
        e_val = e_series[i]

        I = 1.0 if x > q_val else 0.0

        v1 = (1.0 - beta) - I
        v2 = q_val - e_val - (1.0 / (1.0 - beta)) * I * (q_val - x)

        V[i, 0] = v1
        V[i, 1] = v2

    vbar = V.mean(axis=0)
    S = np.cov(V.T, bias=True)
    S_inv = np.linalg.pinv(S)

    return float(n * vbar.T @ S_inv @ vbar)


# ============================================================
# GARCH forecasts
# ============================================================

def fit_garch_forecast(
    train_losses,
    dist="t",
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    rescale=False,
    suppress_warnings=True,
):
    x = np.asarray(train_losses, dtype=float).reshape(-1)

    if x.size < max(p, q) + 10:
        raise ValueError("Training window too short for GARCH fit.")

    with warnings.catch_warnings():
        if suppress_warnings:
            warnings.simplefilter("ignore")

        am = arch_model(
            x,
            mean=mean,
            vol=vol,
            p=p,
            q=q,
            dist=dist,
            rescale=rescale,
        )

        res = am.fit(disp="off", show_warning=False)

    fcast = res.forecast(horizon=1, reindex=False)

    mu = float(fcast.mean.iloc[-1, 0])
    sigma = float(np.sqrt(fcast.variance.iloc[-1, 0]))
    params = res.params

    out = {
        "mu": mu,
        "sigma": sigma,
        "dist": dist,
        "converged": bool(res.convergence_flag == 0),
        "params": params.to_dict(),
    }

    if dist == "t":
        out["nu"] = float(params["nu"])

    return out


def forecast_var(fd, alpha):
    mu = fd["mu"]
    sigma = fd["sigma"]

    if fd["dist"] == "normal":
        z = norm.ppf(1.0 - alpha)
        return float(-(mu + sigma * z))

    if fd["dist"] == "t":
        nu = fd["nu"]
        z = t.ppf(1.0 - alpha, df=nu)
        sf = sigma / np.sqrt(nu / (nu - 2))
        return float(-(mu + sf * z))

    raise ValueError("dist must be 'normal' or 't'")


def forecast_es(fd, alpha):
    mu = fd["mu"]
    sigma = fd["sigma"]

    if fd["dist"] == "normal":
        z = norm.ppf(1.0 - alpha)
        return float(-mu + sigma * norm.pdf(z) / (1.0 - alpha))

    if fd["dist"] == "t":
        nu = fd["nu"]
        z = t.ppf(1.0 - alpha, df=nu)
        pdf = t.pdf(z, df=nu)
        sf = sigma / np.sqrt(nu / (nu - 2))

        es_ret = mu - sf * ((nu + z ** 2) / (nu - 1)) * pdf / (1.0 - alpha)
        return float(-es_ret)

    raise ValueError("dist must be 'normal' or 't'")



# ============================================================
# Empirical rolling PELVE series
# ============================================================

def pelve_from_empirical_window(window_losses, eps_for_c):
    window_losses = np.asarray(window_losses, dtype=float).reshape(-1)

    var_target = empirical_var(window_losses, 1.0 - eps_for_c)

    def f(c):
        level = 1.0 - c * eps_for_c
        level = min(max(level, 1e-12), 1.0 - 1e-12)
        return empirical_es(window_losses, level) - var_target

    try:
        return float(brentq(f, 1.0, 1.0 / eps_for_c - 1e-8))
    except ValueError:
        return np.nan


def rolling_empirical_pelve_series(
    losses_array,
    window,
    eps_for_c,
    progress_every=None,
):
    losses_array = np.asarray(losses_array, dtype=float).reshape(-1)
    n = len(losses_array)

    if n <= window:
        raise ValueError("Need more observations than window length.")

    c_list = []

    for t_idx in range(window, n + 1):
        w = losses_array[t_idx - window:t_idx]

        c_t = pelve_from_empirical_window(w, eps_for_c)
        c_list.append(c_t)

        if progress_every is not None:
            step = t_idx - window + 1
            total = n - window + 1

            if step % progress_every == 0 or step == total:
                print(f"[Empirical PELVE series {step}/{total}] c={c_t:.6f}")

    c_series = np.asarray(c_list, dtype=float)

    return {
        "c_series": c_series,
        "c_const": float(np.nanmedian(c_series)),
    }


# ============================================================
# Empirical grid selection
# ============================================================

def summarize_grid_row(eps_for_c, c_median, beta):
    beta_hat = 1.0 - c_median * eps_for_c
    residual = beta_hat - beta

    return {
        "eps": float(eps_for_c),
        "c_median": float(c_median),
        "beta_hat": float(beta_hat),
        "residual": float(residual),
        "abs_residual": float(abs(residual)),
    }


def select_eps_star_empirical_grid(
    losses_array,
    window,
    beta,
    eps_grid,
    progress_every=None,
):
    rows = []
    pelve_infos = {}

    for j, eps_for_c in enumerate(eps_grid, start=1):
        pelve_info = rolling_empirical_pelve_series(
            losses_array=losses_array,
            window=window,
            eps_for_c=float(eps_for_c),
            progress_every=None,
        )

        c_median = pelve_info["c_const"]

        if not np.isfinite(c_median):
            continue

        row = summarize_grid_row(eps_for_c, c_median, beta)

        rows.append(row)
        pelve_infos[float(eps_for_c)] = pelve_info

        if progress_every is not None:
            if j % progress_every == 0 or j == len(eps_grid):
                print(
                    f"[Empirical eps-grid {j}/{len(eps_grid)}] "
                    f"eps={eps_for_c:.6f} | "
                    f"c_med={c_median:.6f} | "
                    f"beta_hat={row['beta_hat']:.6f} | "
                    f"resid={row['residual']:.4e}"
                )

    if len(rows) == 0:
        raise RuntimeError("No valid empirical eps* found on grid.")

    grid = pd.DataFrame(rows)

    best_idx = int(grid["abs_residual"].idxmin())
    best = grid.loc[best_idx].to_dict()

    eps_star = float(best["eps"])

    return {
        "grid": grid,
        "eps_star": eps_star,
        "alpha_star": float(1.0 - eps_star),
        "c_star": float(best["c_median"]),
        "beta_hat_star": float(best["beta_hat"]),
        "residual_star": float(best["residual"]),
        "pelve_info_star": pelve_infos[eps_star],
        "pelve_infos": pelve_infos,
    }


# ============================================================
# Rolling GARCH forecasts
# ============================================================

def rolling_garch_forecast_series(
    losses_array,
    window,
    garch_dist="t",
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    rescale=False,
    progress_every=None,
):
    losses_array = np.asarray(losses_array, dtype=float).reshape(-1)
    n = len(losses_array)

    if n <= window:
        raise ValueError("Need more observations than window length.")

    forecast_list = []
    converged_list = []

    for t_idx in range(window, n):
        w = losses_array[t_idx - window:t_idx]

        fd = fit_garch_forecast(
            w,
            dist=garch_dist,
            mean=mean,
            vol=vol,
            p=p,
            q=q,
            rescale=rescale,
        )

        forecast_list.append(fd)
        converged_list.append(fd["converged"])

        if progress_every is not None:
            step = t_idx - window + 1
            total = n - window

            if step % progress_every == 0 or step == total:
                print(f"[GARCH fit {step}/{total}] converged={fd['converged']}")

    converged_flags = np.asarray(converged_list, dtype=bool)

    return {
        "forecast_list": forecast_list,
        "converged_flags": converged_flags,
        "converged_share": float(np.mean(converged_flags)),
    }


# ============================================================
# Observed rolling GARCH backtest
# ============================================================

def rolling_backtest_garch_from_forecasts(
    losses_array,
    window,
    beta,
    alpha_target,
    forecast_list,
    hit_prob_reference=None,
    progress_every=None,
):
    losses_array = np.asarray(losses_array, dtype=float).reshape(-1)
    n = len(losses_array)

    if n <= window:
        raise ValueError("Need more observations than window length.")

    n_test = n - window

    if len(forecast_list) < n_test:
        raise ValueError("forecast_list is too short for the requested backtest.")

    var_pelve_list = []
    q_list = []
    e_list = []
    realized_list = []
    hit_list = []
    converged_list = []

    for i in range(n_test):
        t_idx = window + i
        x_next = losses_array[t_idx]

        fd = forecast_list[i]

        q_t = forecast_var(fd, beta)
        e_t = forecast_es(fd, beta)

        var_pelve_t = forecast_var(fd, alpha_target)
        hit_t = 1 if x_next > var_pelve_t else 0

        q_list.append(q_t)
        e_list.append(e_t)
        var_pelve_list.append(var_pelve_t)
        realized_list.append(x_next)
        hit_list.append(hit_t)
        converged_list.append(fd["converged"])

        if progress_every is not None:
            step = i + 1

            if step % progress_every == 0 or step == n_test:
                print(
                    f"[Observed rolling {step}/{n_test}] "
                    f"alpha_target={alpha_target:.6f}"
                )

    realized = np.asarray(realized_list, dtype=float)
    var_pelve_series = np.asarray(var_pelve_list, dtype=float)
    q_series = np.asarray(q_list, dtype=float)
    e_series = np.asarray(e_list, dtype=float)
    hits = np.asarray(hit_list, dtype=int)
    converged = np.asarray(converged_list, dtype=bool)

    hit_count = int(hits.sum())
    hit_rate = float(hit_count / n_test)

    if hit_prob_reference is None:
        hit_prob = float(1.0 - alpha_target)
    else:
        hit_prob = float(hit_prob_reference)

    LR = kupiec_stat(hit_count, n_test, hit_prob)
    T = ziegel_stat_from_series(realized, q_series, e_series, beta)

    return {
        "n_test": n_test,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "hit_prob": hit_prob,
        "alpha_target": float(alpha_target),
        "realized_losses": realized,
        "var_pelve_series": var_pelve_series,
        "q_series": q_series,
        "e_series": e_series,
        "hits": hits,
        "kupiec_stat": LR,
        "ziegel_stat": T,
        "converged_share": float(np.mean(converged)),
        "converged_flags": converged,
    }


# ============================================================
# Empirical rolling backtest for bootstrap calibration
# ============================================================

def rolling_backtest_empirical_from_losses(
    losses_array,
    window,
    beta,
    alpha_target,
):
    losses_array = np.asarray(losses_array, dtype=float).reshape(-1)
    n = len(losses_array)

    if n <= window:
        raise ValueError("Need more observations than window length.")

    var_pelve_list = []
    q_list = []
    e_list = []
    realized_list = []
    hit_list = []

    for t_idx in range(window, n):
        w = losses_array[t_idx - window:t_idx]
        x_next = losses_array[t_idx]

        q_t = empirical_var(w, beta)
        e_t = empirical_es(w, beta)

        var_pelve_t = empirical_var(w, alpha_target)
        hit_t = 1 if x_next > var_pelve_t else 0

        q_list.append(q_t)
        e_list.append(e_t)
        var_pelve_list.append(var_pelve_t)
        realized_list.append(x_next)
        hit_list.append(hit_t)

    realized = np.asarray(realized_list, dtype=float)
    var_pelve_series = np.asarray(var_pelve_list, dtype=float)
    q_series = np.asarray(q_list, dtype=float)
    e_series = np.asarray(e_list, dtype=float)
    hits = np.asarray(hit_list, dtype=int)

    n_test = len(realized)
    hit_count = int(hits.sum())
    hit_rate = float(hit_count / n_test)
    hit_prob = float(1.0 - alpha_target)

    LR = kupiec_stat(hit_count, n_test, hit_prob)
    T = ziegel_stat_from_series(realized, q_series, e_series, beta)

    return {
        "n_test": n_test,
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "hit_prob": hit_prob,
        "realized_losses": realized,
        "var_pelve_series": var_pelve_series,
        "q_series": q_series,
        "e_series": e_series,
        "hits": hits,
        "kupiec_stat": LR,
        "ziegel_stat": T,
    }


# ============================================================
# Empirical bootstrap calibration
# ============================================================

def bootstrap_resample_iid(x, rng):
    x = np.asarray(x, dtype=float).reshape(-1)
    idx = rng.integers(0, len(x), size=len(x))

    return x[idx]


def bootstrap_calibration_empirical_fixed_alpha(
    historical_losses,
    window,
    beta,
    alpha_target_emp,
    alpha_test=0.05,
    mc_cal=1000,
    seed=12345,
    progress_every=100,
):
    rng = np.random.default_rng(seed)
    hist = np.asarray(historical_losses, dtype=float).reshape(-1)

    kupiec_stats = []
    ziegel_stats = []
    hit_counts = []
    hit_rates = []

    for b in range(1, mc_cal + 1):
        losses_b = bootstrap_resample_iid(hist, rng)

        res_b = rolling_backtest_empirical_from_losses(
            losses_array=losses_b,
            window=window,
            beta=beta,
            alpha_target=alpha_target_emp,
        )

        kupiec_stats.append(res_b["kupiec_stat"])
        ziegel_stats.append(res_b["ziegel_stat"])
        hit_counts.append(res_b["hit_count"])
        hit_rates.append(res_b["hit_rate"])

        if progress_every is not None and (b % progress_every == 0 or b == mc_cal):
            print(
                f"[Empirical bootstrap {b}/{mc_cal}] "
                f"alpha_emp={alpha_target_emp:.6f} | "
                f"Kupiec stat={res_b['kupiec_stat']:.4f} | "
                f"Ziegel stat={res_b['ziegel_stat']:.4f}"
            )

    kupiec_stats = np.asarray(kupiec_stats, dtype=float)
    ziegel_stats = np.asarray(ziegel_stats, dtype=float)
    hit_counts = np.asarray(hit_counts, dtype=int)
    hit_rates = np.asarray(hit_rates, dtype=float)

    return {
        "kupiec_stats": kupiec_stats,
        "ziegel_stats": ziegel_stats,
        "hit_counts": hit_counts,
        "hit_rates": hit_rates,
        "kupiec_crit": float(np.quantile(kupiec_stats, 1.0 - alpha_test)),
        "ziegel_crit": float(np.quantile(ziegel_stats, 1.0 - alpha_test)),
        "hit_count_median": float(np.median(hit_counts)),
        "hit_rate_median": float(np.median(hit_rates)),
        "alpha_target_emp": float(alpha_target_emp),
        "hit_prob_emp": float(1.0 - alpha_target_emp),
    }



def make_pelve_dates(losses_index, window, n_c):
    return pd.to_datetime(losses_index[window - 1: window - 1 + n_c])


def plot_pelve_series(
    dates,
    c_series,
    title,
    ylabel="PELVE",
    show_median=True,
    show_e=True,
    ylim=None,
):
    c_series = np.asarray(c_series, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, c_series, linewidth=1, label="PELVE")

    if show_median:
        med = np.nanmedian(c_series)
        ax.axhline(med, linestyle=":", linewidth=1, label=f"median = {med:.3f}")

    if show_e:
        ax.axhline(np.e, linestyle="--", linewidth=1, label="e = 2.718")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.set_title(title)
    ax.legend(frameon=False)

    fig.tight_layout()
    plt.show()


# ============================================================
# Main routine
# ============================================================

def run_historical_garch_pelve_grid_backtest_with_empirical_bootstrap(
    csv_path,
    begin="2009-10-01",
    end="2015-11-15",
    window=500,
    beta=0.95,
    eps_grid_min=0.01,
    eps_grid_max=0.03,
    eps_grid_size=41,
    alpha_test=0.05,
    garch_dist="t",
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    rescale=False,
    mc_cal=1000,
    seed=12345,
    progress_every=100,
    grid_progress_every=None,
    plot_pelve=True,
    show_e=True,
    ylim_pelve=None,
):
    losses, changes = load_losses_3m(csv_path=csv_path, begin=begin, end=end)
    losses = losses.dropna().sort_index()
    x = losses.to_numpy(dtype=float)

    if len(x) <= window:
        raise ValueError("Chosen period is too short for the given window length.")

    eps_grid = np.linspace(eps_grid_min, eps_grid_max, eps_grid_size)

    grid_emp = select_eps_star_empirical_grid(
        losses_array=x,
        window=window,
        beta=beta,
        eps_grid=eps_grid,
        progress_every=grid_progress_every,
    )

    eps_star_emp = grid_emp["eps_star"]
    alpha_star_emp = grid_emp["alpha_star"]

    garch_roll = rolling_garch_forecast_series(
        losses_array=x,
        window=window,
        garch_dist=garch_dist,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        rescale=rescale,
        progress_every=progress_every,
    )

    # --------------------------------------------------------
    # Observed model-based backtest
    # Hits use GARCH VaR at alpha_star_emp.
    # Expected hit probability is eps_star_emp.
    # --------------------------------------------------------
    observed = rolling_backtest_garch_from_forecasts(
        losses_array=x,
        window=window,
        beta=beta,
        alpha_target=alpha_star_emp,
        forecast_list=garch_roll["forecast_list"],
        hit_prob_reference=eps_star_emp,
        progress_every=progress_every,
    )

    kupiec_p_chi2 = float(1.0 - chi2.cdf(observed["kupiec_stat"], df=1))
    ziegel_p_chi2 = float(1.0 - chi2.cdf(observed["ziegel_stat"], df=2))

    boot = bootstrap_calibration_empirical_fixed_alpha(
        historical_losses=x,
        window=window,
        beta=beta,
        alpha_target_emp=alpha_star_emp,
        alpha_test=alpha_test,
        mc_cal=mc_cal,
        seed=seed,
        progress_every=progress_every,
    )

    kupiec_reject_boot = observed["kupiec_stat"] > boot["kupiec_crit"]
    ziegel_reject_boot = observed["ziegel_stat"] > boot["ziegel_crit"]

    kupiec_reject_chi2 = kupiec_p_chi2 < alpha_test
    ziegel_reject_chi2 = ziegel_p_chi2 < alpha_test

    if plot_pelve:
        emp_c = grid_emp["pelve_info_star"]["c_series"]
        dates_emp = make_pelve_dates(losses.index, window, len(emp_c))

        plot_pelve_series(
            dates=dates_emp,
            c_series=emp_c,
            title=(
                f"Empirical rolling PELVE at eps*={eps_star_emp:.5f}, "
                f"beta={beta}, W={window}"
            ),
            show_e=show_e,
            ylim=ylim_pelve,
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n================ HISTORICAL GARCH RESULT "
          "(EMPIRICAL PELVE LEVEL, EMPIRICAL BOOTSTRAP) ================\n")

    print("Period                         :", begin, "to", end)
    print("Window                         :", window)
    print("beta                           :", beta)
    print("eps grid                       :", eps_grid_min, "to", eps_grid_max,
          f"({eps_grid_size} points)")
    print("alpha_test                     :", alpha_test)
    print("GARCH dist                     :", garch_dist)
    print("mean / vol                     :", mean, "/", vol)
    print("bootstrap samples              :", mc_cal)
    print()

    print("---------------- EMPIRICAL PELVE REFERENCE ----------------")
    print("empirical eps*                 :", eps_star_emp)
    print("empirical alpha* used          :", alpha_star_emp)
    print("empirical median c(eps*)       :", grid_emp["c_star"])
    print("empirical beta_hat             :", grid_emp["beta_hat_star"])
    print("empirical residual             :", grid_emp["residual_star"])
    print()

    print("---------------- OBSERVED GARCH BACKTEST ----------------")
    print("total loss obs                 :", len(x))
    print("test sample size               :", observed["n_test"])
    print("hit count                      :", observed["hit_count"])
    print("hit rate                       :", observed["hit_rate"])
    print("expected hit prob (emp eps*)   :", observed["hit_prob"])
    print("expected hit count             :", observed["n_test"] * observed["hit_prob"])
    print("GARCH VaR alpha used           :", observed["alpha_target"])
    print("GARCH fit conv. share          :", garch_roll["converged_share"])
    print("Backtest conv. share           :", observed["converged_share"])
    print()

    print("Kupiec stat                    :", observed["kupiec_stat"])
    print("Kupiec chi2 p-val              :", kupiec_p_chi2)
    print("Kupiec empirical crit          :", boot["kupiec_crit"])
    print("Kupiec reject chi2             :", kupiec_reject_chi2)
    print("Kupiec reject empirical        :", kupiec_reject_boot)
    print()

    print("Ziegel stat                    :", observed["ziegel_stat"])
    print("Ziegel chi2 p-val              :", ziegel_p_chi2)
    print("Ziegel empirical crit          :", boot["ziegel_crit"])
    print("Ziegel reject chi2             :", ziegel_reject_chi2)
    print("Ziegel reject empirical        :", ziegel_reject_boot)
    print()

    print("---------------- BOOTSTRAP SUMMARY ----------------")
    print("bootstrap fixed alpha_emp      :", boot["alpha_target_emp"])
    print("bootstrap hit prob emp         :", boot["hit_prob_emp"])
    print("bootstrap median hit count     :", boot["hit_count_median"])
    print("bootstrap median hit rate      :", boot["hit_rate_median"])
    print()

    return {
        "losses": losses,
        "changes": changes,
        "eps_grid": eps_grid,
        "grid_emp": grid_emp,
        "eps_star_emp": eps_star_emp,
        "alpha_star_emp": alpha_star_emp,
        "pelve_emp_star": grid_emp["pelve_info_star"],
        "garch_roll": garch_roll,
        "observed": observed,
        "bootstrap": boot,
        "kupiec_p_chi2": kupiec_p_chi2,
        "ziegel_p_chi2": ziegel_p_chi2,
        "kupiec_reject_chi2": kupiec_reject_chi2,
        "ziegel_reject_chi2": ziegel_reject_chi2,
        "kupiec_reject_boot": kupiec_reject_boot,
        "ziegel_reject_boot": ziegel_reject_boot,
    }


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    #CSV = r"/Users/annakonrad/Desktop/Masterarbeit/3MY.csv"

    run_historical_garch_pelve_grid_backtest_with_empirical_bootstrap(
        csv_path=DEFAULT_CSV,
        begin="2009-10-01", #"2003-11-04" #"2009-10-01"
        end="2015-11-15", #"2007-06-15" #"2015-11-15"
        window=500,
        beta=0.924,
        eps_grid_min=0.01,
        eps_grid_max=0.05,
        eps_grid_size=41,
        alpha_test=0.05,
        garch_dist="t",   # "t", "normal"
        mean="Constant", # "AR", "Constant"
        vol="GARCH",
        p=1,
        q=1,
        rescale=False,
        mc_cal=1000,
        seed=12345,
        progress_every=10,
        grid_progress_every=1,
        plot_pelve=False,
        show_e=False,
        ylim_pelve=None,
    )