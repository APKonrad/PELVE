"""
Script: empirical_power_grid.py

Purpose:
    Studies the empirical power of the Monte-Carlo calibrated PELVE-based
    Kupiec test and the Ziegel ES backtest under misspecified VaR and ES
    levels.

Thesis reference:
    Chapter 5, Figure 5.5(b):
        Empirical power comparison of the Monte-Carlo calibrated PELVE and
        Ziegel tests for different false VaR--ES tuples.

Description:
    Based on this empirical calibration, Monte-Carlo critical values are
    obtained for both tests. The tests are then applied to independently
    simulated test samples under deliberately misspecified VaR and ES levels.

    The false tuples determine the misspecified VaR level used for the PELVE
    test and the misspecified ES level used for the Ziegel test.

Notes:
    The final settings for Figure 5.5(b) are

        M = 2500,
        repetitions = 10,
        train_size = 200000,
        n_days = 5000,
        beta_true = 0.95,
        nu_dgp = 7,
        alpha_test = 0.05,

    The false tuples considered in the final power comparison are

        (0.970, 0.900),
        (0.986, 0.960),
        (0.979, 0.940),
        (0.984, 0.955),
        (0.980, 0.945).

    The script may take a long time to run because the empirical calibration
    and the Monte-Carlo critical values are recomputed in each repetition. For
    a quick test run, reduce M, repetitions, train_size or the number of false
    tuples. Since no fixed random seed is set, different runs will generally
    produce slightly different rejection rates, while the overall power pattern
    should remain stable.
"""

import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq


# ============================================================
# Simulation
# ============================================================

def simulate_losses_student_t(n_days, nu, sigma, mu):
    sf = sigma / np.sqrt(nu / (nu - 2))
    z = np.random.standard_t(df=nu, size=n_days)

    losses = -(mu + sf * z)
    return losses


# ============================================================
# empirical VaR / ES
# ============================================================

def empirical_var(x, p):
    x = np.sort(np.asarray(x))
    n = len(x)

    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return x[k - 1]


def empirical_es(x, p):
    x = np.sort(np.asarray(x))
    n = len(x)

    k = int(np.ceil(n * p))
    k = np.clip(k, 1, n)

    return x[k - 1:].mean()


# ============================================================
# PELVE
# ============================================================

def pelve_empirical(losses, eps=0.05):
    p = 1 - eps
    var_target = empirical_var(losses, p)

    def f(c):
        level = 1 - c * eps
        es_level = empirical_es(losses, level)
        return es_level - var_target

    try:
        return brentq(f, 1, 1 / eps - 1e-12)
    except ValueError:
        return np.nan


# ============================================================
# Grid search for eps*
# ============================================================

def find_eps_star_by_grid(losses, beta_target, eps_lo=0.01, eps_hi=0.03, n_grid=41):
    eps_grid = np.linspace(eps_lo, eps_hi, n_grid)

    rows = []
    best = None
    best_abs_resid = np.inf

    for eps in eps_grid:
        c_val = pelve_empirical(losses, eps)

        if not np.isfinite(c_val):
            rows.append({
                "eps": eps,
                "c": np.nan,
                "beta_achieved": np.nan,
                "residual": np.nan,
                "abs_residual": np.nan,
            })
            continue

        beta_achieved = 1.0 - c_val * eps
        resid = beta_achieved - beta_target
        abs_resid = abs(resid)

        row = {
            "eps": eps,
            "c": c_val,
            "beta_achieved": beta_achieved,
            "residual": resid,
            "abs_residual": abs_resid,
        }
        rows.append(row)

        if abs_resid < best_abs_resid:
            best_abs_resid = abs_resid
            best = row

    if best is None:
        raise ValueError("No finite grid point found for eps* calibration.")

    return {
        "eps_star": best["eps"],
        "c_star": best["c"],
        "beta_achieved": best["beta_achieved"],
        "residual": best["residual"],
        "alpha_true": 1.0 - best["eps"],
        "grid_rows": rows,
    }


# ============================================================
# Kupiec statistic
# ============================================================

def kupiec_stat(hit_count, n, p):
    phat = hit_count / n

    eps = 1e-12
    phat = np.clip(phat, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    ll_null = (n - hit_count) * np.log(1 - p) + hit_count * np.log(p)
    ll_alt = (n - hit_count) * np.log(1 - phat) + hit_count * np.log(phat)

    return -2 * (ll_null - ll_alt)


# ============================================================
# Ziegel statistic
# ============================================================

def ziegel_stat(losses, q, e, beta):
    V = []

    for x in losses:
        I = 1.0 if x > q else 0.0

        v1 = (1 - beta) - I
        v2 = q - e - (1 / (1 - beta)) * I * (q - x)

        V.append([v1, v2])

    V = np.asarray(V)

    n = V.shape[0]
    vbar = V.mean(axis=0)

    S = np.cov(V.T, bias=True)
    S_inv = np.linalg.pinv(S)

    return n * vbar.T @ S_inv @ vbar


# ============================================================
# MC calibration
# ============================================================

def calibrate_kupiec_mc(n_sim, n_days, eps_target, var_true):
    stats = []

    for i in range(n_sim):
        losses = simulate_losses_student_t(
            n_days, nu_dgp, sigma, mu
        )

        hits = losses > var_true
        stat = kupiec_stat(hits.sum(), n_days, eps_target)

        stats.append(stat)

    return np.quantile(stats, 0.95)


def calibrate_ziegel_mc(n_sim, n_days, q, e, beta):
    stats = []

    for i in range(n_sim):
        losses = simulate_losses_student_t(
            n_days, nu_dgp, sigma, mu
        )

        stat = ziegel_stat(losses, q, e, beta)

        stats.append(stat)

    return np.quantile(stats, 0.95)


# ============================================================
# SETTINGS
# ============================================================

M = 2500
repetitions = 10

train_size = 200000
n_days = 5000

nu_dgp = 7
sigma = 0.01
mu = 0.0

beta_true = 0.95
alpha_test = 0.05

# grid settings for eps calibration
eps_grid_lo = 0.01
eps_grid_hi = 0.03
eps_grid_n = 81


false_tuples = [
    (0.9700, 0.900),
    (0.9865, 0.960),
    (0.9790, 0.940),
    (0.9840, 0.955),
    (0.9805, 0.945)
]


# ============================================================
# POWER EXPERIMENT
# ============================================================

for alpha_false, beta_false in false_tuples:

    print("\n================================")
    print(f"FALSE TUPLE: alpha={alpha_false}, beta={beta_false}")
    print("================================")

    for r in range(repetitions):


        train_losses = simulate_losses_student_t(
            train_size, nu_dgp, sigma, mu
        )


        grid_info = find_eps_star_by_grid(
            losses=train_losses,
            beta_target=beta_true,
            eps_lo=eps_grid_lo,
            eps_hi=eps_grid_hi,
            n_grid=eps_grid_n,
        )

        eps_target = grid_info["eps_star"]
        c_const = grid_info["c_star"]
        alpha_true = grid_info["alpha_true"]

        var_true = empirical_var(train_losses, alpha_true)

        q_true = empirical_var(train_losses, beta_true)
        e_true = empirical_es(train_losses, beta_true)


        kupiec_MC_crit = calibrate_kupiec_mc(
            3000, n_days, eps_target, var_true
        )

        ziegel_MC_crit = calibrate_ziegel_mc(
            3000, n_days, q_true, e_true, beta_true
        )

        rej_pelve = 0
        rej_ziegel = 0

        var_false = empirical_var(train_losses, alpha_false)
        e_false = empirical_es(train_losses, beta_false)

        for m in range(M):
            losses = simulate_losses_student_t(
                n_days, nu_dgp, sigma, mu
            )

            # Kupiec
            hits = losses > var_false
            LR = kupiec_stat(hits.sum(), n_days, eps_target)
            r_kupiec = LR > kupiec_MC_crit

            # Ziegel
            T = ziegel_stat(losses, q_true, e_false, beta_true)
            r_ziegel = T > ziegel_MC_crit

            rej_pelve += r_kupiec
            rej_ziegel += r_ziegel

        print(
            f"rep {r+1:02d}  "
            f"eps*: {eps_target:.6f}   "
            f"c(eps*): {c_const:.6f}   "
            f"beta_hat: {grid_info['beta_achieved']:.6f}   "
            f"resid: {grid_info['residual']:.4e}   "
            f"PELVE MC: {rej_pelve/M:.4f}   "
            f"ZIEGEL MC: {rej_ziegel/M:.4f}"
        )