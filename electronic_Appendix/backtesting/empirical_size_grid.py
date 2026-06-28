"""
Script: empirical_size_grid.py

Purpose:
    Studies the empirical size of the PELVE-based Kupiec test and the Ziegel
    ES backtest when the underlying Student-t model is correctly specified.

Thesis reference:
    Chapter 5, Figure 5.4:
        Reject rate of asymptotic and Monte Carlo calibrated PELVE and Ziegel backtests reported over 10 repetitions.
    Chapter 5, Figure 5.5(a):
        Empirical size comparison of the Monte-Carlo calibrated PELVE and
        Ziegel tests for different numbers of simulation iterations.

Description:
    The script performs a nested Monte Carlo experiment. In each outer
    repetition, a training sample of Student-t losses is generated and used to
    estimate the PELVE-implied VaR level corresponding to the target ES level
    beta = 0.95. Based on this empirical calibration, the script computes
    rejection rates on independently simulated test samples.

    The rejection rates are calculated for

        PELVE/Kupiec test with chi-square critical value,
        PELVE/Kupiec test with Monte-Carlo calibrated critical value,
        Ziegel test with chi-square critical value,
        Ziegel test with Monte-Carlo calibrated critical value.

    For Figure 5.4(a), the Monte-Carlo calibrated versions are used to compare
    the empirical size of both tests against the nominal level.

Notes:
    The final settings for Figure 5.4(a) are

        J = 20,
        M in {100, 1000, 2500, 10000},
        train_size = 20000,
        test_size = 10000,
        beta = 0.95,
        nu_dgp = 7,
        alpha_test = 0.05,
        mc_cal = 3000.

    The script may take a long time to run, especially for large values of M
    and mc_cal. For a quick test run, reduce J, M or mc_cal. Since the empirical
    calibration is repeated in each outer iteration, the resulting rejection
    rates contain both Monte Carlo variation and additional estimation
    uncertainty from the training sample. Nevertheless, the rejection rates
    should concentrate around the nominal level as M increases.
"""



import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq


# ============================================================
# 1) Simulation
# ============================================================

def simulate_losses_student_t(n_days, nu, sigma, mu, seed):
    rng = np.random.default_rng(seed)

    sf = sigma / np.sqrt(nu / (nu - 2))
    z = rng.standard_t(df=nu, size=n_days)

    losses = -(mu + sf * z)
    return losses


# ============================================================
# 2) Empirical VaR / ES
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


# ============================================================
# 3) Empirical PELVE
# ============================================================

def pelve_empirical(losses, eps=0.05):
    p = 1 - eps

    var_1meps = empirical_var(losses, p)
    es_1meps = empirical_es(losses, p)

    if es_1meps < var_1meps:
        return np.nan

    def f(c):
        level = 1 - c * eps
        return empirical_es(losses, level) - var_1meps

    c_low = 1.0
    c_high = 1.0 / eps - 1e-12

    try:
        return brentq(f, c_low, c_high)
    except ValueError:
        return np.nan


# ============================================================
# 3b) Grid search for eps*
# ============================================================

def find_eps_star_by_grid(
    losses,
    beta,
    eps_lo=0.01,
    eps_hi=0.03,
    n_grid=41,
):
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
        resid = beta_achieved - beta
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
        raise ValueError("No finite grid point found for eps* search.")

    return {
        "eps_star": best["eps"],
        "c_star": best["c"],
        "beta_achieved": best["beta_achieved"],
        "residual": best["residual"],
        "alpha_target": 1.0 - best["eps"],
        "grid_rows": rows,
    }


# ============================================================
# 4) Kupiec UC Test (statistic)
# ============================================================

def kupiec_stat(hit_count, n, p):
    phat = hit_count / n

    eps = 1e-12
    phat = np.clip(phat, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    ll_null = (n - hit_count) * np.log(1 - p) + hit_count * np.log(p)
    ll_alt = (n - hit_count) * np.log(1 - phat) + hit_count * np.log(phat)

    LR = -2 * (ll_null - ll_alt)
    return LR


def kupiec_uc_test(hit_count, n, p, alpha):
    LR = kupiec_stat(hit_count, n, p)

    crit = chi2.ppf(1 - alpha, df=1)
    p_value = 1 - chi2.cdf(LR, df=1)

    return LR, p_value, LR > crit


# ============================================================
# 5) Ziegel statistic
# ============================================================

def ziegel_stat(losses, q, e, nu_level):
    V = []

    for x in losses:
        I = 1.0 if x > q else 0.0

        v1 = (1 - nu_level) - I
        v2 = q - e - (1 / (1 - nu_level)) * I * (q - x)

        V.append([v1, v2])

    V = np.asarray(V)

    n = V.shape[0]
    vbar = V.mean(axis=0)
    S = np.cov(V.T, bias=True)
    S_inv = np.linalg.pinv(S)

    T = n * vbar.T @ S_inv @ vbar
    return T


def ziegel_test(losses, q, e, nu_level, alpha):
    T = ziegel_stat(losses, q, e, nu_level)

    crit = chi2.ppf(1 - alpha, df=2)
    p_value = 1 - chi2.cdf(T, df=2)

    return T, p_value, T > crit


# ============================================================
# 6) MC Calibration
# ============================================================

def calibrate_kupiec_mc(
    mc_samples,
    sample_size,
    nu_dgp,
    sigma,
    mu,
    var_level,
    eps_target,
    alpha,
    seed
):
    stats = []

    for i in range(mc_samples):
        losses = simulate_losses_student_t(
            sample_size, nu_dgp, sigma, mu, seed + i
        )

        hits = losses > var_level
        stat = kupiec_stat(hits.sum(), sample_size, eps_target)

        stats.append(stat)

    return np.quantile(stats, 1 - alpha)


def calibrate_ziegel_mc(
    mc_samples,
    sample_size,
    nu_dgp,
    sigma,
    mu,
    q,
    e,
    beta,
    alpha,
    seed
):
    stats = []

    for i in range(mc_samples):
        losses = simulate_losses_student_t(
            sample_size, nu_dgp, sigma, mu, seed + i
        )

        stat = ziegel_stat(losses, q, e, beta)
        stats.append(stat)

    return np.quantile(stats, 1 - alpha)


# ============================================================
# 7) Output for grid search
# ============================================================

def print_grid_result(grid_info, beta):
    print("\n================ GRID SEARCH FOR EPS* =================")
    print(f"Target beta                 : {beta:.8f}")
    print(f"Chosen eps*                 : {grid_info['eps_star']:.10f}")
    print(f"Estimated c(eps*)           : {grid_info['c_star']:.10f}")
    print(f"Achieved beta               : {grid_info['beta_achieved']:.10f}")
    print(f"Residual                    : {grid_info['residual']:.4e}")
    print(f"Final alpha_target          : {grid_info['alpha_target']:.10f}")
    print("=======================================================")

    print("\n---------------- GRID TABLE ----------------")
    print(f"{'eps':>12} {'c(eps)':>14} {'beta_hat':>14} {'residual':>14} {'|resid|':>14}")
    print("-" * 74)

    for row in grid_info["grid_rows"]:
        if np.isfinite(row["c"]):
            print(
                f"{row['eps']:>12.6f} "
                f"{row['c']:>14.8f} "
                f"{row['beta_achieved']:>14.8f} "
                f"{row['residual']:>14.4e} "
                f"{row['abs_residual']:>14.4e}"
            )
        else:
            print(
                f"{row['eps']:>12.6f} "
                f"{'nan':>14} "
                f"{'nan':>14} "
                f"{'nan':>14} "
                f"{'nan':>14}"
            )

    print("-------------------------------------------------------\n")


# ============================================================
# 8) Nested Monte Carlo
# ============================================================

def nested_mc_reject_rates(
    J=10,
    M=500,

    train_size=2500,
    test_size=2500,

    nu_dgp=7,
    sigma=0.01,
    mu=0.0,

    beta=0.95,

    alpha_test=0.05,
    mc_cal=2000,

    base_seed=12345,

    eps_grid_lo=0.01,
    eps_grid_hi=0.03,
    eps_grid_n=41,
):
    for j in range(1, J + 1):


        train_losses = simulate_losses_student_t(
            train_size, nu_dgp, sigma, mu, base_seed + j
        )

        # determine eps* by grid search
        grid_info = find_eps_star_by_grid(
            losses=train_losses,
            beta=beta,
            eps_lo=eps_grid_lo,
            eps_hi=eps_grid_hi,
            n_grid=eps_grid_n,
        )

        eps_target = grid_info["eps_star"]
        c_const = grid_info["c_star"]
        alpha_target = grid_info["alpha_target"]


        var_level = empirical_var(train_losses, alpha_target)

        q = empirical_var(train_losses, beta)
        e = empirical_es(train_losses, beta)

        kupiec_crit_mc = calibrate_kupiec_mc(
            mc_cal,
            test_size,
            nu_dgp,
            sigma,
            mu,
            var_level,
            eps_target,
            alpha_test,
            base_seed + 10000 * j,
        )

        ziegel_crit_mc = calibrate_ziegel_mc(
            mc_cal,
            test_size,
            nu_dgp,
            sigma,
            mu,
            q,
            e,
            beta,
            alpha_test,
            base_seed + 20000 * j,
        )

        rej_pelve = 0
        rej_pelve_mc = 0
        rej_ziegel = 0
        rej_ziegel_mc = 0

        for m in range(1, M + 1):
            test_losses = simulate_losses_student_t(
                test_size,
                nu_dgp,
                sigma,
                mu,
                base_seed + 100000 * j + m,
            )

            hits = test_losses > var_level
            hit_count = hits.sum()

            LR, _, r_kupiec = kupiec_uc_test(
                hit_count, test_size, eps_target, alpha_test
            )

            r_kupiec_mc = LR > kupiec_crit_mc

            T, _, r_ziegel = ziegel_test(
                test_losses, q, e, beta, alpha_test
            )

            r_ziegel_mc = T > ziegel_crit_mc

            rej_pelve += r_kupiec
            rej_pelve_mc += r_kupiec_mc
            rej_ziegel += r_ziegel
            rej_ziegel_mc += r_ziegel_mc

        # ----------------------------------------------------
        # verbose output
        # ----------------------------------------------------
        print("\n================ RESULT =================")
        print("Training sample            :", j)
        print("train_size                 :", train_size)
        print("test_size                  :", test_size)
        print("beta target                :", beta)
        print("grid range                 :", (eps_grid_lo, eps_grid_hi))
        print("grid points                :", eps_grid_n)
        print("estimated c(eps*)          :", c_const)
        print("eps_target = eps*          :", eps_target)
        print("alpha_target               :", alpha_target)
        print("beta achieved              :", grid_info["beta_achieved"])
        print("residual                   :", grid_info["residual"])
        print()
        print("PELVE reject rate          :", rej_pelve / M)
        print("ZIEGEL reject rate         :", rej_ziegel / M)
        print("PELVE MC reject rate       :", rej_pelve_mc / M)
        print("ZIEGEL MC reject rate      :", rej_ziegel_mc / M)
        print("Expected size              :", alpha_test)
        print()

        #print_grid_result(grid_info, beta)


# ============================================================
# 9) Run
# ============================================================

if __name__ == "__main__":

    nested_mc_reject_rates(
        J=20,
        M=2500, #100, 1000, 2500, 10000

        train_size=20000,
        test_size=10000,

        nu_dgp=7,
        sigma=0.01,
        mu=0.0,

        beta=0.95,

        alpha_test=0.05,
        mc_cal=3000,

        eps_grid_lo=0.01,
        eps_grid_hi=0.03,
        eps_grid_n=41,
    )