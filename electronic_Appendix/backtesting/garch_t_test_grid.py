"""
Script: garch_t_test_grid.py

Purpose:
    Compares the rejection behaviour of the PELVE-based Kupiec test and the
    Ziegel VaR--ES backtest when risk forecasts are obtained from fitted GARCH
    models.

Thesis reference:
    Chapter 5, Figure 5.6:
        Reject rate of PELVE and Ziegel backtests reported over 20 repetitions
        under correct Student-t and misspecified normal models.

Description:
    The script simulates Student-t losses and fits a GARCH(1,1) model to each
    training sample. The fitted model is then used to generate VaR and ES
    forecasts.


    For each training sample, independent test samples are generated and evaluated
    using

        PELVE/Kupiec test with chi-square critical value,
        Ziegel test with chi-square critical value,
        PELVE/Kupiec test with Monte Carlo calibrated critical value,
        Ziegel test with Monte Carlo calibrated critical value.

    The resulting rejection rates are printed for each training sample.

Usage:
    Set

        garch_dist = "t"

    for the correctly specified Student-t GARCH model, and set

        garch_dist = "normal"

    for the misspecified normal GARCH model.

    The printed rejection rates over the 20 outer repetitions are collected and used
    to produce the boxplots in Figure 5.6.

Notes:
    The final settings for Figure 5.6 are

        J = 20,
        M = 1000,
        train_size = 10000,
        test_size = 5000,
        nu_dgp = 7,
        sigma = 0.01,
        mu = 0.0,
        beta = 0.95,
        alpha_test = 0.05,
        mc_cal = 2000,
        eps_grid_lo = 0.01,
        eps_grid_hi = 0.03,
        eps_grid_n = 41.

    For the Student-t specification, convergence of the fitted GARCH model should
    be checked, since fitting a dynamic GARCH model to iid Student-t data may lead
    to numerical instabilities in some runs.
"""


import numpy as np
from arch import arch_model
from scipy.optimize import brentq
from scipy.stats import chi2, norm, t


# ============================================================
# simulated Student-t losses
# ============================================================

def simulate_losses_student_t(n_days, nu, sigma, mu, seed):
    rng = np.random.default_rng(seed)
    sf = sigma / np.sqrt(nu / (nu - 2))
    z = rng.standard_t(df=nu, size=n_days)
    losses = -(mu + sf * z)
    return losses


# ============================================================
# Empirical VaR / ES / PELVE
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


def pelve_empirical(losses, eps):
    losses = np.asarray(losses, float).reshape(-1)

    var_1meps = empirical_var(losses, 1.0 - eps)

    def f(c):
        level = 1.0 - c * eps
        level = min(max(level, 1e-12), 1.0 - 1e-12)
        return empirical_es(losses, level) - var_1meps

    try:
        return float(brentq(f, 1.0, 1.0 / eps - 1e-8))
    except ValueError:
        return np.nan


def find_eps_star_empirical(losses, beta, eps_grid_lo, eps_grid_hi, eps_grid_n):
    eps_grid = np.linspace(eps_grid_lo, eps_grid_hi, eps_grid_n)

    rows = []

    best_eps = np.nan
    best_c = np.nan
    best_beta_hat = np.nan
    best_resid = np.nan
    best_abs_resid = np.inf

    for eps in eps_grid:
        c_val = pelve_empirical(losses, eps)

        if not np.isfinite(c_val):
            continue

        beta_hat = 1.0 - c_val * eps
        resid = beta_hat - beta
        abs_resid = abs(resid)

        rows.append({
            "eps": eps,
            "c_emp": c_val,
            "beta_hat": beta_hat,
            "resid": resid,
            "abs_resid": abs_resid,
        })

        if abs_resid < best_abs_resid:
            best_eps = eps
            best_c = c_val
            best_beta_hat = beta_hat
            best_resid = resid
            best_abs_resid = abs_resid

    if not np.isfinite(best_eps):
        raise RuntimeError("No valid empirical eps* found on grid.")

    return {
        "eps_star": float(best_eps),
        "alpha_star": float(1.0 - best_eps),
        "c_star": float(best_c),
        "beta_hat": float(best_beta_hat),
        "beta_resid": float(best_resid),
        "grid_rows": rows,
    }


# ============================================================
# Kupiec test
# ============================================================

def kupiec_stat(hit_count, n, p):
    phat = hit_count / n

    eps = 1e-12
    phat = np.clip(phat, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    ll_null = (n - hit_count) * np.log(1 - p) + hit_count * np.log(p)
    ll_alt = (n - hit_count) * np.log(1 - phat) + hit_count * np.log(phat)

    LR = -2.0 * (ll_null - ll_alt)
    return float(LR)


def kupiec_uc_test(hit_count, n, p, alpha_test):
    LR = kupiec_stat(hit_count, n, p)
    crit = chi2.ppf(1.0 - alpha_test, df=1)
    p_value = 1.0 - chi2.cdf(LR, df=1)

    return float(LR), float(p_value), bool(LR > crit)


# ============================================================
# Ziegel test
# ============================================================

def ziegel_stat(losses, q, e, beta):
    losses = np.asarray(losses, float).reshape(-1)

    V = []

    for x in losses:
        I = 1.0 if x > q else 0.0

        v1 = (1.0 - beta) - I
        v2 = q - e - (1.0 / (1.0 - beta)) * I * (q - x)

        V.append([v1, v2])

    V = np.asarray(V)
    n = V.shape[0]

    vbar = V.mean(axis=0)
    S = np.cov(V.T, bias=True)
    S_inv = np.linalg.pinv(S)

    T = n * vbar.T @ S_inv @ vbar
    return float(T)


def ziegel_test(losses, q, e, beta, alpha_test):
    T = ziegel_stat(losses, q, e, beta)
    crit = chi2.ppf(1.0 - alpha_test, df=2)
    p_value = 1.0 - chi2.cdf(T, df=2)

    return float(T), float(p_value), bool(T > crit)


# ============================================================
# MC calibration
# ============================================================

def calibrate_kupiec_mc(
        mc_samples,
        sample_size,
        nu_dgp,
        sigma,
        mu,
        var_level_emp,
        eps_target_emp,
        alpha_test,
        seed):

    stats = []

    for i in range(mc_samples):
        losses = simulate_losses_student_t(
            sample_size,
            nu_dgp,
            sigma,
            mu,
            seed + i,
        )

        hits = losses > var_level_emp
        stat = kupiec_stat(hits.sum(), sample_size, eps_target_emp)
        stats.append(stat)

    return float(np.quantile(stats, 1.0 - alpha_test))


def calibrate_ziegel_mc(
        mc_samples,
        sample_size,
        nu_dgp,
        sigma,
        mu,
        q_emp,
        e_emp,
        beta,
        alpha_test,
        seed):

    stats = []

    for i in range(mc_samples):
        losses = simulate_losses_student_t(
            sample_size,
            nu_dgp,
            sigma,
            mu,
            seed + i,
        )

        stat = ziegel_stat(losses, q_emp, e_emp, beta)
        stats.append(stat)

    return float(np.quantile(stats, 1.0 - alpha_test))


# ============================================================
# GARCH fit and GARCH VaR / ES forecasts
# ============================================================

def fit_garch_forecast(train_losses, dist="t"):
    x = np.asarray(train_losses, float).reshape(-1)

    am = arch_model(
        x,
        mean="Zero",
        vol="GARCH",
        p=1,
        q=1,
        dist=dist,
        rescale=False,
    )

    res = am.fit(disp="off")
    fcast = res.forecast(horizon=1, reindex=False)

    mu = float(fcast.mean.iloc[-1, 0])
    sigma = float(np.sqrt(fcast.variance.iloc[-1, 0]))

    params = res.params

    omega = params.get("omega", np.nan)
    alpha = params.get("alpha[1]", np.nan)
    beta = params.get("beta[1]", np.nan)

    print("\n--- GARCH FIT ---")
    print(f"mu_hat     : {params.get('mu', np.nan):.6f}")
    print(f"sigma_fcast: {sigma:.6f}")
    print(f"omega      : {omega:.6e}")
    print(f"alpha[1]   : {alpha:.6f}")
    print(f"beta[1]    : {beta:.6f}")
    print(f"alpha+beta : {alpha + beta:.6f}")
    print(f"nu_hat     : {params.get('nu', np.nan):.4f}")
    print(f"converged  : {res.convergence_flag == 0}")
    print("------------------\n")

    out = {
        "mu": mu,
        "sigma": sigma,
        "dist": dist,
        "converged": bool(res.convergence_flag == 0),
    }

    if dist == "t":
        out["nu"] = float(params["nu"])

    return out



def forecast_var(fd, alpha):
    mu = fd["mu"]
    sigma = fd["sigma"]

    if fd["dist"] == "normal":
        z = norm.ppf(alpha)
        return float(mu + sigma * z)

    if fd["dist"] == "t":
        nu = fd["nu"]
        z = t.ppf(alpha, df=nu)
        sf = sigma / np.sqrt(nu / (nu - 2))
        return float(mu + sf * z)

    raise ValueError("dist must be 'normal' or 't'")

def forecast_es(fd, alpha):
    mu = fd["mu"]
    sigma = fd["sigma"]

    if fd["dist"] == "normal":
        z = norm.ppf(alpha)
        return float(mu + sigma * norm.pdf(z) / (1.0 - alpha))

    if fd["dist"] == "t":
        nu = fd["nu"]
        z = t.ppf(alpha, df=nu)
        pdf = t.pdf(z, df=nu)
        sf = sigma / np.sqrt(nu / (nu - 2))

        return float(
            mu
            + sf * ((nu + z ** 2) / (nu - 1)) * pdf / (1.0 - alpha)
        )

    raise ValueError("dist must be 'normal' or 't'")




# ============================================================
# Monte Carlo experiment
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

        garch_dist="t"):

    for j in range(1, J + 1):

        # ====================================================
        # A) Training / calibration sample
        # ====================================================
        train_losses = simulate_losses_student_t(
            train_size,
            nu_dgp,
            sigma,
            mu,
            base_seed + j,
        )

        # ====================================================
        # B) Empirical eps* selection
        # ====================================================
        eps_info_emp = find_eps_star_empirical(
            losses=train_losses,
            beta=beta,
            eps_grid_lo=eps_grid_lo,
            eps_grid_hi=eps_grid_hi,
            eps_grid_n=eps_grid_n,
        )

        eps_target_emp = eps_info_emp["eps_star"]
        alpha_target_emp = eps_info_emp["alpha_star"]

        print("----- EMPIRICAL PELVE LEVEL -----")
        print(f"Target ES level beta       : {beta:.8f}")
        print(f"Empirical eps*             : {eps_target_emp:.10f}")
        print(f"Empirical alpha_target     : {alpha_target_emp:.10f}")
        print(f"Empirical c(eps*)          : {eps_info_emp['c_star']:.10f}")
        print(f"Achieved beta_hat          : {eps_info_emp['beta_hat']:.10f}")
        print(f"Residual beta_hat - beta   : {eps_info_emp['beta_resid']:.4e}")
        print("---------------------------------\n")

        # ====================================================
        # C) GARCH forecast model fitted on training sample
        # ====================================================
        fd = fit_garch_forecast(train_losses, dist=garch_dist)
        print(fd)

        # GARCH VaR at empirically PELVE-implied level
        var_level_garch = forecast_var(fd, alpha_target_emp)

        # GARCH VaR/ES at beta for Ziegel test
        q_garch = forecast_var(fd, beta)
        e_garch = forecast_es(fd, beta)

        # ====================================================
        # D) Empirical quantities for MC calibration
        # ====================================================
        var_level_emp = empirical_var(train_losses, alpha_target_emp)
        q_emp = empirical_var(train_losses, beta)
        e_emp = empirical_es(train_losses, beta)

        kupiec_crit_mc = calibrate_kupiec_mc(
            mc_samples=mc_cal,
            sample_size=test_size,
            nu_dgp=nu_dgp,
            sigma=sigma,
            mu=mu,
            var_level_emp=var_level_emp,
            eps_target_emp=eps_target_emp,
            alpha_test=alpha_test,
            seed=base_seed + 10000 * j,
        )

        ziegel_crit_mc = calibrate_ziegel_mc(
            mc_samples=mc_cal,
            sample_size=test_size,
            nu_dgp=nu_dgp,
            sigma=sigma,
            mu=mu,
            q_emp=q_emp,
            e_emp=e_emp,
            beta=beta,
            alpha_test=alpha_test,
            seed=base_seed + 20000 * j,
        )

        # ====================================================
        # E) Test repetitions
        # ====================================================
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


            hits = test_losses > var_level_garch
            hit_count = int(hits.sum())

            LR, _, r_kupiec = kupiec_uc_test(
                hit_count=hit_count,
                n=test_size,
                p=eps_target_emp,
                alpha_test=alpha_test,
            )

            r_kupiec_mc = LR > kupiec_crit_mc


            T, _, r_ziegel = ziegel_test(
                losses=test_losses,
                q=q_garch,
                e=e_garch,
                beta=beta,
                alpha_test=alpha_test,
            )

            r_ziegel_mc = T > ziegel_crit_mc

            rej_pelve += int(r_kupiec)
            rej_pelve_mc += int(r_kupiec_mc)

            rej_ziegel += int(r_ziegel)
            rej_ziegel_mc += int(r_ziegel_mc)

        # ====================================================
        # F) Output
        # ====================================================
        print("\n================ RESULT =================")
        print("Training sample:", j)
        print("PELVE reject rate     :", rej_pelve / M)
        print("ZIEGEL reject rate    :", rej_ziegel / M)
        print("PELVE MC reject rate  :", rej_pelve_mc / M)
        print("ZIEGEL MC reject rate :", rej_ziegel_mc / M)
        print("Expected size         :", alpha_test)
        print("eps_target_emp        :", eps_target_emp)
        print("alpha_target_emp      :", alpha_target_emp)
        print("GARCH VaR level       :", var_level_garch)
        print("Kupiec MC crit        :", kupiec_crit_mc)
        print("Ziegel MC crit        :", ziegel_crit_mc)
        print()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    nested_mc_reject_rates(

        J=20,
        M=1000,

        train_size=10000,
        test_size=5000,

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

        garch_dist="t", #t
    )