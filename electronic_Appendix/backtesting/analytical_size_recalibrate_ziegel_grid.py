"""
Script: analytical_size_recalibrate_ziegel_grid.py

Purpose:
    Compares the empirical size of the PELVE-based Kupiec test and the Ziegel
    ES backtest under correctly specified Student-t losses.

Thesis reference:
    Chapter 5, Figure 5.2:
        Size comparison of the PELVE and Ziegel tests for Student-t losses
        with n = 5,000.

Description:
    The script first determines the PELVE-implied VaR level by solving for
    epsilon such that the PELVE relation matches the target ES level beta = 0.95.
    It then simulates Student-t losses under the null model and computes
    rejection rates for

        PELVE/Kupiec test,
        Ziegel test with chi-square critical value,
        Ziegel test with Monte-Carlo calibrated critical value.

    The experiment is repeated for different numbers of Monte Carlo iterations.

Notes:
    The final settings for Figure 5.2 are

        MC_levels = [100, 1000, 2500, 5000],
        repetitions = 20,
        n_days = 5000,
        beta_global = 0.95,
        nu_global = 7.

    The script may take a long time to run for the larger MC_levels. For a quick
    test run, reduce MC_levels or repetitions.
    Since no fixed random seed is set, different runs of the script will generally
    produce slightly different rejection rates. However, for the final settings
    above, the overall pattern should remain stable and close to the values shown
    in Figure 5.1.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, chi2
from scipy.optimize import brentq


# ============================================================
# 1) Simulation
# ============================================================

def simulate_losses_student_t(n_days, nu, sigma, mu):
    sf = sigma / np.sqrt(nu / (nu - 2))
    z = np.random.standard_t(df=nu, size=n_days)
    losses = -(mu + sf * z)
    return losses


# ============================================================
# 2) Theoretical VaR / ES
# ============================================================

def var_student_t_loss(alpha, nu, mu, sigma):
    sf = sigma / np.sqrt(nu / (nu - 2))
    z = t.ppf(1 - alpha, df=nu)
    return -(mu + sf * z)


def es_student_t_loss(alpha, nu, mu, sigma):
    sf = sigma / np.sqrt(nu / (nu - 2))
    z = t.ppf(1 - alpha, df=nu)
    pdf = t.pdf(z, df=nu)

    es_z = -((nu + z**2) / (nu - 1)) * pdf / (1 - alpha)
    return -(mu + sf * es_z)


# ============================================================
# 3) PELVE
# ============================================================

def pelve_c_theoretical_once(eps, nu, mu, sigma):
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must lie in (0,1).")

    var_target = var_student_t_loss(1 - eps, nu, mu, sigma)

    def f(c):
        level = 1 - c * eps
        es_level = es_student_t_loss(level, nu, mu, sigma)
        return es_level - var_target

    c_upper = 1.0 / eps - 1e-8
    return brentq(f, 1.0, c_upper)


# ============================================================
# Outer numerical solution for eps given beta
# ============================================================

def solve_eps_for_beta_theoretical(beta, nu, mu, sigma,
                                   eps_lo=1e-4, eps_hi=0.05):
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0,1).")
    if not (0.0 < eps_lo < eps_hi < 1.0):
        raise ValueError("Need 0 < eps_lo < eps_hi < 1.")

    def g(eps):
        c_val = pelve_c_theoretical_once(eps, nu, mu, sigma)
        return 1.0 - c_val * eps - beta

    g_lo = g(eps_lo)
    g_hi = g(eps_hi)

    if g_lo * g_hi > 0:
        raise ValueError(
            "No sign change found for outer root search. "
            f"g(eps_lo)={g_lo:.6e}, g(eps_hi)={g_hi:.6e}. "
            "Try adjusting eps_lo / eps_hi."
        )

    eps_star = brentq(g, eps_lo, eps_hi)
    return eps_star


# ============================================================
# 4) Kupiec Test (PELVE)
# ============================================================

def kupiec_uc_test(hit_count, n, p, alpha):
    phat = hit_count / n

    eps = 1e-12
    phat = np.clip(phat, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    ll_null = (n - hit_count) * np.log(1 - p) + hit_count * np.log(p)
    ll_alt = (n - hit_count) * np.log(1 - phat) + hit_count * np.log(phat)

    LR = -2 * (ll_null - ll_alt)
    crit = chi2.ppf(1 - alpha, df=1)

    return LR > crit


# ============================================================
# 5) Ziegel Test statistic
# ============================================================

def ziegel_statistic(losses, beta, nu, mu, sigma):
    q = var_student_t_loss(beta, nu, mu, sigma)
    e = es_student_t_loss(beta, nu, mu, sigma)

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

    T = n * vbar.T @ S_inv @ vbar
    return T


# ============================================================
# 6) Monte-Carlo calibration of Ziegel
# ============================================================

def calibrate_ziegel_MC(n_sim, n_days, beta, nu, mu, sigma):
    T_values = []

    for _ in range(n_sim):
        losses = simulate_losses_student_t(n_days, nu, sigma, mu)
        T = ziegel_statistic(losses, beta, nu, mu, sigma)
        T_values.append(T)

    crit = np.quantile(T_values, 0.95)
    return crit


# ============================================================
# 7) Helper: report theoretical calibration quantities
# ============================================================

def compute_theoretical_calibration(beta, nu, mu, sigma,
                                    eps_lo=1e-4, eps_hi=0.05):
    eps_star = solve_eps_for_beta_theoretical(
        beta=beta,
        nu=nu,
        mu=mu,
        sigma=sigma,
        eps_lo=eps_lo,
        eps_hi=eps_hi
    )

    c_star = pelve_c_theoretical_once(eps_star, nu, mu, sigma)
    beta_achieved = 1.0 - c_star * eps_star
    residual_beta = beta_achieved - beta
    residual_tail = c_star * eps_star - (1.0 - beta)
    alpha_target = 1.0 - eps_star

    return {
        "beta_target": beta,
        "eps_star": eps_star,
        "c_star": c_star,
        "beta_achieved": beta_achieved,
        "residual_beta": residual_beta,
        "residual_tail": residual_tail,
        "alpha_target": alpha_target
    }


# ============================================================
# 8) Single MC run
# ============================================================

def run_mc_once(M, ziegel_MC_crit, print_setup=False):
    n_days = 5000
    nu_dgp = 7
    sigma = 0.01
    mu = 0.0
    beta = 0.95
    alpha_test = 0.05

    calibration = compute_theoretical_calibration(
        beta=beta,
        nu=nu_dgp,
        mu=mu,
        sigma=sigma,
        eps_lo=1e-4,
        eps_hi=0.05
    )

    eps_target = calibration["eps_star"]
    c_const = calibration["c_star"]
    alpha_target = calibration["alpha_target"]

    if print_setup:
        print("\n================ THEORETICAL CALIBRATION ================")
        print(f"Target ES level beta          : {calibration['beta_target']:.8f}")
        print(f"Numerically solved eps*       : {calibration['eps_star']:.10f}")
        print(f"PELVE c(eps*)                : {calibration['c_star']:.10f}")
        print(f"Achieved beta                : {calibration['beta_achieved']:.10f}")
        print(f"Residual beta_achieved-beta  : {calibration['residual_beta']:.4e}")
        print(f"Residual c(eps*)eps*-(1-beta): {calibration['residual_tail']:.4e}")
        print(f"Final VaR level alpha_target : {calibration['alpha_target']:.10f}")
        print(f"Final hit probability eps*   : {eps_target:.10f}")
        print(f"Test sample size n_days      : {n_days}")
        print(f"DGP                          : Student-t, nu={nu_dgp}, mu={mu}, sigma={sigma}")
        print("=========================================================\n")

    chi2_crit = chi2.ppf(1 - alpha_test, df=2)

    rej_pelve = 0
    rej_ziegel_chi2 = 0
    rej_ziegel_mc = 0

    for _ in range(M):
        losses = simulate_losses_student_t(
            n_days,
            nu_dgp,
            sigma,
            mu
        )

        var_level = var_student_t_loss(alpha_target, nu_dgp, mu, sigma)

        hits = losses > var_level
        hit_count = hits.sum()

        r_kupiec = kupiec_uc_test(
            hit_count,
            n_days,
            eps_target,
            alpha_test
        )

        T = ziegel_statistic(
            losses,
            beta,
            nu_dgp,
            mu,
            sigma
        )

        r_ziegel_chi2 = T > chi2_crit
        r_ziegel_mc = T > ziegel_MC_crit

        rej_pelve += r_kupiec
        rej_ziegel_chi2 += r_ziegel_chi2
        rej_ziegel_mc += r_ziegel_mc

    return (
        rej_pelve / M,
        rej_ziegel_chi2 / M,
        rej_ziegel_mc / M,
        calibration
    )


# ============================================================
# 9) Experiment
# ============================================================

MC_levels = [100, 1000, 2500, 5000]
repetitions = 20

beta_global = 0.95
nu_global = 7
mu_global = 0.0
sigma_global = 0.01

print("Calibrating Ziegel critical value via MC...")
ziegel_MC_crit = calibrate_ziegel_MC(
    n_sim=5000,
    n_days=5000,
    beta=beta_global,
    nu=nu_global,
    mu=mu_global,
    sigma=sigma_global
)

print("MC critical value :", ziegel_MC_crit)
print("Chi2 critical value:", chi2.ppf(0.95, df=2))


setup_info = compute_theoretical_calibration(
    beta=beta_global,
    nu=nu_global,
    mu=mu_global,
    sigma=sigma_global,
    eps_lo=1e-4,
    eps_hi=0.05
)

print("\n================ FINAL VALUES USED IN THE PROGRAM ================")
print(f"Target ES level beta          : {setup_info['beta_target']:.8f}")
print(f"Numerically solved eps*       : {setup_info['eps_star']:.10f}")
print(f"PELVE c(eps*)                : {setup_info['c_star']:.10f}")
print(f"Achieved beta                : {setup_info['beta_achieved']:.10f}")
print(f"Residual beta_achieved-beta  : {setup_info['residual_beta']:.4e}")
print(f"Residual c(eps*)eps*-(1-beta): {setup_info['residual_tail']:.4e}")
print(f"Final VaR level alpha_target : {setup_info['alpha_target']:.10f}")
print("=================================================================\n")

results_pelve = []
results_ziegel_chi2 = []
results_ziegel_mc = []

for M in MC_levels:
    pelve_rates = []
    ziegel_chi2_rates = []
    ziegel_mc_rates = []

    print("\n==============================")
    print("MC iterations:", M)
    print("==============================")

    for r in range(repetitions):
        pelve_r, ziegel_c, ziegel_m, calibration = run_mc_once(
            M,
            ziegel_MC_crit,
            print_setup=(r == 0 and M == MC_levels[0])
        )

        pelve_rates.append(pelve_r)
        ziegel_chi2_rates.append(ziegel_c)
        ziegel_mc_rates.append(ziegel_m)

        print(
            f"rep {r+1:02d}  "
            f"PELVE: {pelve_r:.4f}   "
            f"ZIEGEL chi2: {ziegel_c:.4f}   "
            f"ZIEGEL MC: {ziegel_m:.4f}"
        )

    results_pelve.append(pelve_rates)
    results_ziegel_chi2.append(ziegel_chi2_rates)
    results_ziegel_mc.append(ziegel_mc_rates)