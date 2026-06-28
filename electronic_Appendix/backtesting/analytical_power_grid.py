"""
Script: analytical_power_grid.py

Purpose:
    Compares the power of the PELVE-based Kupiec test and the Ziegel ES
    backtest under misspecified VaR and ES levels.

Thesis reference:
    Chapter 5, Figure 5.3:
        Rejection rates of the PELVE and Ziegel tests for different false
        VaR--ES tuples, reported over 10 repetitions.

Description:
    The script simulates Student-t losses and computes rejection rates
    for different false VaR--ES tuples for

        PELVE/Kupiec test with chi-square critical value,
        Ziegel test with Monte-Carlo calibrated critical value.

    The experiment is repeated 10 times for each false tuple.

Notes:
    The final settings for Figure 5.3 are

        M = 2500,
        repetitions = 10,
        n_days = 5000,
        beta_true = 0.95,
        nu_dgp = 7.

    The script may take a long time to run. For a quick test run, reduce M,
    repetitions or the number of false tuples. Since no fixed random seed is
    set, different runs will generally produce slightly different rejection
    rates. However, the overall pattern should remain stable and close to the
    values shown in Figure 5.2.
"""

import numpy as np
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
# 2) VaR / ES
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

    return brentq(f, 1.0, 1.0 / eps - 1e-8)


# ============================================================
# Outer numerical solution for eps given beta
# ============================================================

def solve_eps_for_beta_theoretical(beta, nu, mu, sigma,
                                   eps_lo=1e-4, eps_hi=0.05):
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must lie in (0,1).")

    def g(eps):
        c_val = pelve_c_theoretical_once(eps, nu, mu, sigma)
        return 1.0 - c_val * eps - beta

    g_lo = g(eps_lo)
    g_hi = g(eps_hi)

    if g_lo * g_hi > 0:
        raise ValueError(
            "Outer root search failed: no sign change found. "
            f"g(eps_lo)={g_lo:.6e}, g(eps_hi)={g_hi:.6e}. "
            "Try changing eps_lo / eps_hi."
        )

    return brentq(g, eps_lo, eps_hi)


# ============================================================
# Collect all calibration quantities
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
        "alpha_target": alpha_target,
    }


# ============================================================
# 4) Kupiec Test
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
# 5) Ziegel statistic
# ============================================================

def ziegel_statistic(losses, beta_true, beta_es, nu, mu, sigma):
    q = var_student_t_loss(beta_true, nu, mu, sigma)
    e = es_student_t_loss(beta_es, nu, mu, sigma)

    V = []

    for x in losses:
        I = 1.0 if x > q else 0.0

        v1 = (1 - beta_true) - I
        v2 = q - e - (1 / (1 - beta_true)) * I * (q - x)

        V.append([v1, v2])

    V = np.asarray(V)

    n = V.shape[0]
    vbar = V.mean(axis=0)

    S = np.cov(V.T, bias=True)
    S_inv = np.linalg.pinv(S)

    T = n * vbar.T @ S_inv @ vbar
    return T


# ============================================================
# 6) MC calibration for Ziegel
# ============================================================

def calibrate_ziegel_MC(n_sim, n_days, beta_true, nu, mu, sigma):
    T_values = []

    for _ in range(n_sim):
        losses = simulate_losses_student_t(n_days, nu, sigma, mu)

        T = ziegel_statistic(
            losses,
            beta_true,
            beta_true,
            nu,
            mu,
            sigma
        )

        T_values.append(T)

    return np.quantile(T_values, 0.95)


# ============================================================
# 7) POWER EXPERIMENT
# ============================================================

M = 2500
repetitions = 10

n_days = 5000
nu_dgp = 7
sigma = 0.01
mu = 0.0

beta_true = 0.95
alpha_test = 0.05




setup_info = compute_theoretical_calibration(
    beta=beta_true,
    nu=nu_dgp,
    mu=mu,
    sigma=sigma,
    eps_lo=1e-4,
    eps_hi=0.05
)

eps_target = setup_info["eps_star"]
c_const = setup_info["c_star"]
alpha_true = setup_info["alpha_target"]

print("================ TRUE CALIBRATION USED ================")
print(f"Target ES level beta_true       : {setup_info['beta_target']:.8f}")
print(f"Numerically solved eps*         : {setup_info['eps_star']:.10f}")
print(f"PELVE c(eps*)                  : {setup_info['c_star']:.10f}")
print(f"Achieved beta                  : {setup_info['beta_achieved']:.10f}")
print(f"Residual beta_achieved-beta    : {setup_info['residual_beta']:.4e}")
print(f"Residual c(eps*)eps*-(1-beta)  : {setup_info['residual_tail']:.4e}")
print(f"Final VaR level alpha_true     : {setup_info['alpha_target']:.10f}")
print(f"Final hit probability eps*     : {eps_target:.10f}")
print(f"DGP                            : Student-t, nu={nu_dgp}, mu={mu}, sigma={sigma}")
print("=======================================================\n")


# ------------------------------------------------
# Ziegel MC calibration
# ------------------------------------------------

print("Calibrating Ziegel critical value...")

ziegel_MC_crit = calibrate_ziegel_MC(
    n_sim=5000,
    n_days=n_days,
    beta_true=beta_true,
    nu=nu_dgp,
    mu=mu,
    sigma=sigma
)

print("MC critical value:", ziegel_MC_crit)
print("")


# ------------------------------------------------
# FALSE TUPLES
# ------------------------------------------------
false_tuples = [
    (0.9700, 0.900),
    (0.9862, 0.960),
    (0.978, 0.940),
    (0.9840, 0.955),
    (0.9805, 0.945),
]




# ============================================================
# iterate over tuples
# ============================================================

for alpha_false, beta_false in false_tuples:

    print("================================")
    print(f"FALSE TUPLE: alpha={alpha_false}, beta={beta_false}")
    print("================================")
    print(f"True alpha used in PELVE/Kupiec : {alpha_true:.10f}")
    print(f"True eps* used in PELVE/Kupiec  : {eps_target:.10f}")
    print(f"True beta target               : {beta_true:.10f}")
    print(f"False alpha for misspec. VaR   : {alpha_false:.10f}")
    print(f"False beta for misspec. ES     : {beta_false:.10f}")
    print("")

    var_false = var_student_t_loss(alpha_false, nu_dgp, mu, sigma)

    for r in range(repetitions):

        rej_pelve = 0
        rej_ziegel = 0

        for _ in range(M):

            losses = simulate_losses_student_t(
                n_days,
                nu_dgp,
                sigma,
                mu
            )

            # PELVE (Kupiec)
            hits = losses > var_false
            hit_count = hits.sum()

            r_kupiec = kupiec_uc_test(
                hit_count,
                n_days,
                eps_target,
                alpha_test
            )

            # Ziegel
            T = ziegel_statistic(
                losses,
                beta_true,
                beta_false,
                nu_dgp,
                mu,
                sigma
            )

            r_ziegel = T > ziegel_MC_crit

            rej_pelve += r_kupiec
            rej_ziegel += r_ziegel

        pelve_rate = rej_pelve / M
        ziegel_rate = rej_ziegel / M

        print(
            f"rep {r+1:02d}  "
            f"PELVE: {pelve_rate:.4f}   "
            f"ZIEGEL MC: {ziegel_rate:.4f}"
        )