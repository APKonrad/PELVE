"""
Script: eps_analysis.py

Purpose:
    Studies the finite-sample behaviour of empirical PELVE estimates over

Thesis reference:
    Chapter 4, Figure 4.4:
        Empirical PELVE for normal and Student-t models, comparing Monte Carlo
        medians and 5%-95% bands with the analytical PELVE.

Description:
    The script computes the analytical PELVE curve over an epsilon grid and
    compares it with Monte Carlo estimates obtained from simulated samples.
    The Monte Carlo median and pointwise 5%-95% band are plotted together with
    the analytical curve.

Usage:
    Set DIST to "normal" or "student". The parameters N_EMP, EPS_MIN, EPS_MAX,
    EPS_POINTS and MC_REPS control the sample size, epsilon grid and number of
    Monte Carlo repetitions.

Notes:
    Figure 4.4 is obtained by running the script for both distributions and for

        n = 250, 1000, 5000.

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.optimize import brentq


# ============================================================
# SETTINGS
# ============================================================

DIST = "student"         # "normal" "student"

# Parameters
MU = 0.0
SIGMA = 1.0

STUDENT_DF = 4.0

# Empirical sample size
N_EMP = 250


SEED = 42


EPS_MIN = 0.001
EPS_MAX = 0.2
EPS_POINTS = 250

# Monte Carlo settings
MC_REPS = 100
SHOW_MC_PATHS = False
MC_PATHS_TO_SHOW = 20


SHOW_TAIL_QUANTILE_PLOT = False


# ============================================================
# ANALYTICAL
# ============================================================

def make_distribution(dist_name):
    if dist_name == "normal":
        return stats.norm(loc=MU, scale=SIGMA)
    elif dist_name == "student":
        return stats.t(df=STUDENT_DF, loc=0.0, scale=1.0)
    else:
        raise ValueError("dist_name must be 'normal' or 'student'")


def var_analytical(X, eps):
    return X.ppf(1.0 - eps)


def es_analytical(X, eps):
    upper = 1.0 - 1e-12
    val, _ = scipy.integrate.quad(lambda u: X.ppf(u), 1.0 - eps, upper, limit=200)
    return val / eps


def pelve_analytical(X, eps):
    var0 = var_analytical(X, eps)

    def f(c):
        return es_analytical(X, c * eps) - var0

    c_low = 1.0
    c_high = 1.0 / eps - 1e-10
    return brentq(f, c_low, c_high)


# ============================================================
# EMPIRICAL
# ============================================================

def empirical_var(sample, eps):
    return np.quantile(sample, 1.0 - eps, method="higher")


def empirical_es(sample, eps):
    q = empirical_var(sample, eps)
    tail = sample[sample >= q]

    if tail.size == 0:
        return q

    return np.mean(tail)


def pelve_empirical(sample, eps):
    var0 = empirical_var(sample, eps)

    def f(c):
        return empirical_es(sample, c * eps) - var0

    c_low = 1.0
    c_high = 1.0 / eps - 1e-10

    f_low = f(c_low)
    f_high = f(c_high)

    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan

    try:
        return brentq(f, c_low, c_high)
    except ValueError:
        return np.nan


# ============================================================
# SAMPLING
# ============================================================

def generate_sample(dist_name, n, seed=None):
    rng = np.random.default_rng(seed)

    if dist_name == "normal":
        sample = rng.normal(loc=MU, scale=SIGMA, size=n)
    elif dist_name == "student":
        sample = rng.standard_t(df=STUDENT_DF, size=n)
    else:
        raise ValueError("dist_name must be 'normal' or 'student'")

    return np.sort(sample)


# ============================================================
# COMPUTATION
# ============================================================

def main():
    X = make_distribution(DIST)
    eps_grid = np.linspace(EPS_MIN, EPS_MAX, EPS_POINTS)


    c_analytical = []
    for eps in eps_grid:
        print(eps)
        c_analytical.append(pelve_analytical(X, eps))
    c_analytical = np.array(c_analytical)


    mc_curves = np.full((MC_REPS, EPS_POINTS), np.nan)

    for m in range(MC_REPS):
        sample = generate_sample(DIST, N_EMP, seed=SEED + m + 1)

        for j, eps in enumerate(eps_grid):
            mc_curves[m, j] = pelve_empirical(sample, eps)

        if (m + 1) % 20 == 0:
            print(f"[MC {m+1}/{MC_REPS}] done")


    c_median = np.nanmedian(mc_curves, axis=0)
    c_lo = np.nanpercentile(mc_curves, 5, axis=0)
    c_hi = np.nanpercentile(mc_curves, 95, axis=0)

    plt.figure(figsize=(10, 6))

    if SHOW_MC_PATHS:
        num_paths = min(MC_PATHS_TO_SHOW, MC_REPS)
        for m in range(num_paths):
            plt.plot(
                eps_grid,
                mc_curves[m, :],
                color="lightgray",
                linewidth=0.8,
                alpha=0.8
            )

    plt.fill_between(
        eps_grid,
        c_lo,
        c_hi,
        alpha=0.25,
        label="MC 5%-95% band"
    )

    plt.plot(
        eps_grid,
        c_median,
        linewidth=1.5,
        label=fr"MC median (n={N_EMP})"
    )

    plt.plot(
        eps_grid,
        c_analytical,
        color="black",
        linewidth=1.5,
        label="Analytical PELVE"
    )

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("PELVE")
    plt.ylim(1, 4)

    plt.legend(frameon=False)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()


    if SHOW_TAIL_QUANTILE_PLOT:
        sample = generate_sample(DIST, N_EMP, seed=SEED)
        u_grid = np.linspace(1.0 - EPS_MAX, 1.0 - EPS_MIN, 250)

        var_true = X.ppf(u_grid)
        var_emp = [np.quantile(sample, u, method="higher") for u in u_grid]
        var_emp = np.array(var_emp)

        plt.figure(figsize=(9, 5))
        plt.plot(u_grid, var_true, linewidth=2, label="Analytical tail quantile")
        plt.step(u_grid, var_emp, where="mid", linewidth=1.2, label="Empirical tail quantile")

        plt.xlabel(r"$u$")
        plt.ylabel(r"$\mathrm{VaR}_u$")

        if DIST == "normal":
            plt.title("Tail quantile function: Normal")
        else:
            plt.title(fr"Tail quantile function: Student-t($\nu={STUDENT_DF}$)")

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()