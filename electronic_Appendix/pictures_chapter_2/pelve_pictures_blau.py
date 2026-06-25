"""
Script: pelve_pictures_blau.py

Purpose:
    Computes and plots PELVE values for selected parametric distribution families
    for a fixed level epsilon. The script is used to generate the fixed-epsilon
    panels of Figure 2.2 in the thesis.

Thesis reference:
    Chapter 2, Figure 2.2:
        (a) Pareto distribution for fixed epsilon = 0.01
        (c) Student-t distribution for fixed epsilon = 0.01
        (e) Lognormal distribution for fixed epsilon = 0.01

Description:
    For a chosen distribution family, the script varies the corresponding
    distributional parameter and computes the PELVE value c satisfying

        ES_{1-c epsilon}(X) = VaR_{1-epsilon}(X).

    The red dashed horizontal line marks Euler's number e, which is used as
    a reference value in the discussion of PELVE.

Usage:
    Select the distribution in the CONFIG block by setting

        SELECT = "pareto", "student", or "lognormal".

"""

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# CONFIG
# ============================================================

SELECT = "pareto"       # "pareto", "student", or "lognormal"
EPS = 0.01

SAVE_FIG = False
SHOW_FIG = True
OUTFILE = f"pelve_parameter_{SELECT}_eps_{EPS}.png"

FIGSIZE = (7.5, 4.8)
DPI = 300

CURVE_COLOR = "#5B8DB8"   # calm blue #BLUE_COLORS = ["#0B2545", "#1F4E79", "#5B8DB8", "#A8C7E6"]
E_COLOR = "#C0392B"       # red benchmark

SHOW_TITLE = False
SHOW_LEGEND = False

# Parameter grids
PARAM_GRIDS = {
    "pareto": np.linspace(1.2, 50.0, 80),
    "student": np.linspace(1.2, 50.0, 80),
    "lognormal": np.linspace(0.01, 2.0, 80),
}

# Labels
X_LABELS = {
    "pareto": r"$\alpha$",
    "student": r"$\nu$",
    "lognormal": r"$\sigma^2$",
}

TITLES = {
    "pareto": rf"Pareto($\alpha$), $\varepsilon={EPS}$",
    "student": rf"Student-$t$($\nu$), $\varepsilon={EPS}$",
    "lognormal": rf"LN($\sigma^2$), $\varepsilon={EPS}$",
}

# ============================================================
# PELVE FUNCTIONS
# ============================================================

def VaR(X, eps):
    return X.ppf(1.0 - eps)


def expected_shortfall(X, eps):
    val, _ = scipy.integrate.quad(lambda u: X.ppf(u), 1.0 - eps, 1.0 - 1e-10)
    return val / eps


def pelve(X, eps):
    def f(c):
        return expected_shortfall(X, c * eps) - VaR(X, eps)

    c_low = 1.0
    c_high = 1.0 / eps - 1e-10
    return scipy.optimize.brentq(f, c_low, c_high)



def VaR_lognormal(sigma, eps, mu=0.0):
    z = norm.ppf(1.0 - eps)
    return np.exp(mu + sigma * z)


def ES_lognormal(sigma, eps, mu=0.0):
    z = norm.ppf(1.0 - eps)
    return (np.exp(mu + 0.5 * sigma**2) * norm.cdf(sigma - z)) / eps


def pelve_lognormal_from_sigma2(sigma2, eps, mu=0.0):
    sigma = np.sqrt(sigma2)
    var0 = VaR_lognormal(sigma, eps, mu)

    def f(c):
        return ES_lognormal(sigma, c * eps, mu) - var0

    c_low = 1.0
    c_high = 1.0 / eps - 1e-12
    return scipy.optimize.brentq(f, c_low, c_high)


def safe_pelve_for_param(distribution, param, eps):
    try:
        if distribution == "pareto":
            X = scipy.stats.pareto(b=param, loc=0, scale=1)
            return pelve(X, eps)
        if distribution == "student":
            X = scipy.stats.t(df=param, loc=0, scale=1)
            return pelve(X, eps)
        if distribution == "lognormal":
            sigma2 = param
            return pelve_lognormal_from_sigma2(sigma2, eps)
        raise ValueError("SELECT must be 'pareto', 'student', or 'lognormal'.")
    except Exception:
        return np.nan


# ============================================================
# MAIN PLOT
# ============================================================

def main():
    if SELECT not in PARAM_GRIDS:
        raise ValueError("SELECT must be 'pareto', 'student', or 'lognormal'.")

    params = PARAM_GRIDS[SELECT]
    pelves = np.array([safe_pelve_for_param(SELECT, p, EPS) for p in params])

    print("Selected distribution:", SELECT)
    print("epsilon:", EPS)
    print("parameter range:", (float(params[0]), float(params[-1])))
    print("number of valid PELVE values:", np.isfinite(pelves).sum(), "of", len(pelves))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(params, pelves, color=CURVE_COLOR, linewidth=1.8, label="PELVE")
    ax.axhline(np.e, color=E_COLOR, linestyle="--", linewidth=1.1, label=r"$e$")

    ax.set_xlabel(X_LABELS[SELECT])
    ax.set_ylabel("PELVE")

    if SHOW_TITLE:
        ax.set_title(TITLES[SELECT])
    if SHOW_LEGEND:
        ax.legend(frameon=False)

    fig.tight_layout()

    if SAVE_FIG:
        fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
        print("Saved figure to:", OUTFILE)
    if SHOW_FIG:
        plt.show()


if __name__ == "__main__":
    main()