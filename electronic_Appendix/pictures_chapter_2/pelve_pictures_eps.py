"""
Script: pelve_pictures_eps.py

Purpose:
    Computes and plots PELVE values for selected parametric distribution families
    over varying levels of epsilon. The script is used to generate the
    varying-epsilon panels of Figure 2.2 in the thesis.

Thesis reference:
    Chapter 2, Figure 2.2:
        (b) Normal distribution for varying epsilon
        (d) Student-t distribution for varying epsilon
        (f) Lognormal distribution for varying epsilon

Description:
    For a chosen distribution family, the script computes the PELVE value c
    satisfying

        ES_{1-c epsilon}(X) = VaR_{1-epsilon}(X)

    over a grid of epsilon values. The x-axis is displayed as 1 - epsilon,
    corresponding to the VaR confidence level.

    The red dashed horizontal line marks Euler's number e, which is used as
    a reference value in the discussion of PELVE.

Usage:
    Select the distribution in the CONFIG block by setting

        SELECT = "normal", "student", "lognormal", or "pareto".

"""

import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# CONFIG
# ============================================================

SELECT = "lognormal"       # "normal", "student", "lognormal", or "pareto"


PARAM_LISTS = {
    "normal": [1.0],
    "student": [2, 4, 10, 30],
    "lognormal": [0.01, 0.04, 0.25, 1.0],
    "pareto": [2, 4, 10],
}

EPS_MIN = 1e-4
EPS_MAX = 0.2
N_EPS = 180


X_AXIS = "one_minus_eps"     # "one_minus_eps" or "eps"

SAVE_FIG = True
SHOW_FIG = True
OUTFILE = f"pelve_eps_{SELECT}.png"

FIGSIZE = (7.5, 4.8)
DPI = 300

BLUE_COLORS = ["#0B2545", "#1F4E79", "#5B8DB8", "#A8C7E6"]
E_COLOR = "#C0392B"

SHOW_TITLE = False
SHOW_LEGEND = True

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


# Closed-form helper for lognormal: much faster and more stable
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


def make_distribution(distribution, param):
    if distribution == "normal":
        return scipy.stats.norm(loc=0, scale=param)
    if distribution == "student":
        return scipy.stats.t(df=param, loc=0, scale=1)
    if distribution == "pareto":
        return scipy.stats.pareto(b=param, loc=0, scale=1)
    raise ValueError("Distribution handled separately or unknown.")


def safe_pelve_for_eps(distribution, param, eps):
    try:
        if distribution == "lognormal":
            return pelve_lognormal_from_sigma2(param, eps)
        X = make_distribution(distribution, param)
        return pelve(X, eps)
    except Exception:
        return np.nan


def curve_label(distribution, param):
    if distribution == "normal":
        return rf"$\sigma={param}$"
    if distribution == "student":
        return rf"$\nu={param}$"
    if distribution == "lognormal":
        return rf"$\sigma^2={param}$"
    if distribution == "pareto":
        return rf"$\alpha={param}$"
    return str(param)


def title_text(distribution):
    if distribution == "normal":
        return r"Normal, $\varepsilon\in(0,0.2)$"
    if distribution == "student":
        return r"Student-$t$, $\varepsilon\in(0,0.2)$"
    if distribution == "lognormal":
        return r"Lognormal, $\varepsilon\in(0,0.2)$"
    if distribution == "pareto":
        return r"Pareto, $\varepsilon\in(0,0.2)$"
    return distribution


# ============================================================
# MAIN PLOT
# ============================================================

def main():
    if SELECT not in PARAM_LISTS:
        raise ValueError("SELECT must be 'normal', 'student', 'lognormal', or 'pareto'.")

    params = PARAM_LISTS[SELECT]
    eps_grid = np.linspace(EPS_MIN, EPS_MAX, N_EPS)

    if X_AXIS == "one_minus_eps":
        x = 1.0 - eps_grid
        xlabel = r"$1-\varepsilon$"
    elif X_AXIS == "eps":
        x = eps_grid
        xlabel = r"$\varepsilon$"
    else:
        raise ValueError("X_AXIS must be 'one_minus_eps' or 'eps'.")

    print("Selected distribution:", SELECT)
    print("parameters:", params)
    print("epsilon range:", (EPS_MIN, EPS_MAX))
    print("x-axis:", X_AXIS)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.axhline(np.e, color=E_COLOR, linestyle="--", linewidth=1.1, label=r"$e$")

    colors = BLUE_COLORS[:len(params)]
    if len(colors) < len(params):
        raise ValueError("Add more colors to BLUE_COLORS for this many parameters.")

    for color, param in zip(colors, params):
        pelves = np.array([safe_pelve_for_eps(SELECT, param, eps) for eps in eps_grid])
        valid = np.isfinite(pelves).sum()
        print(f"  param={param}: valid PELVE values {valid}/{len(eps_grid)}")

        ax.plot(x, pelves, color=color, linewidth=1.7, label=curve_label(SELECT, param))

    ax.set_xlabel(xlabel)
    ax.set_ylabel("PELVE")

    if X_AXIS == "one_minus_eps":
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if X_AXIS == "eps":
        ax.set_xlim(EPS_MAX, EPS_MIN)  # epsilon decreases to the right

    if SHOW_TITLE:
        ax.set_title(title_text(SELECT))
    if SHOW_LEGEND:
        ax.legend(loc="upper left", frameon=False, facecolor="white", framealpha=1)

    fig.tight_layout()

    if SAVE_FIG:
        fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
        print("Saved figure to:", OUTFILE)
    if SHOW_FIG:
        plt.show()


if __name__ == "__main__":
    main()