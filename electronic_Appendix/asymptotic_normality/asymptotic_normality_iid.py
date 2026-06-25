"""
Script: asymptotic_normality_iid.py

Purpose:
    Simulates empirical PELVE estimators in the i.i.d. case and illustrates
    their asymptotic normality.

Thesis reference:
    Chapter 3, Figures 3.1 and 3.2:
        Figure 3.1: Pareto(4), epsilon = 0.05, n = 5,000.
        Figure 3.2: Exp(1), epsilon = 0.05, n = 5,000.

Description:
    The script repeatedly simulates samples from a chosen distribution,
    computes the empirical PELVE estimator c_hat_n, and plots histograms of

        c_hat_n
        sqrt(n) (c_hat_n - c),

    together with the corresponding normal approximations.

Usage:
    Select the distribution in the run section, e.g.

        dist = pareto(4, loc=0, scale=1)

    or

        dist = expon(scale=1).

    The parameters n, eps and R control the sample size, PELVE level and
    number of Monte Carlo repetitions. The normal overlays are controlled by
    varZ_overlay and c_overlay.

Notes:
    Final settings:
        Figure 3.1: dist = pareto(4, loc=0, scale=1),
                    varZ_overlay = 94.425, c_overlay = 3.16.
        Figure 3.2: dist = expon(scale=1),
                    varZ_overlay = 39.05, c_overlay = 2.72.

    In both cases:
        n = 5000, eps = 0.05.
"""


import scipy
from scipy.stats import pareto, expon
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

np.random.seed(123)

###############################################################
# analytical calculation
###############################################################
def VaR(X, eps):
    return X.ppf(1 - eps)

def expected_shortfall(X, eps):
    integral, _ = scipy.integrate.quad(lambda u: X.ppf(u), 1 - eps, 1 - 1e-10)
    return integral / eps

def pelve(X, eps):
    def f(c):
        return expected_shortfall(X, c * eps) - VaR(X, eps)

    c_low = 1.0
    c_high = 1.0 / eps - 1e-10
    return scipy.optimize.brentq(f, c_low, c_high)

###############################################################
# empirical calculation
###############################################################
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

def pelve_emp(window_losses, var_1meps, es_1meps, eps=0.05):
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

###############################################################
# Monte Carlo for Z = sqrt(n)(c_hat - c_true)
###############################################################
def mc_Z(dist, n, eps, R, progress=1000):
    c_true = pelve(dist, eps)

    Z = np.empty(R, float)
    chat = np.empty(R, float)

    for r in range(R):
        if progress is not None and r % progress == 0:
            print(r)

        x = dist.rvs(size=n)

        p = 1.0 - eps
        var_hat = empirical_var(x, p)
        es_hat = empirical_es(x, p)
        c_hat = pelve_emp(x, var_hat, es_hat, eps)

        chat[r] = c_hat
        Z[r] = np.sqrt(n) * (c_hat - c_true)

    return Z, chat, c_true

###############################################################
# plotting
###############################################################
def plot_hist_and_qq_Z(
    Z,
    chat,
    n,
    bins=40,
    varZ_overlay=None,
    c_overlay=None,
    figsize=(7, 4.8)
):


    if varZ_overlay is None:
        raise ValueError("Please provide varZ_overlay.")
    if c_overlay is None:
        raise ValueError("Please provide c_overlay.")

    Z = np.asarray(Z, float)
    chat = np.asarray(chat, float)

    Z = Z[np.isfinite(Z)]
    chat = chat[np.isfinite(chat)]

    muZ_overlay = 0.0
    sigmaZ_overlay = np.sqrt(varZ_overlay)

    muc_overlay = c_overlay
    varc_overlay = varZ_overlay / n
    sigmac_overlay = np.sqrt(varc_overlay)

    # ---------------------------------------------------------
    # 1) Histogram of Z
    # ---------------------------------------------------------
    plt.figure(figsize=figsize)
    plt.hist(
        Z,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        linewidth=0.8
    )

    xZ = np.linspace(Z.min(), Z.max(), 500)
    plt.plot(
        xZ,
        scipy.stats.norm.pdf(xZ, loc=muZ_overlay, scale=sigmaZ_overlay),
        "r",
        linewidth=1.2
    )

    plt.xlabel("Z")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 2) QQ-plot of Z against analytical normal
    # ---------------------------------------------------------
    plt.figure(figsize=figsize)
    scipy.stats.probplot(
        Z,
        dist=scipy.stats.norm(loc=muZ_overlay, scale=sigmaZ_overlay),
        plot=plt
    )
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Sample quantiles")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 3) Histogram of c_hat
    # ---------------------------------------------------------
    plt.figure(figsize=figsize)
    counts, bin_edges, _ = plt.hist(
        chat,
        bins=bins,
        density=False,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.8
    )

    binwidth = bin_edges[1] - bin_edges[0]
    xgrid = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y = scipy.stats.norm.pdf(
        xgrid,
        loc=muc_overlay,
        scale=sigmac_overlay
    ) * (chat.size * binwidth)

    plt.plot(xgrid, y, "r", linewidth=1.2)

    plt.xlabel(r"$\hat{c}_n$")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # 4) QQ-plot of c_hat against analytical normal
    # ---------------------------------------------------------
    plt.figure(figsize=figsize)
    scipy.stats.probplot(
        chat,
        dist=scipy.stats.norm(loc=muc_overlay, scale=sigmac_overlay),
        plot=plt
    )
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Sample quantiles")
    plt.tight_layout()
    plt.show()

###############################################################
# run
###############################################################
n = 5000
eps = 0.05
R = 10000

# choose distribution
#dist = pareto(4, loc=0, scale=1)
dist = expon(scale=1)

Z, chat, c_true = mc_Z(dist, n=n, eps=eps, R=R, progress=500)

print(
    "c_true =", c_true,
    "| Z-mean =", Z[np.isfinite(Z)].mean(),
    "Z-std =", Z[np.isfinite(Z)].std(ddof=1),
    "| chat-mean =", chat[np.isfinite(chat)].mean(),
    "chat-var =", chat[np.isfinite(chat)].var(ddof=1),
)

###############################################################
# analytical parameters
###############################################################
# For Z ~ N(0, varZ_overlay)
varZ_overlay = 39.05 #94.425   # example

# For c_hat ~ N(c_overlay, varZ_overlay / n)
c_overlay = 2.72 #3.16      # example

plot_hist_and_qq_Z(
    Z,
    chat,
    n=n,
    bins=40,
    varZ_overlay=varZ_overlay,
    c_overlay=c_overlay,
    figsize=(7, 4.8)
)