"""
Script: asymptotic_normality_alpha_AR.py

Purpose:
    Simulates empirical PELVE estimators for a Gaussian AR(1) process and
    illustrates the asymptotic normality result for alpha-mixing data.

Thesis reference:
    Chapter 3, Figure 3.4:
        Empirical estimators for Gaussian AR(1), phi = 0.1,

Description:
    The script runs a Monte Carlo study. In each repetition, a sample from a
    stationary Gaussian AR(1) process is simulated and the empirical PELVE
    estimator c_hat_n is computed. These simulated values form the histograms of

        c_hat_n
        sqrt(n) (c_hat_n - c).

    The red normal densies variance is computed from the limiting variance formula in Proposition 3.6.
    Since this formula involves integrals and covariance terms which are not available in a
    simple closed form, the variance is approximated numerically on a grid.

Usage:
    The parameters phi, n, eps and R control the AR(1) coefficient, sample size,
    PELVE level and number of Monte Carlo repetitions. The parameter G controls
    the grid size used for the numerical approximation of the limiting variance.
    The parameter K controls the truncation of the infinite lag sum.

Notes:
    Final settings for Figure 3.4:

        phi = 0.1,
        n = 5000,
        eps = 0.05,
        R = 10000,
        G = 600,
        K = 10.
"""


import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

np.random.seed(123)

###############################################################
# regular calculation
###############################################################
def VaR(X, eps):
    var = X.ppf(1 - eps)
    return var

def expexted_shortfall(X, eps):
    int, _ = scipy.integrate.quad(lambda u: X.ppf(u), 1 - eps, 1 - 1e-10)
    es = 1 / (eps) * int
    return es

def pelve(X, eps):
    def f(c):
        return expexted_shortfall(X, c * eps) - VaR(X, eps)

    c_low  = 1.0
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

    c_low  = 1.0
    c_high = 1.0 / eps - 1e-12

    f_low  = f(c_low)
    f_high = f(c_high)
    if f_low * f_high > 0:
        return np.nan

    return float(brentq(f, c_low, c_high))

###############################################################
# alpha-mixing: Gaussian AR(1)
###############################################################
def ar1_gaussian(n, phi=0.8, sigma_eps=1.0, burn=2000):
    if not (-1.0 < phi < 1.0):
        raise ValueError("Need |phi|<1 for stationarity.")

    sigma_stat = sigma_eps / np.sqrt(1.0 - phi**2)

    # Start aus stationärer Verteilung
    x_prev = np.random.normal(loc=0.0, scale=sigma_stat)

    x = np.empty(burn + n, float)
    for t in range(burn + n):
        e = np.random.normal(loc=0.0, scale=sigma_eps)
        x_prev = phi * x_prev + e
        x[t] = x_prev

    return x[burn:]

###############################################################
# Monte-Carlo: Z = sqrt(n)(c_hat - c_true) for AR(1)
###############################################################
def mc_Z_ar1(n, eps, R, phi=0.8, sigma_eps=1.0, progress=1000):
    sigma_stat = sigma_eps / np.sqrt(1.0 - phi**2)
    dist_marg  = scipy.stats.norm(loc=0.0, scale=sigma_stat)

    c_true = pelve(dist_marg, eps)

    Z    = np.empty(R, float)
    chat = np.empty(R, float)

    p = 1.0 - eps

    for r in range(R):
        if (progress is not None) and (r % progress == 0):
            print(r)

        x = ar1_gaussian(n, phi=phi, sigma_eps=sigma_eps)

        var_hat = empirical_var(x, p)
        es_hat  = empirical_es(x, p)
        c_hat   = pelve_emp(x, var_hat, es_hat, eps)

        chat[r] = c_hat
        Z[r]    = np.sqrt(n) * (c_hat - c_true)

    return Z, chat, c_true, dist_marg

###############################################################
# RHS-variance via grid
###############################################################
def covW_ar1(u, v, phi, dist_marg, K=300):

    u = float(u)
    v = float(v)


    cov = min(u, v) - u * v

    sigma_stat = dist_marg.std()
    zu = dist_marg.ppf(u) / sigma_stat
    zv = dist_marg.ppf(v) / sigma_stat

    for k in range(1, K + 1):
        rho = phi ** k
        Phi2 = scipy.stats.multivariate_normal.cdf(
            [zu, zv],
            mean=[0.0, 0.0],
            cov=[[1.0, rho], [rho, 1.0]],
        )
        cov += 2.0 * (Phi2 - u * v)

    return cov

def sigma2_rhs_ar1(eps, c_true, phi, sigma_eps, G=300, K=300):
    sigma_stat = sigma_eps / np.sqrt(1.0 - phi**2)
    dist_marg  = scipy.stats.norm(loc=0.0, scale=sigma_stat)

    p = 1.0 - eps
    q = 1.0 - c_true * eps

    t0 = q
    t1 = 1.0 - 1e-6
    tgrid = np.linspace(t0, t1, G)

    dt = (t1 - t0) / (G - 1)
    w = np.full(G, dt)
    w[0] *= 0.5
    w[-1] *= 0.5

    xgrid = dist_marg.ppf(tgrid)
    f_x   = dist_marg.pdf(xgrid)
    g     = 1.0 / (eps * f_x)

    Fp = dist_marg.ppf(p)
    Fq = dist_marg.ppf(q)
    denom = (Fp - Fq)

    a_t = (w * g) / denom

    fp = dist_marg.pdf(Fp)
    a_p = -(c_true / fp) / denom

    pts = np.concatenate([tgrid, np.array([p])])
    m = pts.size

    Sigma = np.empty((m, m), float)
    for i in range(m):
        print(i)
        for j in range(i, m):
            cij = covW_ar1(pts[i], pts[j], phi, dist_marg, K=K)
            Sigma[i, j] = cij
            Sigma[j, i] = cij

    a = np.concatenate([a_t, np.array([a_p])])

    # sigma^2 = a' Σ a
    sigma2 = float(a @ Sigma @ a)
    return sigma2

###############################################################
# plotting
###############################################################
def plot_hist_and_qq_Z(Z, chat, c_true, sigma2_rhs, n, phi, bins=40):
    Z = Z[np.isfinite(Z)]
    muZ  = 0.0
    sigZ = np.sqrt(sigma2_rhs)

    plt.figure(figsize=(6,4))
    plt.hist(Z, bins=bins, density=True, alpha=0.6, edgecolor="black", linewidth=0.8)

    xZ = np.linspace(Z.min(), Z.max(), 400)
    plt.plot(xZ, scipy.stats.norm.pdf(xZ, loc=muZ, scale=sigZ), "r", linewidth=1.2)

    #plt.title(rf"Histogramm von $\sqrt{{n}}(\hat{{c}} - c)$, Gaussian AR(1), $\phi={phi}$")
    plt.xlabel("Z")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    scipy.stats.probplot(Z, dist=scipy.stats.norm(loc=muZ, scale=sigZ), plot=plt)
    plt.title(r"QQ-Plot von $Z$ gegen Normal $N(0,\sigma^2)$")
    plt.tight_layout()
    plt.show()

    chat = chat[np.isfinite(chat)]
    mu    = c_true
    var   = sigma2_rhs / n
    sigma = np.sqrt(var)

    plt.figure(figsize=(7,4.8))
    counts, bin_edges, _ = plt.hist(
        chat, bins=bins, density=False,
        alpha=0.35, edgecolor="black", linewidth=0.8
    )

    binwidth = bin_edges[1] - bin_edges[0]
    xgrid = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y = scipy.stats.norm.pdf(xgrid, loc=mu, scale=sigma) * (chat.size * binwidth)
    plt.plot(xgrid, y, "r", linewidth=1.2)

    #plt.title(rf"Histogramm von $\hat c_n$ (PELVE) Gaussian AR(1) phi={phi} |  N({mu:.2f}, {var:.4f})")
    plt.xlabel(r"$\hat c_n$")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    scipy.stats.probplot(chat, dist=scipy.stats.norm(loc=mu, scale=sigma), plot=plt)
    plt.title(r"QQ-Plot von $\hat c_n$ gegen theoretische Normalverteilung")
    plt.tight_layout()
    plt.show()

###############################################################
# run
###############################################################
n   = 5000
eps = 0.05
R   = 10000

phi       = 0.1   # 0.0, 0.5, 0.8, 0.95
sigma_eps = 1.0

Z, chat, c_true, dist_marg = mc_Z_ar1(
    n=n, eps=eps, R=R,
    phi=phi, sigma_eps=sigma_eps,
    progress=500
)

sigma2_rhs = sigma2_rhs_ar1(
    eps=eps, c_true=c_true,
    phi=phi, sigma_eps=sigma_eps,
    G=600, K=10  # für phi=0.05 reicht kleines K=5; für phi=0.95 sollte K deutlich größer sein
)

Zf = Z[np.isfinite(Z)]
chatf = chat[np.isfinite(chat)]

print(
    "phi =", phi,
    "| c_true =", c_true,
    "| Z-mean =", Zf.mean(),
    " Z-std =", Zf.std(ddof=1),
    "| chat-mean =", chatf.mean(),
    " chat-var =", chatf.var(ddof=1),
)

print(
    "sigma2_rhs (grid) =", sigma2_rhs,
    "| sigma2_MC =", Zf.var(ddof=1)
)

plot_hist_and_qq_Z(
    Z, chat,
    c_true=c_true,
    sigma2_rhs=sigma2_rhs,
    n=n,
    phi=phi,
    bins=30
)