"""
Paired-comparison models and the sister-pair test from the study design.

Each comparison is (i, j, y) with y = 1 when participant i's exposures were
judged riskier ("win" for i), 0 when j's were, and 0.5 for a tie.

- Bradley-Terry: P(i riskier than j) = sigmoid(theta_i - theta_j); ties count
  as half a win for each side.
- Davidson (1970): models ties explicitly,
      P(i) = pi_i / D,  P(j) = pi_j / D,  P(tie) = nu * sqrt(pi_i * pi_j) / D,
      D = pi_i + pi_j + nu * sqrt(pi_i * pi_j),  pi = exp(theta).
  Given a decisive outcome, P(i riskier) is still sigmoid(theta_i - theta_j).

Both are fitted by maximum likelihood with exact gradients and a small ridge
penalty (identifies the scale and keeps estimates finite when a participant
wins or loses every comparison). Scores are centred to mean zero.

Sister-pair test: models are fitted on non-sister comparisons only. If the
judgments carry real signal, the affected sister should score as riskier than
her unaffected sister more often than a random pair would. The statistic is
the mean P(affected riskier than unaffected) over sister pairs; its null
distribution comes from repeatedly drawing the same number of random
non-sister pairs with random orientation.
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

WIN, LOSS, TIE = 1.0, 0.0, 0.5


@dataclass
class Fit:
    model: str
    theta: np.ndarray  # centred log-strengths (higher = riskier)
    nu: float | None = None  # Davidson tie parameter
    n_comparisons: int = 0
    log_likelihood: float = 0.0
    converged: bool = True

    def prob_riskier(self, i, j):
        """P(i judged riskier than j | decisive outcome)."""
        return expit(self.theta[i] - self.theta[j])


def _arrays(comparisons):
    arr = np.asarray(comparisons, dtype=float)
    if arr.size == 0:
        raise ValueError("no comparisons to fit")
    i, j, y = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]
    if not np.isin(y, (WIN, LOSS, TIE)).all():
        raise ValueError("outcomes must be 1 (i riskier), 0 (j riskier) or 0.5 (tie)")
    return i, j, y


def fit_bradley_terry(n, comparisons, ridge=1e-3):
    """Fit Bradley-Terry strengths for `n` participants (ties = half wins)."""
    i, j, y = _arrays(comparisons)

    def objective(theta):
        d = theta[i] - theta[j]
        p = expit(d)
        eps = 1e-12
        ll = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        g_d = y - p  # d ll / d d
        grad = np.bincount(i, g_d, n) - np.bincount(j, g_d, n)
        return -(ll - ridge / 2 * theta @ theta), -(grad - ridge * theta)

    res = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B")
    theta = res.x - res.x.mean()
    return Fit("bradley_terry", theta, None, len(y), -res.fun, bool(res.success))


def fit_davidson(n, comparisons, ridge=1e-3):
    """Fit Davidson strengths and tie parameter nu for `n` participants."""
    i, j, y = _arrays(comparisons)
    w_i = (y == WIN).astype(float)
    w_j = (y == LOSS).astype(float)
    w_t = (y == TIE).astype(float)

    def objective(params):
        theta, phi = params[:n], params[n]
        ti, tj = theta[i], theta[j]
        tt = phi + (ti + tj) / 2
        log_d = logsumexp(np.stack([ti, tj, tt]), axis=0)
        ll = np.sum(w_i * ti + w_j * tj + w_t * tt - log_d)
        p_i, p_j, p_t = np.exp(ti - log_d), np.exp(tj - log_d), np.exp(tt - log_d)
        g_i = w_i + w_t / 2 - (p_i + p_t / 2)
        g_j = w_j + w_t / 2 - (p_j + p_t / 2)
        grad = np.empty(n + 1)
        grad[:n] = np.bincount(i, g_i, n) + np.bincount(j, g_j, n) - ridge * theta
        grad[n] = np.sum(w_t - p_t)
        return -(ll - ridge / 2 * theta @ theta), -grad

    res = minimize(objective, np.zeros(n + 1), jac=True, method="L-BFGS-B")
    theta = res.x[:n] - res.x[:n].mean()
    return Fit("davidson", theta, float(np.exp(res.x[n])), len(y), -res.fun, bool(res.success))


def candidate_pairs(pair_ids):
    """All unordered participant pairs from different families (index pairs)."""
    return [(a, b) for a, b in combinations(range(len(pair_ids)), 2) if pair_ids[a] != pair_ids[b]]


def sister_pairs(pair_ids, affected):
    """(affected index, unaffected index) for every family with known labels."""
    families = {}
    for idx, (pid, aff) in enumerate(zip(pair_ids, affected)):
        families.setdefault(pid, []).append((idx, aff))
    result = []
    for members in families.values():
        a = [idx for idx, aff in members if aff is True]
        u = [idx for idx, aff in members if aff is False]
        if len(a) == 1 and len(u) == 1:
            result.append((a[0], u[0]))
    return result


@dataclass
class SisterTest:
    statistic: float  # mean P(affected riskier than unaffected)
    null_mean: float
    null_sd: float
    p_value: float  # one-sided: affected sisters riskier than random pairs
    n_sister_pairs: int
    per_pair: list = field(default_factory=list)


def sister_pair_test(fit, sisters, null_pairs, n_null=10_000, seed=0):
    """Compare sister-pair probabilities with a random-pair null distribution."""
    if not sisters:
        raise ValueError("no sister pairs with known affected/unaffected labels")
    if not null_pairs:
        raise ValueError("no non-sister pairs for the null distribution")
    rng = np.random.default_rng(seed)

    per_pair = [float(fit.prob_riskier(a, u)) for a, u in sisters]
    statistic = float(np.mean(per_pair))

    m = len(sisters)
    pairs = np.asarray(null_pairs)
    draws = pairs[rng.integers(0, len(pairs), size=(n_null, m))]  # (n_null, m, 2)
    flip = rng.random((n_null, m)) < 0.5
    first = np.where(flip, draws[..., 1], draws[..., 0])
    second = np.where(flip, draws[..., 0], draws[..., 1])
    null = expit(fit.theta[first] - fit.theta[second]).mean(axis=1)

    p_value = (1 + np.sum(null >= statistic)) / (1 + n_null)
    return SisterTest(statistic, float(null.mean()), float(null.std()), float(p_value), m, per_pair)
