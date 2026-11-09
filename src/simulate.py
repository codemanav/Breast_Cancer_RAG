"""
Simulate the sister-pair study to validate the statistics and estimate power.

Each participant has a latent risk score:
    theta = family + individual + effect * affected
with family ~ N(0, family_sd) shared by sisters and individual ~ N(0, 1).
Judgments for every non-sister pair are drawn from a Davidson model with that
theta and tie parameter nu, i.e. a judge whose calls are noisy but carry
`effect` standard deviations of real signal about the affected sister.
"""

from dataclasses import dataclass

import numpy as np

from src.ranking import WIN, LOSS, TIE, candidate_pairs, fit_davidson, sister_pair_test, sister_pairs


@dataclass
class Study:
    pair_ids: list
    affected: list
    theta: np.ndarray
    comparisons: list  # (i, j, outcome) for non-sister pairs


def simulate_study(n_pairs=30, effect=1.0, family_sd=1.0, nu=0.5, seed=0):
    rng = np.random.default_rng(seed)
    pair_ids = [f"S{k:02d}" for k in range(n_pairs) for _ in range(2)]
    affected = [member == 0 for _ in range(n_pairs) for member in range(2)]
    family = np.repeat(rng.normal(0, family_sd, n_pairs), 2)
    theta = family + rng.normal(0, 1, 2 * n_pairs) + effect * np.array(affected, dtype=float)

    comparisons = []
    for i, j in candidate_pairs(pair_ids):
        weights = np.array([np.exp(theta[i]), np.exp(theta[j]), nu * np.exp((theta[i] + theta[j]) / 2)])
        outcome = rng.choice([WIN, LOSS, TIE], p=weights / weights.sum())
        comparisons.append((i, j, outcome))
    return Study(pair_ids, affected, theta, comparisons)


def power(n_pairs=30, effect=1.0, n_sims=200, alpha=0.05, n_null=2000, seed=0, **kwargs):
    """Share of simulated studies in which the sister-pair test rejects at `alpha`."""
    rejections = 0
    for s in range(n_sims):
        study = simulate_study(n_pairs, effect, seed=seed + s, **kwargs)
        fit = fit_davidson(len(study.pair_ids), study.comparisons)
        test = sister_pair_test(
            fit,
            sister_pairs(study.pair_ids, study.affected),
            candidate_pairs(study.pair_ids),
            n_null=n_null,
            seed=seed + s,
        )
        rejections += test.p_value < alpha
    return rejections / n_sims
