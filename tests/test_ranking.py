import numpy as np
import pytest
from scipy.optimize import approx_fprime

from src import ranking
from src.ranking import (
    LOSS,
    TIE,
    WIN,
    candidate_pairs,
    fit_bradley_terry,
    fit_davidson,
    sister_pair_test,
    sister_pairs,
)
from src.simulate import power, simulate_study


def test_candidate_pairs_matches_study_design():
    """30 sister pairs -> C(60, 2) = 1770 pairs, minus 30 sister pairs = 1740."""
    pair_ids = [f"S{k}" for k in range(30) for _ in range(2)]
    pairs = candidate_pairs(pair_ids)

    assert len(pairs) == 1740
    assert all(pair_ids[a] != pair_ids[b] for a, b in pairs)


def test_sister_pairs_orient_affected_first_and_skip_unlabelled():
    pair_ids = ["S1", "S1", "S2", "S2", "S3", "S3"]
    affected = [False, True, None, None, True, True]  # S2 unknown, S3 invalid

    assert sister_pairs(pair_ids, affected) == [(1, 0)]


def test_bradley_terry_recovers_true_order():
    study = simulate_study(n_pairs=30, effect=0.0, seed=1)
    fit = fit_bradley_terry(60, study.comparisons)

    assert fit.converged
    assert abs(fit.theta.mean()) < 1e-9
    assert np.corrcoef(fit.theta, study.theta)[0, 1] > 0.95


def test_davidson_recovers_order_and_tie_rate():
    study = simulate_study(n_pairs=30, effect=0.0, nu=0.8, seed=2)
    fit = fit_davidson(60, study.comparisons)

    assert fit.converged
    assert np.corrcoef(fit.theta, study.theta)[0, 1] > 0.95
    assert fit.nu == pytest.approx(0.8, rel=0.15)


def test_davidson_gradient_is_exact(monkeypatch):
    """Compare the analytic gradient with finite differences."""
    study = simulate_study(n_pairs=4, effect=0.5, seed=3)
    captured = {}
    original = ranking.minimize

    def spy(fun, x0, jac, method):
        captured["fun"] = fun
        return original(fun, x0, jac=jac, method=method)

    monkeypatch.setattr(ranking, "minimize", spy)
    fit_davidson(8, study.comparisons)

    point = np.random.default_rng(0).normal(size=9)
    analytic = captured["fun"](point)[1]
    numeric = approx_fprime(point, lambda p: captured["fun"](p)[0], 1e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_all_ties_give_equal_strengths():
    comparisons = [(0, 1, TIE), (1, 2, TIE), (0, 2, TIE)]

    fit = fit_davidson(3, comparisons)

    np.testing.assert_allclose(fit.theta, 0, atol=1e-6)
    assert fit.nu > 10  # ties dominate


def test_perfect_record_stays_finite():
    """A participant who wins every comparison must not diverge to infinity."""
    comparisons = [(0, 1, WIN), (0, 2, WIN), (1, 2, LOSS)]

    fit = fit_bradley_terry(3, comparisons)

    assert np.isfinite(fit.theta).all()
    assert fit.theta[0] == fit.theta.max()


def test_rejects_invalid_outcomes():
    with pytest.raises(ValueError, match="outcomes must be"):
        fit_bradley_terry(2, [(0, 1, 2.0)])
    with pytest.raises(ValueError, match="no comparisons"):
        fit_davidson(2, [])


def _test(study, n_null=2000):
    fit = fit_davidson(len(study.pair_ids), study.comparisons)
    return sister_pair_test(
        fit, sister_pairs(study.pair_ids, study.affected), candidate_pairs(study.pair_ids), n_null=n_null
    )


def test_sister_test_detects_real_signal():
    result = _test(simulate_study(n_pairs=30, effect=1.5, seed=4))

    assert result.statistic > 0.65
    assert result.p_value < 0.01
    assert result.null_mean == pytest.approx(0.5, abs=0.02)


def test_sister_test_is_calibrated_without_signal():
    """With no effect, p < 0.05 should happen about 5% of the time."""
    rate = power(n_pairs=20, effect=0.0, n_sims=200, n_null=500, seed=100)

    assert rate == pytest.approx(0.05, abs=0.04)


def test_sister_test_requires_labels():
    study = simulate_study(n_pairs=3, seed=5)
    fit = fit_davidson(6, study.comparisons)

    with pytest.raises(ValueError, match="no sister pairs"):
        sister_pair_test(fit, [], candidate_pairs(study.pair_ids))
