from __future__ import annotations

import logging
import time

import pytest

from ..shared.logger import get_logger
from .comparison.algorithm.trueskill_rating import (
    INITIAL_MEAN,
    EPSILON,
    _add_dynamics_noise,
    _clamp_uncertainty,
    Rating,
    normal_cumulative_distribution,
    normal_probability_density,
    rating_from_row,
    expected_win_probability,
    public_score_from_rating,
    update_ratings,
)

logger = get_logger(__name__)


def test_default_ratings_are_balanced() -> None:
    _start = time.perf_counter()
    left = Rating()
    right = Rating()
    probability = expected_win_probability(left, right)
    assert 0.49 <= probability <= 0.51
    assert public_score_from_rating(left) == pytest.approx(0.5, rel=1e-9)


def test_winner_rating_moves_up_and_loser_moves_down() -> None:
    _start = time.perf_counter()
    winner = Rating()
    loser = Rating()
    new_winner, new_loser = update_ratings(winner, loser)

    assert new_winner.mu_skill > INITIAL_MEAN
    assert new_loser.mu_skill < INITIAL_MEAN
    assert new_winner.sigma_uncertainty < winner.sigma_uncertainty
    assert new_loser.sigma_uncertainty < loser.sigma_uncertainty
    assert public_score_from_rating(new_winner) > public_score_from_rating(new_loser)
    assert public_score_from_rating(new_winner) > 0.0


def test_math_helpers_and_row_conversion() -> None:
    _start = time.perf_counter()

    assert normal_probability_density(0.0) == pytest.approx(0.3989422804, rel=1e-6)
    assert normal_cumulative_distribution(0.0) == pytest.approx(0.5, rel=1e-9)
    assert _clamp_uncertainty(0.0) == EPSILON
    assert _add_dynamics_noise(0.0) > EPSILON

    rating = rating_from_row({"rating_mu": "26.5", "rating_sigma": "8.2"})
    assert rating == Rating(mu_skill=26.5, sigma_uncertainty=8.2)
