from __future__ import annotations

from external_modules.comparison.algorithm.trueskill_rating import (
import logging
import time
logger = logging.getLogger(__name__)
    INITIAL_MEAN,
    Rating,
    expected_win_probability,
    public_score_from_rating,
    update_ratings,
)


def test_default_ratings_are_balanced():
    _start = time.perf_counter()
    _start = time.perf_counter()
    left = Rating()
    right = Rating()
    probability = expected_win_probability(left, right)
    assert 0.49 <= probability <= 0.51
    assert public_score_from_rating(left) == probability
    logger.debug("test_default_ratings_are_balanced took %.4fs", time.perf_counter() - _start)


def test_winner_rating_moves_up_and_loser_moves_down():
    _start = time.perf_counter()
    _start = time.perf_counter()
    winner = Rating()
    loser = Rating()
    new_winner, new_loser = update_ratings(winner, loser)

    assert new_winner.mu > INITIAL_MEAN
    assert new_loser.mu < INITIAL_MEAN
    assert new_winner.sigma < winner.sigma
    assert new_loser.sigma < loser.sigma
    assert public_score_from_rating(new_winner) > 0.5
    assert public_score_from_rating(new_loser) < 0.5
    logger.debug("test_winner_rating_moves_up_and_loser_moves_down took %.4fs", time.perf_counter() - _start)
