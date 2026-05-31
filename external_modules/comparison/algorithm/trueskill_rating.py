"""
TrueSkill rating system for pairwise image ranking.

Implements a simplified two-player TrueSkill update without draws,
along with utility functions for computing win probabilities and
converting between database rows and rating objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default initialisation constants (Microsoft's TrueSkill defaults)
# ---------------------------------------------------------------------------

INITIAL_MEAN = 25.0
"""Mean rating assigned to a new, unrated entity."""

INITIAL_UNCERTAINTY = INITIAL_MEAN / 3.0
"""Standard deviation (uncertainty) for a new rating."""

PERFORMANCE_VARIATION = INITIAL_MEAN / 6.0
"""Skill-class width – how much performance varies from game to game."""

DYNAMICS_NOISE = INITIAL_MEAN / 300.0
"""Dynamics factor – small additive noise per match to prevent sigma->0."""

EPSILON = 1e-9
"""Small numerical guard used to avoid division by zero / log-of-zero."""


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rating:
    """A player's skill estimate expressed as a Gaussian (mu, sigma).

    Attributes:
        mu:    The inferred mean skill.
        sigma: The standard deviation (uncertainty) around that mean.
    """

    mu: float = INITIAL_MEAN
    sigma: float = INITIAL_UNCERTAINTY


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------


def normal_probability_density(x: float) -> float:
    """Probability density function of the standard normal distribution."""

    result = exp(-(x * x) / 2.0) / sqrt(2.0 * pi)

    return result


def normal_cumulative_distribution(x: float) -> float:
    """Cumulative distribution function of the standard normal distribution."""

    result = 0.5 * (1.0 + erf(x / sqrt(2.0)))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp_uncertainty(uncertainty: float) -> float:
    """Clamp uncertainty to at least EPSILON to avoid degenerate values."""

    result = max(uncertainty, EPSILON)

    return result


def _add_dynamics_noise(uncertainty: float) -> float:
    """Add dynamics noise (DYNAMICS_NOISE) to uncertainty before an update."""

    result = sqrt((_clamp_uncertainty(uncertainty) ** 2) + (DYNAMICS_NOISE**2))

    return result


# ---------------------------------------------------------------------------
# Public TrueSkill operations
# ---------------------------------------------------------------------------


def update_ratings(winner: Rating, loser: Rating) -> tuple[Rating, Rating]:
    """Compute the new ratings after a single match where winner beat loser.

    This is a simplified two-player TrueSkill update (no draws).
    Both ratings are updated in-place using the standard update equations.

    Args:
        winner: The Rating of the winning player before the match.
        loser:  The Rating of the losing player before the match.

    Returns:
        A tuple (new_winner_rating, new_loser_rating).
    """

    winner_uncertainty = _add_dynamics_noise(winner.sigma)
    loser_uncertainty = _add_dynamics_noise(loser.sigma)

    combined_variance = (
        (2.0 * (PERFORMANCE_VARIATION**2))
        + (winner_uncertainty**2)
        + (loser_uncertainty**2)
    )
    combined_deviation = sqrt(max(combined_variance, EPSILON))

    mean_difference = winner.mu - loser.mu
    normalised_difference = mean_difference / combined_deviation

    cumulative_probability = max(
        normal_cumulative_distribution(normalised_difference), EPSILON
    )
    skill_adjustment_weight = (
        normal_probability_density(normalised_difference) / cumulative_probability
    )
    variance_adjustment_weight = skill_adjustment_weight * (
        skill_adjustment_weight + normalised_difference
    )

    winner_variance = winner_uncertainty**2
    loser_variance = loser_uncertainty**2

    winner_new_mean = (
        winner.mu + (winner_variance / combined_deviation) * skill_adjustment_weight
    )
    loser_new_mean = (
        loser.mu - (loser_variance / combined_deviation) * skill_adjustment_weight
    )

    winner_new_variance = winner_variance * max(
        1.0 - (winner_variance / combined_variance) * variance_adjustment_weight,
        EPSILON,
    )
    loser_new_variance = loser_variance * max(
        1.0 - (loser_variance / combined_variance) * variance_adjustment_weight,
        EPSILON,
    )

    result = (
        Rating(mu=winner_new_mean, sigma=sqrt(winner_new_variance)),
        Rating(mu=loser_new_mean, sigma=sqrt(loser_new_variance)),
    )

    return result


def expected_win_probability(first_rating: Rating, second_rating: Rating) -> float:
    """Probability that first_rating beats second_rating given current ratings.

    Uses the pairwise win-probability formula from the TrueSkill model.

    Args:
        first_rating:  The Rating of the first (reference) player.
        second_rating: The Rating of the opposing player.

    Returns:
        A float in [0, 1] representing the first player's win probability.
    """

    denominator = sqrt(
        (2.0 * (PERFORMANCE_VARIATION**2))
        + (_clamp_uncertainty(first_rating.sigma) ** 2)
        + (_clamp_uncertainty(second_rating.sigma) ** 2)
    )
    result = min(
        1.0,
        max(
            0.0,
            normal_cumulative_distribution(
                (first_rating.mu - second_rating.mu) / max(denominator, EPSILON)
            ),
        ),
    )

    return result


def public_score_from_rating(rating: Rating) -> float:
    """Convert a (mu, sigma) rating into a single scalar in [0, 1].

    The result is the win probability against a fresh player with the default
    initial rating (INITIAL_MEAN, INITIAL_UNCERTAINTY).
    Higher values indicate stronger skill.

    Args:
        rating: The Rating to convert.

    Returns:
        A float between 0 and 1 representing the public-facing score.
    """

    result = expected_win_probability(
        rating, Rating(mu=INITIAL_MEAN, sigma=INITIAL_UNCERTAINTY)
    )

    return result


def rating_from_row(row: dict) -> Rating:
    """Deserialise a Rating from a database row (dictionary).

    Keys expected:
        - rating_mu
        - rating_sigma

    Args:
        row: A dict representing one database row.

    Returns:
        A Rating object populated from the row data.
    """

    result = Rating(
        mu=float(row["rating_mu"]),
        sigma=float(row["rating_sigma"]),
    )

    return result
