"""
analytics/trend_score.py — Technology Trend Score calculation.

Computes a composite 0-100 score measuring how strongly a technology
topic is trending, based on patent and research publication activity.
See docs/trend_score.md for full documentation.
"""

import logging
import numpy as np
import pandas as pd
from config import (
    TREND_SCORE_WEIGHTS, TREND_SCORE_LOOKBACK_YEARS,
    TREND_SCORE_MIN_ACTIVITY, TREND_LABELS,
)

logger = logging.getLogger(__name__)


def _growth_rate(counts):
    """
    Compute year-over-year growth rate.

    Uses compound annual growth rate (CAGR) over the lookback period.
    Returns a raw score (not yet scaled 0-100).
    """
    if len(counts) < 2:
        return 0.0
    start = max(counts[0], 1)  # avoid division by zero
    end = max(counts[-1], 1)
    n_years = len(counts) - 1
    cagr = (end / start) ** (1.0 / n_years) - 1.0
    return cagr * 100  # percentage


def _consistency(counts):
    """
    Measure how many years showed positive growth.

    Returns fraction of years with positive YoY change (0 to 1).
    """
    if len(counts) < 2:
        return 0.0
    yoy_changes = np.diff(counts)
    positive = np.sum(yoy_changes > 0)
    return positive / len(yoy_changes)


def _acceleration(counts):
    """
    Measure whether growth is speeding up or slowing down.

    Compares average growth in the recent half vs. the earlier half.
    Positive = accelerating, negative = decelerating.
    """
    if len(counts) < 4:
        return 0.0
    mid = len(counts) // 2
    early_growth = np.mean(np.diff(counts[:mid + 1]))
    late_growth = np.mean(np.diff(counts[mid:]))
    if abs(early_growth) < 0.001:
        return late_growth * 10  # large acceleration from near-zero
    return ((late_growth - early_growth) / abs(early_growth)) * 100


def _activity_volume(total_activity):
    """
    Log-scaled activity volume to prevent large topics from dominating.
    """
    if total_activity <= 0:
        return 0.0
    return np.log1p(total_activity) * 10


def _citation_signal(avg_citations):
    """
    Citation trend signal: positive if recent citations are growing.
    """
    if len(avg_citations) < 2:
        return 50.0  # neutral if insufficient data
    valid = [c for c in avg_citations if c is not None and not np.isnan(c)]
    if len(valid) < 2:
        return 50.0
    mid = len(valid) // 2
    early_avg = np.mean(valid[:mid + 1])
    late_avg = np.mean(valid[mid:])
    if early_avg < 0.001:
        return 70.0 if late_avg > 0 else 50.0
    change = ((late_avg - early_avg) / early_avg) * 100
    return min(100, max(0, 50 + change))


def _min_max_scale(values):
    """Scale a list of values to 0-100."""
    arr = np.array(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 0.001:
        return np.full_like(arr, 50.0)
    return ((arr - vmin) / (vmax - vmin)) * 100


def _assign_label(score):
    """Assign a trend label based on score thresholds."""
    for label, (low, high) in TREND_LABELS.items():
        if low <= score <= high:
            return label.title()
    return "Stable"


def compute_trend_scores(indicators_df):
    """
    Compute Trend Scores for all technologies.

    Args:
        indicators_df: DataFrame with columns:
            technology_id, technology_name, year,
            patent_count, paper_count, total_count, avg_citations

    Returns:
        DataFrame with trend scores and component values for each technology.
    """
    logger.info("Computing trend scores...")

    # Get unique technologies
    tech_ids = indicators_df["technology_id"].unique()
    results = []

    # Collect raw component values first (for cross-topic scaling)
    raw_components = []

    for tech_id in tech_ids:
        tech_data = indicators_df[
            indicators_df["technology_id"] == tech_id
        ].sort_values("year")

        tech_name = tech_data["technology_name"].iloc[0]

        # Use lookback period
        recent = tech_data.tail(TREND_SCORE_LOOKBACK_YEARS)
        counts = recent["total_count"].values.astype(float)
        total = counts.sum()
        avg_cites = recent["avg_citations"].values

        # Check minimum activity threshold
        if total < TREND_SCORE_MIN_ACTIVITY:
            logger.info("  %s: below minimum activity (%d < %d)",
                        tech_name, total, TREND_SCORE_MIN_ACTIVITY)
            results.append({
                "technology_id": tech_id,
                "technology_name": tech_name,
                "score": 0.0,
                "label": "Insufficient Data",
                "growth_rate": 0, "consistency": 0,
                "acceleration": 0, "activity_volume": 0,
                "citation_signal": 0,
            })
            continue

        raw = {
            "technology_id": tech_id,
            "technology_name": tech_name,
            "growth_rate": _growth_rate(counts),
            "consistency": _consistency(counts) * 100,
            "acceleration": _acceleration(counts),
            "activity_volume": _activity_volume(total),
            "citation_signal": _citation_signal(avg_cites),
        }
        raw_components.append(raw)

    if not raw_components:
        return pd.DataFrame(results)

    # Scale components across topics
    raw_df = pd.DataFrame(raw_components)
    components = ["growth_rate", "consistency", "acceleration",
                   "activity_volume", "citation_signal"]

    for comp in components:
        raw_df[f"{comp}_scaled"] = _min_max_scale(raw_df[comp].values)

    # Compute weighted score
    weights = TREND_SCORE_WEIGHTS
    for _, row in raw_df.iterrows():
        score = sum(
            row[f"{comp}_scaled"] * weights[comp]
            for comp in components
        )
        score = min(100, max(0, score))

        results.append({
            "technology_id": row["technology_id"],
            "technology_name": row["technology_name"],
            "score": round(score, 1),
            "label": _assign_label(score),
            "growth_rate": round(row["growth_rate_scaled"], 1),
            "consistency": round(row["consistency_scaled"], 1),
            "acceleration": round(row["acceleration_scaled"], 1),
            "activity_volume": round(row["activity_volume_scaled"], 1),
            "citation_signal": round(row["citation_signal_scaled"], 1),
        })

    result_df = pd.DataFrame(results)
    logger.info("Trend scores computed for %d technologies", len(result_df))
    return result_df
