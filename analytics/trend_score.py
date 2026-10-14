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


def classify_publication_trend(yearly_df, min_activity=10):
    """
    Classify research trend based on actual annual publication growth rate (CAGR),
    rather than directly from the composite score.

    Thresholds:
      - High Growth:     CAGR > 20%
      - Moderate Growth: 5% to 20%
      - Stable:          -5% to 5%
      - Low Growth:      -20% to -5%
      - Declining:       CAGR < -20%
      - Insufficient Data: Total count < min_activity
    """
    disclaimer = "Important: A research trend classification describes publication activity, not commercial success or technology quality."

    if yearly_df is None or yearly_df.empty:
        return {
            "label": "Insufficient Data",
            "headline": "Insufficient historical data",
            "subtext": "No publication records available for this query.",
            "explanation": "No publication data was found across the observation window to classify a research trend.",
            "cagr_pct": None,
            "total_papers": 0,
            "start_year": None,
            "end_year": None,
            "start_count": 0,
            "end_count": 0,
            "disclaimer": disclaimer,
            "badge_bg": "#f1f5f9",
            "badge_color": "#475569",
            "badge_border": "#e2e8f0",
        }

    count_col = "paper_count" if "paper_count" in yearly_df.columns else "total_count"
    df_sorted = yearly_df[yearly_df["year"] >= 2018].sort_values("year").copy() if "year" in yearly_df.columns else yearly_df.copy()
    if df_sorted.empty:
        df_sorted = yearly_df.sort_values("year").copy()

    total_papers = int(df_sorted[count_col].sum())

    if total_papers < min_activity:
        return {
            "label": "Insufficient Data",
            "headline": "Insufficient reliable data to classify trend",
            "subtext": "Retrieved volume is below the statistical threshold.",
            "explanation": f"Total retrieved volume ({total_papers} works) is below the reliable threshold of {min_activity} publications across the observation period. TECHSIGHT requires adequate sample depth before assigning a trajectory classification.",
            "cagr_pct": None,
            "total_papers": total_papers,
            "start_year": int(df_sorted["year"].iloc[0]) if not df_sorted.empty else 2018,
            "end_year": int(df_sorted["year"].iloc[-1]) if not df_sorted.empty else 2026,
            "start_count": int(df_sorted[count_col].iloc[0]) if not df_sorted.empty else 0,
            "end_count": int(df_sorted[count_col].iloc[-1]) if not df_sorted.empty else 0,
            "disclaimer": disclaimer,
            "badge_bg": "#f8fafc",
            "badge_color": "#64748b",
            "badge_border": "#cbd5e1",
        }

    counts = df_sorted[count_col].values.astype(float)
    n = len(counts)
    start_year = int(df_sorted["year"].iloc[0])
    end_year = int(df_sorted["year"].iloc[-1])
    start_val = max(counts[0], 1.0)
    end_val = max(counts[-1], 1.0)

    if n >= 2:
        cagr = ((end_val / start_val) ** (1.0 / (n - 1)) - 1.0) * 100.0
    else:
        cagr = 0.0

    cagr_rounded = round(cagr, 1)

    if cagr > 20.0:
        label = "High Growth"
        headline = "Research activity is rising rapidly"
        subtext = "Publication growth is strong compared with the chosen historical period."
        explanation = f"Assigned High Growth because publication volume grew at an annualized rate of +{cagr_rounded:.1f}% (expanding from {int(start_val):,} to {int(end_val):,} papers between {start_year} and {end_year}), exceeding the +20% threshold."
        badge_bg = "#ecfdf5"
        badge_color = "#047857"
        badge_border = "#a7f3d0"
    elif cagr >= 5.0:
        label = "Moderate Growth"
        headline = "Research activity is increasing"
        subtext = "Publication activity is growing, but not as rapidly as in the high-growth category."
        explanation = f"Assigned Moderate Growth because publication volume grew at an annualized rate of +{cagr_rounded:.1f}% (between the +5% and +20% thresholds) between {start_year} and {end_year}."
        badge_bg = "#eff6ff"
        badge_color = "#1d4ed8"
        badge_border = "#bfdbfe"
    elif cagr >= -5.0:
        label = "Stable"
        headline = "Research activity is relatively steady"
        subtext = "Publication counts show little overall change."
        explanation = f"Assigned Stable because publication output exhibited minimal net fluctuation ({cagr_rounded:+.1f}% annual change, within the −5% to +5% band) between {start_year} and {end_year}."
        badge_bg = "#f1f5f9"
        badge_color = "#334155"
        badge_border = "#cbd5e1"
    elif cagr >= -20.0:
        label = "Low Growth"
        headline = "Research activity is growing slowly"
        subtext = "Publication growth is positive but comparatively weak over the selected period."
        explanation = f"Assigned Low Growth because publication rate changed by {cagr_rounded:+.1f}% annually (within the −20% to −5% range) between {start_year} and {end_year}."
        badge_bg = "#fefce8"
        badge_color = "#854d0e"
        badge_border = "#fef08a"
    else:
        label = "Declining"
        headline = "Research activity is decreasing"
        subtext = "Publication counts are falling over the selected comparison period."
        explanation = f"Assigned Declining because publication volume contracted at an annualized rate of {cagr_rounded:+.1f}% (below the −20% threshold) between {start_year} and {end_year}."
        badge_bg = "#fef2f2"
        badge_color = "#b91c1c"
        badge_border = "#fecaca"

    return {
        "label": label,
        "headline": headline,
        "subtext": subtext,
        "explanation": explanation,
        "cagr_pct": cagr_rounded,
        "total_papers": total_papers,
        "start_year": start_year,
        "end_year": end_year,
        "start_count": int(start_val),
        "end_count": int(end_val),
        "disclaimer": disclaimer,
        "badge_bg": badge_bg,
        "badge_color": badge_color,
        "badge_border": badge_border,
    }

