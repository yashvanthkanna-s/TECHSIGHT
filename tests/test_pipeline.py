"""
tests/test_pipeline.py — Test suite for TechSight pipeline.

Tests core components: data cleaning, NLP, trend scores, and forecasting.
Uses mock data — no database or API keys required.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd


class TestMockData:
    """Tests for mock data generation."""

    def test_generate_documents(self):
        from mock.mock_data import generate_documents
        df = generate_documents(max_per_domain=50)
        assert len(df) > 0
        assert "title" in df.columns
        assert "source" in df.columns
        assert set(df["source"].unique()) == {"openalex", "uspto"}

    def test_yearly_indicators(self):
        from mock.mock_data import get_yearly_indicators
        df = get_yearly_indicators()
        assert len(df) > 0
        assert "year" in df.columns
        assert df["year"].min() >= 2012
        assert df["year"].max() <= 2024

    def test_technologies(self):
        from mock.mock_data import get_technologies
        df = get_technologies()
        assert len(df) == 3  # 3 domains

    def test_trend_scores(self):
        from mock.mock_data import get_trend_scores
        df = get_trend_scores()
        assert len(df) == 3
        assert all(0 <= s <= 100 for s in df["score"])

    def test_forecasts(self):
        from mock.mock_data import get_forecasts
        df = get_forecasts()
        assert len(df) > 0
        assert all(y > 2024 for y in df["year"])


class TestProcessing:
    """Tests for data cleaning pipeline."""

    def test_clean_text(self):
        from processing.cleaner import clean_text
        assert clean_text("  Hello World!  ") == "hello world"
        assert clean_text(None) == ""
        assert clean_text("") == ""

    def test_normalize_year(self):
        from processing.cleaner import normalize_year
        assert normalize_year(2020) == 2020
        assert normalize_year("2020") == 2020
        assert normalize_year("abc") is None
        assert normalize_year(1800) is None

    def test_clean_documents(self):
        from processing.cleaner import clean_documents
        from mock.mock_data import generate_documents
        df = generate_documents(max_per_domain=20)
        cleaned = clean_documents(df)
        assert len(cleaned) > 0
        assert "text_for_nlp" in cleaned.columns
        assert cleaned["year"].dtype in [np.int32, np.int64]


class TestAnalytics:
    """Tests for trend score computation."""

    def test_compute_trend_scores(self):
        from analytics.trend_score import compute_trend_scores
        from mock.mock_data import get_yearly_indicators
        indicators = get_yearly_indicators()
        scores = compute_trend_scores(indicators)
        assert len(scores) == 3
        assert all(0 <= s <= 100 for s in scores["score"])
        assert all(label in ["Emerging", "Growing", "Stable", "Declining", "Insufficient Data"]
                   for label in scores["label"])


class TestForecasting:
    """Tests for forecasting models."""

    def test_moving_average(self):
        from forecasting.models import moving_average_forecast
        values = [10, 20, 30, 40, 50]
        preds = moving_average_forecast(values, window=3, horizon=2)
        assert len(preds) == 2
        assert all(p > 0 for p in preds)

    def test_linear_regression(self):
        from forecasting.models import linear_regression_forecast
        years = [2020, 2021, 2022, 2023, 2024]
        values = [10, 20, 30, 40, 50]
        preds, model = linear_regression_forecast(
            years, values, [2025, 2026]
        )
        assert len(preds) == 2
        assert preds[0] > values[-1]  # Upward trend should continue

    def test_evaluate_model(self):
        from forecasting.models import evaluate_model
        actual = [10, 20, 30]
        predicted = [12, 18, 32]
        metrics = evaluate_model(actual, predicted)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert metrics["MAE"] > 0


class TestLiveSearch:
    """Tests for live search, topic discovery safety, and gap analysis."""

    def test_discover_live_topics_small_dataset(self):
        from analytics.live_search import discover_live_topics
        # Empty docs
        res_empty = discover_live_topics([])
        assert res_empty["has_topics"] is False

        # Too few docs
        res_few = discover_live_topics([{"text": "Short paper title and abstract text."}])
        assert res_few["has_topics"] is False
        assert "Insufficient" in res_few["reason"]

    def test_calculate_live_trend_score_and_forecast(self):
        from analytics.live_search import calculate_live_trend_score, generate_live_forecast
        df_y = pd.DataFrame({
            "year": [2020, 2021, 2022, 2023, 2024],
            "paper_count": [100, 150, 220, 310, 430],
            "growth_pct": [0.0, 50.0, 46.7, 40.9, 38.7],
        })
        score = calculate_live_trend_score(df_y)
        assert 0 <= score["score"] <= 100
        assert score["label"] in ["Emerging", "Growing", "Stable", "Declining"]

        fc = generate_live_forecast(df_y, horizon=3)
        assert len(fc) == 3
        assert list(fc["year"]) == [2025, 2026, 2027]
        assert all(fc["predicted_value"] > 0)
        assert all(fc["upper_bound"] >= fc["predicted_value"])
        assert all(fc["lower_bound"] <= fc["predicted_value"])

    def test_analyze_research_patent_gap(self):
        from analytics.live_search import analyze_research_patent_gap
        df_y = pd.DataFrame({
            "year": [2020, 2021, 2022, 2023, 2024],
            "paper_count": [100, 200, 400, 800, 1600],
            "growth_pct": [0.0, 100.0, 100.0, 100.0, 100.0],
        })
        gap = analyze_research_patent_gap(df_y, "Graphene")
        assert "status" in gap
        assert "insight" in gap
        assert "google_patents_url" in gap
        assert "https://patents.google.com/?q=Graphene" in gap["google_patents_url"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

