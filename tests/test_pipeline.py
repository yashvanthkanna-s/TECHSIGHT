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
        assert df["year"].max() <= 2026

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
        assert all(y > 2026 for y in df["year"])


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
        assert "epo_espacenet_url" in gap
        assert "worldwide.espacenet.com" in gap["epo_espacenet_url"]


class TestResearchInsights:
    """Tests for Research Insights generation."""

    def test_generate_research_insights_normal(self):
        from analytics.insights import generate_research_insights
        df_y = pd.DataFrame({
            "year": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
            "paper_count": [1000, 1200, 1500, 1900, 2400, 3100, 4000, 5200, 6800],
            "patent_count": [200, 250, 310, 400, 520, 680, 890, 1150, 1500],
        })
        topics = {
            "has_topics": True,
            "k": 2,
            "topics": [
                {"cluster_id": 1, "theme": "Hardware Efficiency", "keywords": "tensor, silicon, low-rank", "doc_count": 12},
                {"cluster_id": 2, "theme": "Algorithm Optimization", "keywords": "gradient, loss, convergence", "doc_count": 8},
            ],
        }
        sample_docs = [
            {"id": "W123", "title": "Scalable Tensor Architectures", "citations": 450, "year": 2024, "doi": "https://doi.org/10.123/456"},
        ]
        fc = pd.DataFrame({"year": [2027], "predicted_value": [8500], "lower_bound": [7200], "upper_bound": [9800]})
        gap = {"epo_espacenet_url": "https://worldwide.espacenet.com/patent/search?q=AI", "status": "Active Translation Phase"}

        insights = generate_research_insights("AI", df_y, topics, sample_docs, fc, gap, is_live_mode=True)
        assert "what_happened" in insights
        assert "which_areas_stand_out" in insights
        assert "what_might_happen_next" in insights
        assert "what_to_explore_next" in insights

        assert insights["what_happened"]["status"] == "normal"
        assert insights["what_happened"]["pct_change"] > 0
        assert insights["which_areas_stand_out"]["status"] == "clustered"
        assert len(insights["which_areas_stand_out"]["clusters"]) == 2
        assert insights["what_might_happen_next"]["status"] == "modeled"
        assert len(insights["what_to_explore_next"]["recommended_papers"]) > 0

    def test_generate_research_insights_sparse_data(self):
        from analytics.insights import generate_research_insights
        df_y = pd.DataFrame({"year": [2022, 2023], "paper_count": [2, 3]})
        insights = generate_research_insights("RareTech", df_y, {"has_topics": False}, [], pd.DataFrame(), {}, is_live_mode=True)
        assert insights["what_happened"]["status"] == "low_sample"
        assert "sparse" in insights["what_happened"]["headline"].lower() or "insufficient" in insights["what_happened"]["headline"].lower()
        assert insights["which_areas_stand_out"]["status"] == "insufficient_abstracts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


