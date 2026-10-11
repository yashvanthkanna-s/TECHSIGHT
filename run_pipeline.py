"""
run_pipeline.py — Main pipeline runner for TechSight.

Runs the full data pipeline end-to-end:
1. Ingest data (or use mock data)
2. Clean and integrate
3. Run NLP topic discovery
4. Compute yearly indicators
5. Calculate Trend Scores
6. Generate forecasts

Usage: python run_pipeline.py
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run():
    """Execute the full TechSight pipeline."""
    logger.info("=" * 60)
    logger.info("TechSight Pipeline — Starting")
    logger.info("=" * 60)

    # --- Stage 1: Data Ingestion ---
    logger.info("Stage 1: Data Ingestion")
    from mock.mock_data import generate_documents
    documents = generate_documents(max_per_domain=200)
    logger.info("  Loaded %d documents (mock mode)", len(documents))

    # --- Stage 2: Data Cleaning ---
    logger.info("Stage 2: Data Cleaning & Integration")
    from processing.cleaner import clean_documents
    cleaned = clean_documents(documents)
    logger.info("  Cleaned: %d documents ready for analysis", len(cleaned))

    # --- Stage 3: NLP Topic Discovery ---
    logger.info("Stage 3: NLP Topic Discovery")
    from nlp.topic_model import TopicModel
    model = TopicModel()
    model.fit(cleaned, text_column="text_for_nlp")
    topics = model.get_topics()
    labels = model.get_cluster_labels()
    cleaned["cluster_id"] = labels
    logger.info("  Discovered %d topics", len(topics))
    for _, topic in topics.iterrows():
        logger.info("    Cluster %d (%d docs): %s",
                     topic["cluster_id"], topic["document_count"],
                     topic["top_terms"][:60])

    # --- Stage 4: Yearly Indicators ---
    logger.info("Stage 4: Computing Yearly Indicators")
    from mock.mock_data import get_yearly_indicators
    indicators = get_yearly_indicators()
    logger.info("  Computed indicators for %d technology-year pairs", len(indicators))

    # --- Stage 5: Trend Scores ---
    logger.info("Stage 5: Computing Trend Scores")
    from analytics.trend_score import compute_trend_scores
    scores = compute_trend_scores(indicators)
    for _, row in scores.iterrows():
        logger.info("  %s: %.1f (%s)", row["technology_name"],
                     row["score"], row["label"])

    # --- Stage 6: Forecasting ---
    logger.info("Stage 6: Generating Forecasts")
    from forecasting.models import generate_forecasts
    forecasts, metrics = generate_forecasts(indicators)
    logger.info("  Generated %d forecast points", len(forecasts))
    logger.info("  Model evaluation metrics:")
    for _, m in metrics.iterrows():
        logger.info("    %s | %s | %s: MAE=%.1f, RMSE=%.1f, MAPE=%.1f%%",
                     m["technology"], m["series"], m["model"],
                     m["MAE"], m["RMSE"], m["MAPE"])

    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start API:       uvicorn api.main:app --reload --port 8000")
    logger.info("  2. Start Dashboard: streamlit run dashboard/app.py")
    logger.info("")

    return {
        "documents": cleaned,
        "topics": topics,
        "indicators": indicators,
        "scores": scores,
        "forecasts": forecasts,
        "metrics": metrics,
    }


if __name__ == "__main__":
    run()
