"""
processing/cleaner.py — Data cleaning and integration pipeline.

Cleans raw patent and research records, normalizes fields,
deduplicates, and produces a unified dataset for NLP analysis.
"""

import re
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def clean_text(text):
    """Normalize text: lowercase, strip extra whitespace, remove special chars."""
    if not text or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_year(year):
    """Ensure year is a valid integer within expected range."""
    try:
        y = int(year)
        if 1990 <= y <= 2030:
            return y
    except (ValueError, TypeError):
        pass
    return None


def clean_documents(df):
    """
    Clean and normalize a DataFrame of raw documents.

    Steps:
    1. Drop rows with missing titles
    2. Clean title and abstract text
    3. Normalize year field
    4. Deduplicate on (source, source_id)
    5. Fill missing fields with appropriate defaults

    Args:
        df: Raw DataFrame from ingestion

    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning %d raw documents...", len(df))
    initial_count = len(df)

    # Drop rows with no title
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip().astype(bool)]

    # Clean text fields
    df["title_clean"] = df["title"].apply(clean_text)
    df["abstract_clean"] = df["abstract"].apply(clean_text)

    # Normalize years
    df["year"] = df["year"].apply(normalize_year)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Deduplicate
    df = df.drop_duplicates(subset=["source", "source_id"], keep="first")

    # Fill nulls
    df["citation_count"] = df["citation_count"].where(
        df["citation_count"].notna(), other=None
    )
    df["organizations"] = df["organizations"].fillna("")
    df["authors"] = df["authors"].fillna("")

    # Combine title and abstract for NLP
    df["text_for_nlp"] = (
        df["title_clean"] + " " + df["abstract_clean"]
    ).str.strip()

    dropped = initial_count - len(df)
    logger.info("Cleaning complete: %d kept, %d dropped", len(df), dropped)

    return df.reset_index(drop=True)


def integrate_sources(papers_df, patents_df):
    """
    Combine papers and patents into a single DataFrame.

    Both DataFrames should already be cleaned. This function
    concatenates them and adds any cross-source fields.

    Args:
        papers_df: Cleaned papers from OpenAlex
        patents_df: Cleaned patents from USPTO

    Returns:
        Integrated DataFrame
    """
    combined = pd.concat([papers_df, patents_df], ignore_index=True)
    logger.info(
        "Integrated %d papers + %d patents = %d total documents",
        len(papers_df), len(patents_df), len(combined)
    )
    return combined
