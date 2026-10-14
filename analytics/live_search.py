"""
analytics/live_search.py — Live On-Demand Technology Intelligence Engine.

Executes live queries against OpenAlex API, runs unsupervised TF-IDF + KMeans
clustering on real abstracts, calculates dynamic trend scores and forecasts,
and performs Research-Patent Gap Analysis.
"""

import time
import logging
import urllib.parse
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from config import (
    OPENALEX_API_KEY, OPENALEX_EMAIL,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
)
from ingestion.epo_connector import (
    is_epo_configured,
    fetch_epo_patent_counts,
    fetch_epo_sample_patents,
)


logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index):
    """Safely reconstruct abstract string from OpenAlex inverted index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
    except Exception:
        return None


def fetch_live_yearly_counts(query, start_year=2018, end_year=2026):
    """
    Query OpenAlex group_by=publication_year to retrieve genuine aggregate
    publication volume per year.

    Returns:
        DataFrame with columns: ['year', 'paper_count', 'growth_pct']
    """
    clean_query = query.strip()
    encoded = urllib.parse.quote(clean_query)
    url = f"{OPENALEX_BASE}?filter=default.search:%22{encoded}%22&group_by=publication_year"
    if OPENALEX_API_KEY:
        url += f"&api_key={OPENALEX_API_KEY}"
    if OPENALEX_EMAIL:
        url += f"&mailto={OPENALEX_EMAIL}"

    counts_by_year = {}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("group_by", []):
                    key = str(item.get("key", ""))
                    if key.isdigit():
                        y = int(key)
                        if start_year <= y <= end_year:
                            counts_by_year[y] = int(item.get("count", 0))
                break
            time.sleep(RETRY_DELAY)
        except Exception as e:
            logger.warning("OpenAlex yearly counts fetch failed attempt %d: %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)

    # Ensure all years in range are present
    rows = []
    for y in range(start_year, end_year + 1):
        c = counts_by_year.get(y, 0)
        rows.append({"year": y, "paper_count": c})

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    # Calculate YoY growth
    df["growth_pct"] = 0.0
    for i in range(1, len(df)):
        prev = max(df.loc[i - 1, "paper_count"], 1)
        curr = df.loc[i, "paper_count"]
        df.loc[i, "growth_pct"] = round(((curr - prev) / prev) * 100, 1)

    return df


def fetch_live_sample_documents(query, limit=30):
    """
    Fetch a sample batch of actual paper titles and abstracts from OpenAlex
    for topic modeling.

    Returns:
        List of dicts with 'id', 'title', 'abstract', 'year', 'citations', 'doi'
    """
    clean_query = query.strip()
    encoded = urllib.parse.quote(clean_query)
    url = (
        f"{OPENALEX_BASE}?filter=default.search:%22{encoded}%22"
        f"&per_page={limit}&select=id,title,abstract_inverted_index,publication_year,cited_by_count,doi"
    )
    if OPENALEX_API_KEY:
        url += f"&api_key={OPENALEX_API_KEY}"
    if OPENALEX_EMAIL:
        url += f"&mailto={OPENALEX_EMAIL}"

    records = []
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            for w in data.get("results", []):
                title = w.get("title") or "Untitled"
                abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
                records.append({
                    "id": w.get("id", ""),
                    "title": title,
                    "abstract": abstract,
                    "year": w.get("publication_year"),
                    "citations": w.get("cited_by_count", 0),
                    "doi": w.get("doi") or "",
                    "text": f"{title}. {abstract if abstract else ''}".strip(),
                })
    except Exception as e:
        logger.error("Failed to fetch OpenAlex sample papers: %s", e)

    return records


def discover_live_topics(documents, max_k=4):
    """
    Perform unsupervised TF-IDF + KMeans clustering on retrieved documents.
    Safely handles empty abstracts and small datasets.

    Returns:
        Dict with 'has_topics', 'k', 'topics': [ {'cluster_id', 'theme', 'keywords', 'doc_count'} ]
    """
    if not documents:
        return {"has_topics": False, "reason": "No documents retrieved for this query.", "topics": []}

    clean_docs = [d for d in documents if d["text"] and len(d["text"]) > 20]

    if len(clean_docs) < 4:
        return {
            "has_topics": False,
            "reason": f"Insufficient abstract volume for semantic clustering ({len(clean_docs)} valid documents).",
            "topics": [],
        }

    # Dynamically select k based on available document count
    k = min(max_k, max(2, len(clean_docs) // 4))

    texts = [d["text"] for d in clean_docs]

    try:
        vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)

        topics = []
        for cluster_id in range(k):
            centroid = kmeans.cluster_centers_[cluster_id]
            top_indices = centroid.argsort()[::-1][:4]
            top_terms = [feature_names[i] for i in top_indices if i < len(feature_names)]
            cluster_docs_count = int(np.sum(labels == cluster_id))

            theme_name = " & ".join(top_terms[:2]).title() if top_terms else f"Topic {cluster_id + 1}"

            topics.append({
                "cluster_id": cluster_id + 1,
                "theme": theme_name,
                "keywords": ", ".join(top_terms),
                "doc_count": cluster_docs_count,
            })

        return {"has_topics": True, "k": k, "topics": topics}

    except Exception as e:
        logger.error("Error during live topic discovery: %s", e)
        return {"has_topics": False, "reason": f"Clustering error: {str(e)}", "topics": []}


def calculate_live_trend_score(yearly_df):
    """
    Calculate 5-factor composite trend score (0–100) from actual yearly counts.
    """
    if yearly_df.empty or len(yearly_df) < 2:
        return {"score": 50.0, "label": "Stable", "growth_rate": 0, "consistency": 0, "acceleration": 0}

    counts = yearly_df["paper_count"].values.astype(float)
    n = len(counts)

    # 1. Growth Rate (CAGR)
    start_val = max(counts[0], 1.0)
    end_val = max(counts[-1], 1.0)
    cagr = ((end_val / start_val) ** (1.0 / (n - 1)) - 1.0) * 100.0
    growth_score = min(100.0, max(0.0, (cagr / 30.0) * 100.0))  # 30% CAGR = 100 score

    # 2. Consistency
    diffs = np.diff(counts)
    pos_years = np.sum(diffs > 0)
    consistency_score = (pos_years / len(diffs)) * 100.0 if len(diffs) > 0 else 50.0

    # 3. Acceleration
    if n >= 4:
        mid = n // 2
        early = np.mean(diffs[:mid])
        late = np.mean(diffs[mid:])
        accel = late - early
        accel_score = min(100.0, max(0.0, 50.0 + (accel / (np.mean(counts) + 1.0)) * 50.0))
    else:
        accel_score = 50.0

    # 4. Volume Signal
    total_vol = np.sum(counts)
    vol_score = min(100.0, max(10.0, (np.log10(max(total_vol, 1)) / 5.0) * 100.0))

    # 5. Citation / Momentum Proxy
    citation_score = min(100.0, max(20.0, (counts[-1] / (np.mean(counts) + 1.0)) * 60.0))

    # Composite weighted score
    composite = (
        0.35 * growth_score +
        0.25 * consistency_score +
        0.20 * accel_score +
        0.10 * vol_score +
        0.10 * citation_score
    )
    score_val = round(min(100.0, max(0.0, composite)), 1)

    if score_val >= 75.0:
        label = "Emerging"
    elif score_val >= 55.0:
        label = "Growing"
    elif score_val >= 40.0:
        label = "Stable"
    else:
        label = "Declining"

    return {
        "score": score_val,
        "label": label,
        "growth_rate": round(growth_score, 1),
        "consistency": round(consistency_score, 1),
        "acceleration": round(accel_score, 1),
        "volume": round(vol_score, 1),
        "citation": round(citation_score, 1),
        "cagr_pct": round(cagr, 1),
    }


def generate_live_forecast(yearly_df, horizon=3, target_col="paper_count"):
    """
    Dynamically forecast the next 3 years beyond the latest historical year
    using Linear Regression. Includes 80% confidence interval.
    """
    if yearly_df.empty or len(yearly_df) < 3 or target_col not in yearly_df.columns:
        return pd.DataFrame()

    years = yearly_df["year"].values
    counts = yearly_df[target_col].values

    latest_year = int(years[-1])
    future_years = np.array([latest_year + i for i in range(1, horizon + 1)])

    X_train = years.reshape(-1, 1)
    y_train = counts

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_future = future_years.reshape(-1, 1)
    predictions = model.predict(X_future)
    predictions = np.maximum(predictions, 0)

    # Residual standard error for confidence interval
    y_pred_train = model.predict(X_train)
    residuals = y_train - y_pred_train
    s_err = np.std(residuals) if len(residuals) > 1 else max(float(np.mean(counts) * 0.15), 1.0)

    rows = []
    for y, pred in zip(future_years, predictions):
        p_val = round(float(pred))
        rows.append({
            "year": int(y),
            "series_type": target_col,
            "predicted_value": p_val,
            "lower_bound": round(max(0, p_val - 1.28 * s_err)),
            "upper_bound": round(p_val + 1.28 * s_err),
            "is_future": True,
        })

    return pd.DataFrame(rows)


def analyze_research_patent_gap(yearly_papers_df, query, epo_active=False):
    """
    Research-Patent Gap Analysis:
    Compares scholarly publication velocity with intellectual property filings.
    Provides direct official EPO Espacenet verification link and EPO OPS status.
    """
    encoded_query = urllib.parse.quote(query)
    epo_espacenet_url = f"https://worldwide.espacenet.com/patent/search?q={encoded_query}"

    if yearly_papers_df.empty:
        return {
            "status": "Awaiting Data",
            "ratio_label": "N/A",
            "insight": "Run search to calculate gap analysis.",
            "epo_espacenet_url": epo_espacenet_url,
            "google_patents_url": epo_espacenet_url,
            "epo_active": epo_active,
        }

    total_papers = int(yearly_papers_df["paper_count"].sum())
    total_patents = int(yearly_papers_df["patent_count"].sum()) if "patent_count" in yearly_papers_df.columns else 0

    if epo_active and total_patents > 0:
        ratio = total_papers / max(total_patents, 1)
        if ratio > 4.0:
            gap_status = "High Academic Precedence (IP Gap Detected)"
            gap_insight = (
                f"Scholarly publication volume ({total_papers:,} papers) significantly outpaces worldwide patent filings ({total_patents:,} patents via official EPO OPS). "
                "This indicates strong fundamental research momentum preceding industrial IP conversion."
            )
            ratio_indicator = f"Paper-to-Patent Ratio: {ratio:.1f}:1"
        elif ratio < 0.9:
            gap_status = "Commercial & Industrial Dominance"
            gap_insight = (
                f"Worldwide patent filings ({total_patents:,} patents via official EPO OPS) exceed academic research volume ({total_papers:,} papers). "
                "Industry players are actively protecting proprietary implementations in mature engineering domains."
            )
            ratio_indicator = f"Patent Heavy ({ratio:.1f}:1)"
        else:
            gap_status = "Balanced Translation Phase"
            gap_insight = (
                f"Healthy balance between scientific research ({total_papers:,} papers) and patent activity ({total_patents:,} patents via official EPO OPS). "
                "Demonstrates active technology translation from scholarly exploration into intellectual property protection."
            )
            ratio_indicator = f"Balanced ({ratio:.1f}:1)"
    else:
        cagr = yearly_papers_df.get("growth_pct", pd.Series([0])).mean()
        if cagr > 20.0:
            gap_status = "Academic R&D Surge (Pre-Commercial Phase)"
            gap_insight = (
                f"Scholarly publication volume ({total_papers:,} real papers) is accelerating rapidly (+{cagr:.1f}% avg YoY). "
                "High scientific discovery velocity typically precedes industrial patent filings by 12–18 months."
            )
            ratio_indicator = "High Scientific Velocity"
        elif cagr > 8.0:
            gap_status = "Active Translation Phase"
            gap_insight = (
                f"Research growth is steady ({total_papers:,} papers, +{cagr:.1f}% avg YoY). "
                "Technology shows consistent sustained scientific interest transitioning toward commercial implementations."
            )
            ratio_indicator = "Steady R&D Expansion"
        else:
            gap_status = "Mature / Consolidated Stage"
            gap_insight = (
                f"Publication growth has stabilized ({total_papers:,} papers, +{cagr:.1f}% avg YoY), "
                "indicative of an established engineering paradigm."
            )
            ratio_indicator = "Consolidated Domain"

    return {
        "status": gap_status,
        "ratio_label": ratio_indicator,
        "insight": gap_insight,
        "total_papers": total_papers,
        "total_patents": total_patents,
        "epo_espacenet_url": epo_espacenet_url,
        "google_patents_url": epo_espacenet_url,
        "epo_active": epo_active,
        "epo_registration_url": "https://developers.epo.org/",
        "disclaimer": (
            "Research analysis is derived from 100% genuine live OpenAlex scholarly data. "
            "Patent records can be verified via official European Patent Office (EPO) Espacenet."
        ),
    }


def execute_live_search(query):
    """
    Orchestrate full end-to-end live analysis for an arbitrary technology query:
    1. Fetches genuine annual aggregate publication volume via OpenAlex group_by (2018–2026).
    2. Fetches sample paper titles & abstracts from OpenAlex.
    3. Ingests genuine worldwide patent statistics from EPO OPS when credentials are configured.
    4. Runs unsupervised TF-IDF + KMeans clustering on real abstracts.
    5. Calculates multi-factor trend momentum score (CAGR, consistency, acceleration, volume, citations).
    6. Dynamically forecasts the next 3 years (2027–2029) with 80% confidence bounds.
    7. Generates official EPO Espacenet exploration link for patent inspection.
    """
    yearly_df = fetch_live_yearly_counts(query, start_year=2018, end_year=2026)
    sample_docs = fetch_live_sample_documents(query, limit=25)
    topics = discover_live_topics(sample_docs, max_k=4)
    trend_score = calculate_live_trend_score(yearly_df)

    # Official EPO OPS Patent Integration
    epo_active = is_epo_configured()
    sample_patents = []
    if epo_active:
        logger.info("EPO OPS configured: querying official European Patent Office for '%s'...", query)
        patent_counts = fetch_epo_patent_counts(query, start_year=2018, end_year=2026)
        if patent_counts:
            yearly_df["patent_count"] = yearly_df["year"].map(lambda y: patent_counts.get(y, 0))
        else:
            yearly_df["patent_count"] = 0
        sample_patents = fetch_epo_sample_patents(query, limit=10)
    else:
        yearly_df["patent_count"] = 0

    yearly_df["total_count"] = yearly_df["paper_count"] + yearly_df["patent_count"]

    forecast_papers = generate_live_forecast(yearly_df, horizon=3, target_col="paper_count")
    forecast_patents = generate_live_forecast(yearly_df, horizon=3, target_col="patent_count") if epo_active else pd.DataFrame()

    gap = analyze_research_patent_gap(yearly_df, query, epo_active=epo_active)
    gap["sample_patents"] = sample_patents

    return {
        "query": query,
        "yearly_df": yearly_df,
        "sample_docs": sample_docs,
        "sample_patents": sample_patents,
        "topics": topics,
        "trend_score": trend_score,
        "forecast": forecast_papers,
        "forecast_patents": forecast_patents,
        "gap": gap,
        "epo_active": epo_active,
    }
