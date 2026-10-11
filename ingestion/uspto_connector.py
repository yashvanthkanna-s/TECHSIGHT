"""
ingestion/uspto_connector.py — Data connector for USPTO Open Data Portal.

Fetches patent metadata for configured technology domains.
Uses the new ODP API (replacing deprecated PatentsView).

MOCK MODE: When USPTO_API_KEY is not set, loads mock data instead.
"""

import time
import logging
import requests
import pandas as pd
from config import (
    USPTO_API_KEY, SEARCH_DOMAINS,
    START_YEAR, END_YEAR, MAX_RECORDS_PER_DOMAIN,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    RATE_LIMIT_DELAY,
)

logger = logging.getLogger(__name__)

# USPTO Open Data Portal API
BASE_URL = "https://api.data.uspto.gov/v1/patents"


def _build_query(search_terms, year):
    """Build a query string for the USPTO API."""
    terms_query = " OR ".join(f'"{t}"' for t in search_terms)
    return {
        "q": terms_query,
        "dateRange": f"{year}-01-01,{year}-12-31",
        "rows": 100,
        "start": 0,
    }


def _fetch_with_retry(url, params, headers):
    """Make an HTTP GET request with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Rate limited. Waiting %d seconds...", RETRY_DELAY * 3)
                time.sleep(RETRY_DELAY * 3)
            else:
                logger.warning("HTTP %d on attempt %d", resp.status_code, attempt + 1)
                time.sleep(RETRY_DELAY)
        except requests.RequestException as e:
            logger.warning("Request error on attempt %d: %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)
    return None


def fetch_patents_for_domain(domain_name, search_terms):
    """
    Fetch patents from USPTO for a single technology domain.

    Args:
        domain_name: Key from SEARCH_DOMAINS
        search_terms: List of search phrases

    Returns:
        List of dicts matching the documents table schema.
    """
    if not USPTO_API_KEY:
        logger.warning(
            "USPTO_API_KEY not set — skipping live fetch for %s", domain_name
        )
        return []

    headers = {
        "Authorization": f"Bearer {USPTO_API_KEY}",
        "Accept": "application/json",
    }

    logger.info("Fetching patents for domain: %s", domain_name)
    all_records = []

    for year in range(START_YEAR, END_YEAR + 1):
        params = _build_query(search_terms, year)
        year_count = 0

        while len(all_records) < MAX_RECORDS_PER_DOMAIN:
            data = _fetch_with_retry(BASE_URL, params, headers)

            if not data or "results" not in data:
                break

            for patent in data["results"]:
                record = {
                    "source": "uspto",
                    "source_id": patent.get("patentNumber", ""),
                    "title": patent.get("inventionTitle", "Untitled"),
                    "abstract": patent.get("abstract", None),
                    "year": year,
                    "classifications": "; ".join(
                        c.get("description", "")
                        for c in patent.get("classifications", [])[:5]
                    ),
                    "authors": "; ".join(
                        inv.get("name", "")
                        for inv in patent.get("inventors", [])
                    ),
                    "organizations": "; ".join(
                        a.get("name", "")
                        for a in patent.get("assignees", [])
                    ),
                    "citation_count": None,  # USPTO citation data is often unavailable
                    "url": f"https://patents.google.com/patent/US{patent.get('patentNumber', '')}",
                    "domain": domain_name,
                }
                all_records.append(record)
                year_count += 1

            # Pagination
            total = data.get("totalCount", 0)
            params["start"] += params["rows"]
            if params["start"] >= total:
                break

            time.sleep(RATE_LIMIT_DELAY)

        logger.info("  Year %d: %d patents", year, year_count)

    logger.info("Total patents for %s: %d", domain_name, len(all_records))
    return all_records


def fetch_all_patents():
    """
    Fetch patents for all configured domains.

    Returns:
        pandas DataFrame of all records.
    """
    all_records = []
    for domain, terms in SEARCH_DOMAINS.items():
        records = fetch_patents_for_domain(domain, terms)
        all_records.extend(records)
    return pd.DataFrame(all_records) if all_records else pd.DataFrame()
