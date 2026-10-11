"""
ingestion/openalex_connector.py — Data connector for OpenAlex API.

Fetches research publication metadata for configured technology domains.
Uses cursor-based pagination and polite-pool headers.

MOCK MODE: When OPENALEX_API_KEY is not set, loads mock data instead.
"""

import time
import logging
import requests
import pandas as pd
from config import (
    OPENALEX_API_KEY, OPENALEX_EMAIL, SEARCH_DOMAINS,
    START_YEAR, END_YEAR, MAX_RECORDS_PER_DOMAIN,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    RATE_LIMIT_DELAY, OPENALEX_PER_PAGE,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org/works"


def _build_params(search_terms, year, cursor="*"):
    """Build query parameters for an OpenAlex API request."""
    query = " OR ".join(f'"{t}"' for t in search_terms)
    params = {
        "filter": f"default.search:{query},publication_year:{year}",
        "per_page": OPENALEX_PER_PAGE,
        "cursor": cursor,
        "select": "id,title,abstract_inverted_index,publication_year,"
                  "authorships,cited_by_count,primary_location,topics",
    }
    if OPENALEX_EMAIL:
        params["mailto"] = OPENALEX_EMAIL
    return params


def _reconstruct_abstract(inverted_index):
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return None
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def _extract_authors(authorships):
    """Extract semicolon-separated author names."""
    if not authorships:
        return ""
    names = []
    for a in authorships:
        name = a.get("author", {}).get("display_name", "")
        if name:
            names.append(name)
    return "; ".join(names)


def _extract_organizations(authorships):
    """Extract semicolon-separated institution names."""
    if not authorships:
        return ""
    orgs = set()
    for a in authorships:
        for inst in a.get("institutions", []):
            name = inst.get("display_name", "")
            if name:
                orgs.add(name)
    return "; ".join(orgs)


def _fetch_with_retry(url, params):
    """Make an HTTP GET request with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Rate limited. Waiting %d seconds...", RETRY_DELAY * 2)
                time.sleep(RETRY_DELAY * 2)
            else:
                logger.warning("HTTP %d on attempt %d", resp.status_code, attempt + 1)
                time.sleep(RETRY_DELAY)
        except requests.RequestException as e:
            logger.warning("Request error on attempt %d: %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)
    return None


def fetch_papers_for_domain(domain_name, search_terms):
    """
    Fetch research papers from OpenAlex for a single technology domain.

    Args:
        domain_name: Key from SEARCH_DOMAINS (e.g. "quantum_computing")
        search_terms: List of search phrases

    Returns:
        List of dicts matching the documents table schema.
    """
    logger.info("Fetching papers for domain: %s", domain_name)
    all_records = []

    for year in range(START_YEAR, END_YEAR + 1):
        cursor = "*"
        year_count = 0

        while cursor and len(all_records) < MAX_RECORDS_PER_DOMAIN:
            params = _build_params(search_terms, year, cursor)
            data = _fetch_with_retry(BASE_URL, params)

            if not data or "results" not in data:
                break

            for work in data["results"]:
                record = {
                    "source": "openalex",
                    "source_id": work.get("id", ""),
                    "title": work.get("title", "Untitled"),
                    "abstract": _reconstruct_abstract(
                        work.get("abstract_inverted_index")
                    ),
                    "year": work.get("publication_year", year),
                    "classifications": "; ".join(
                        t.get("display_name", "")
                        for t in work.get("topics", [])[:5]
                    ),
                    "authors": _extract_authors(work.get("authorships", [])),
                    "organizations": _extract_organizations(
                        work.get("authorships", [])
                    ),
                    "citation_count": work.get("cited_by_count"),
                    "url": work.get("id", ""),
                    "domain": domain_name,
                }
                all_records.append(record)
                year_count += 1

            # Get next cursor for pagination
            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")

            time.sleep(RATE_LIMIT_DELAY)

        logger.info("  Year %d: %d papers", year, year_count)

    logger.info("Total papers for %s: %d", domain_name, len(all_records))
    return all_records


def fetch_all_papers():
    """
    Fetch papers for all configured domains.

    Returns:
        pandas DataFrame of all records.
    """
    all_records = []
    for domain, terms in SEARCH_DOMAINS.items():
        records = fetch_papers_for_domain(domain, terms)
        all_records.extend(records)
    return pd.DataFrame(all_records)
