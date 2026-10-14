"""
ingestion/epo_connector.py — European Patent Office (EPO) Open Patent Services Connector.

Provides live access to worldwide patent data (EPO, USPTO, WIPO, JPO, etc.)
via the official EPO OPS v3.2 REST API.
"""

import time
import base64
import logging
import urllib.parse
import re
import requests
import pandas as pd

from config import (
    EPO_CONSUMER_KEY, EPO_CONSUMER_SECRET,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
)

logger = logging.getLogger(__name__)

EPO_AUTH_URL = "https://ops.epo.org/3.2/auth/accesstoken"
EPO_SEARCH_URL = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"

# Cached access token in-memory
_TOKEN_CACHE = {
    "token": None,
    "expires_at": 0,
}


def is_epo_configured():
    """Check if valid EPO OPS credentials are provided."""
    return bool(EPO_CONSUMER_KEY and EPO_CONSUMER_SECRET and EPO_CONSUMER_KEY != "your_epo_consumer_key_here")


def get_epo_access_token():
    """
    Retrieve or refresh OAuth2 Bearer token from EPO OPS.
    """
    if not is_epo_configured():
        return None

    now = time.time()
    if _TOKEN_CACHE["token"] and _TOKEN_CACHE["expires_at"] > now + 60:
        return _TOKEN_CACHE["token"]

    try:
        credentials = f"{EPO_CONSUMER_KEY}:{EPO_CONSUMER_SECRET}"
        encoded_auth = base64.b64encode(credentials.encode("ascii")).decode("ascii")

        headers = {
            "Authorization": f"Basic {encoded_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {"grant_type": "client_credentials"}

        resp = requests.post(EPO_AUTH_URL, headers=headers, data=data, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            payload = resp.json()
            access_token = payload.get("access_token")
            expires_in = int(payload.get("expires_in", 1199))
            _TOKEN_CACHE["token"] = access_token
            _TOKEN_CACHE["expires_at"] = now + expires_in
            logger.info("Successfully acquired fresh EPO OPS access token (expires in %ds)", expires_in)
            return access_token
        else:
            logger.error("Failed to authenticate with EPO OPS (HTTP %d): %s", resp.status_code, resp.text)
            return None
    except Exception as e:
        logger.error("EPO OPS authentication exception: %s", e)
        return None


def fetch_epo_patent_counts(query, start_year=2018, end_year=2026):
    """
    Fetch genuine aggregate patent counts per publication year from EPO OPS.
    Uses title/abstract Common Query Language: ta="{clean_query}" AND pd={year}

    Returns:
        dict: {year: patent_count} or None if unconfigured/failed
    """
    token = get_epo_access_token()
    if not token:
        return None

    clean_query = query.strip()
    counts_by_year = {}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/xml",
    }

    for year in range(start_year, end_year + 1):
        # CQL query: ta="query" AND pd=year
        cql = f'ta="{clean_query}" AND pd={year}'
        encoded_cql = urllib.parse.quote(cql)
        url = f"{EPO_SEARCH_URL}?q={encoded_cql}"

        year_count = 0
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    # Parse total-result-count from XML attribute
                    match = re.search(r'total-result-count="(\d+)"', resp.text)
                    if match:
                        year_count = int(match.group(1))
                    break
                elif resp.status_code == 404:
                    # 404 in OPS Search means 0 results found
                    year_count = 0
                    break
                elif resp.status_code == 403:
                    logger.warning("EPO OPS quota limit reached or invalid permission.")
                    break
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.warning("EPO OPS fetch error for year %d (attempt %d): %s", year, attempt + 1, e)
                time.sleep(RETRY_DELAY)

        counts_by_year[year] = year_count
        time.sleep(0.5)  # Be polite to OPS rate limiter

    return counts_by_year


def fetch_epo_sample_patents(query, limit=10):
    """
    Fetch a sample batch of real patent documents from EPO OPS for inspection.

    Returns:
        list of dicts: [{'source_id', 'title', 'country', 'date', 'kind', 'url'}]
    """
    token = get_epo_access_token()
    if not token:
        return []

    clean_query = query.strip()
    cql = f'ta="{clean_query}"'
    encoded_cql = urllib.parse.quote(cql)
    url = f"{EPO_SEARCH_URL}?q={encoded_cql}&Range=1-{limit}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/xml",
    }

    results = []
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            # Extract document-id entries
            doc_matches = re.findall(
                r'<publication-reference[^>]*>.*?<country>([A-Z]{2})</country>.*?<doc-number>([^<]+)</doc-number>.*?<kind>([^<]*)</kind>.*?(?:<date>(\d{8})</date>)?',
                resp.text,
                re.DOTALL,
            )
            for m in doc_matches[:limit]:
                country, num, kind, date_val = m
                pub_id = f"{country}-{num}{kind}".strip()
                yr = int(date_val[:4]) if date_val else 2024
                results.append({
                    "source_id": pub_id,
                    "title": f"Patent Publication {pub_id} ({country})",
                    "country": country,
                    "year": yr,
                    "url": f"https://worldwide.espacenet.com/patent/search/family/en/publication/{country}{num}{kind}",
                })
    except Exception as e:
        logger.error("Error fetching sample patents from EPO: %s", e)

    return results
