"""
ingestion/arxiv_connector.py — Data connector for arXiv API.

Fetches research preprint metadata for configured technology domains.
arXiv requires NO API key and provides direct public access.

API Docs: https://info.arxiv.org/help/api/user-manual.html
"""

import time
import logging
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from config import (
    SEARCH_DOMAINS, START_YEAR, END_YEAR,
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _build_arxiv_query(search_terms):
    """
    Build query string for arXiv API.
    Example: all:"quantum computing" OR all:"quantum algorithm"
    """
    clauses = [f'all:"{term}"' for term in search_terms]
    return " OR ".join(clauses)


def _fetch_arxiv_batch(query, start=0, max_results=50):
    """Fetch a single batch of records from arXiv with retry logic."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 429:
                logger.warning("arXiv rate limit reached. Backing off for %ds...", RETRY_DELAY * 3)
                time.sleep(RETRY_DELAY * 3)
            else:
                logger.warning("arXiv HTTP %d on attempt %d", resp.status_code, attempt + 1)
                time.sleep(RETRY_DELAY)
        except requests.RequestException as e:
            logger.warning("arXiv request error attempt %d: %s", attempt + 1, e)
            time.sleep(RETRY_DELAY)

    return None


def _parse_arxiv_xml(xml_content, domain_name):
    """Parse Atom XML response into structured document records."""
    if not xml_content:
        return []

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.error("Failed to parse arXiv XML: %s", e)
        return []

    records = []
    entries = root.findall(f"{ATOM_NS}entry")

    for entry in entries:
        try:
            # Entry ID
            id_elem = entry.find(f"{ATOM_NS}id")
            source_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""

            # Title
            title_elem = entry.find(f"{ATOM_NS}title")
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else "Untitled"

            # Summary / Abstract
            summary_elem = entry.find(f"{ATOM_NS}summary")
            abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else None

            # Publication Date & Year
            pub_elem = entry.find(f"{ATOM_NS}published")
            pub_date = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else ""
            year = int(pub_date[:4]) if len(pub_date) >= 4 and pub_date[:4].isdigit() else END_YEAR

            # Accept papers within research range
            if year < 2010 or year > 2026:
                continue

            # Authors
            authors = []
            for author_elem in entry.findall(f"{ATOM_NS}author"):
                name_elem = author_elem.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            authors_str = "; ".join(authors)

            # Categories / Classifications
            categories = []
            for cat in entry.findall(f"{ATOM_NS}category"):
                term = cat.get("term")
                if term:
                    categories.append(term)
            cat_str = "; ".join(categories)

            # URL
            url = source_id

            record = {
                "source": "arxiv",
                "source_id": source_id,
                "title": title,
                "abstract": abstract,
                "year": year or END_YEAR,
                "classifications": cat_str,
                "authors": authors_str,
                "organizations": "",
                "citation_count": None,
                "url": url,
                "domain": domain_name,
            }
            records.append(record)

        except Exception as e:
            logger.debug("Skipping malformed arXiv entry: %s", e)
            continue

    return records


def fetch_arxiv_for_domain(domain_name, search_terms, max_records=50):
    """
    Fetch papers from arXiv for a single domain.

    Args:
        domain_name: Key from SEARCH_DOMAINS
        search_terms: List of search keywords
        max_records: Maximum records to pull
    """
    logger.info("Fetching arXiv papers for domain: %s", domain_name)
    query = _build_arxiv_query(search_terms)

    all_records = []
    start = 0
    batch_size = min(max_records, 50)

    while len(all_records) < max_records:
        xml_data = _fetch_arxiv_batch(query, start=start, max_results=batch_size)
        if not xml_data:
            break

        records = _parse_arxiv_xml(xml_data, domain_name)
        if not records:
            break

        all_records.extend(records)
        start += batch_size

        # arXiv asks for 3s delay between calls
        time.sleep(2.0)

    logger.info("Retrieved %d arXiv papers for %s", len(all_records), domain_name)
    return all_records[:max_records]


def fetch_all_arxiv(max_per_domain=50):
    """
    Fetch papers for all configured search domains from arXiv.

    Returns:
        pandas DataFrame of all records.
    """
    all_records = []
    for domain, terms in SEARCH_DOMAINS.items():
        records = fetch_arxiv_for_domain(domain, terms, max_records=max_per_domain)
        all_records.extend(records)

    return pd.DataFrame(all_records) if all_records else pd.DataFrame()
