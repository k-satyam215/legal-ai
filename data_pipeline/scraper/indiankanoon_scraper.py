"""Indian Kanoon scraper — case law judgments."""
import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL   = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/"
HEADERS    = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}

CATEGORY_QUERIES = {
    "rent":       "security deposit refund tenant landlord",
    "consumer":   "consumer forum deficiency service refund",
    "criminal":   "bail FIR section 302 IPC judgment",
    "employment": "wrongful termination employee industrial dispute",
    "general":    "Indian law judgment Supreme Court",
}


@dataclass
class CaseLaw:
    title: str
    court: str
    date: str
    citation: str
    text: str
    url: str
    category: str


def _get(url: str, retries: int = 3, delay: float = 2.0) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            logger.warning(f"[Scraper] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    return None


def _clean_text(raw: str) -> str:
    raw = re.sub(r"\s+", " ", raw)
    return re.sub(r"[^\x00-\x7F]+", " ", raw).strip()


def _parse_case_page(url: str, category: str) -> Optional[CaseLaw]:
    resp = _get(url)
    if not resp:
        return None
    soup      = BeautifulSoup(resp.text, "html.parser")
    title_tag = soup.find("h2", class_="doc_title") or soup.find("title")
    title     = title_tag.get_text(strip=True) if title_tag else "Unknown"
    doc_div   = soup.find("div", id="judgments") or soup.find("div", class_="doc_content")
    text      = _clean_text(doc_div.get_text()) if doc_div else ""
    court     = "Unknown Court"
    date      = "Unknown Date"
    citation  = url.split("/")[-2] if "/" in url else "N/A"
    if bench := soup.find("div", class_="docsource_main"):
        court = bench.get_text(strip=True)
    if date_tag := soup.find("div", class_="doc_date"):
        date = date_tag.get_text(strip=True)
    if len(text) < 200:
        return None
    return CaseLaw(title=title, court=court, date=date, citation=citation,
                   text=text[:8000], url=url, category=category)


def scrape_search_page(query: str, category: str, page: int = 0) -> list[str]:
    params = {"formInput": query, "pagenum": page}
    try:
        r    = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        return [BASE_URL + a.get("href", "") for a in soup.select(".result_title a")
                if a.get("href", "").startswith("/doc/")]
    except Exception:
        return []


def scrape_category(category: str, max_pages: int = 3, delay: float = 1.5) -> list[CaseLaw]:
    query     = CATEGORY_QUERIES.get(category, "Indian law")
    all_cases = []
    for page in range(max_pages):
        urls = scrape_search_page(query, category, page)
        for url in urls:
            case = _parse_case_page(url, category)
            if case:
                all_cases.append(case)
            time.sleep(delay)
        time.sleep(delay * 2)
    logger.info(f"[Scraper] {category}: {len(all_cases)} cases")
    return all_cases


def scrape_all_categories(max_pages_per_cat: int = 2) -> list[dict]:
    all_docs = []
    for cat in CATEGORY_QUERIES:
        cases = scrape_category(cat, max_pages=max_pages_per_cat)
        all_docs.extend([asdict(c) for c in cases])
    return all_docs
