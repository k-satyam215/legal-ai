"""India Code scraper — Acts & Sections from indiacode.nic.in"""
import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

BASE_URL = "https://www.indiacode.nic.in"
HEADERS  = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}

TARGET_ACTS = [
    "Transfer of Property Act, 1882",
    "Consumer Protection Act, 2019",
    "Indian Penal Code, 1860",
    "Bharatiya Nyaya Sanhita, 2023",
    "Industrial Disputes Act, 1947",
    "Code of Civil Procedure, 1908",
    "Limitation Act, 1963",
    "Specific Relief Act, 1963",
    "Registration Act, 1908",
    "Rent Control Act",
]

CATEGORY_MAP = {
    "Transfer of Property": "rent",
    "Consumer Protection":  "consumer",
    "Indian Penal Code":    "criminal",
    "Bharatiya Nyaya":      "criminal",
    "Industrial Disputes":  "employment",
    "Rent Control":         "rent",
    "Civil Procedure":      "general",
    "Limitation":           "general",
    "Specific Relief":      "general",
    "Registration":         "general",
}


@dataclass
class ActSection:
    law_name: str
    section: str
    section_title: str
    text: str
    category: str
    source: str


def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"[IndiaCode] Attempt {attempt}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


def _clean(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", " ", re.sub(r"\s+", " ", text)).strip()


def _detect_category(law_name: str) -> str:
    for key, cat in CATEGORY_MAP.items():
        if key.lower() in law_name.lower():
            return cat
    return "general"


def scrape_act_sections(act_url: str, law_name: str) -> list[ActSection]:
    resp = _get(act_url)
    if not resp:
        return []
    soup     = BeautifulSoup(resp.text, "html.parser")
    sections = []
    divs     = soup.select(".section-content") or soup.select("div.act-section")
    if not divs:
        text = _clean(" ".join(p.get_text() for p in soup.find_all("p")))
        if len(text) > 200:
            sections.append(ActSection(law_name=law_name, section="Full Act",
                                        section_title=law_name, text=text[:6000],
                                        category=_detect_category(law_name), source=act_url))
        return sections
    for div in divs:
        heading     = div.find(["h3", "h4", "strong"])
        sec_title   = heading.get_text(strip=True) if heading else "Section"
        text        = _clean(div.get_text())
        sec_match   = re.match(r"(Section\s+\d+[\w.-]*)", sec_title, re.IGNORECASE)
        section_num = sec_match.group(1) if sec_match else sec_title[:40]
        if len(text) < 50:
            continue
        sections.append(ActSection(law_name=law_name, section=section_num,
                                    section_title=sec_title, text=text[:3000],
                                    category=_detect_category(law_name), source=act_url))
    logger.info(f"[IndiaCode] {law_name}: {len(sections)} sections")
    return sections


def scrape_target_acts(delay: float = 1.5) -> list[dict]:
    all_sections = []
    for act_name in TARGET_ACTS:
        search_url = f"{BASE_URL}/simple-search?query={requests.utils.quote(act_name)}&rpp=5"
        resp = _get(search_url)
        if not resp:
            continue
        soup  = BeautifulSoup(resp.text, "html.parser")
        links = soup.select(".artifact-title a")
        for link in links[:2]:
            href = link.get("href", "")
            if not href.startswith("http"):
                href = BASE_URL + href
            secs = scrape_act_sections(href, act_name)
            all_sections.extend([asdict(s) for s in secs])
            time.sleep(delay)
        time.sleep(delay)
    logger.info(f"[IndiaCode] Total sections: {len(all_sections)}")
    return all_sections
