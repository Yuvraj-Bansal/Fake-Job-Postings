"""
scraper.py — job posting URL scraper for Fake Job Detector
Supports: LinkedIn, Indeed, Naukri, Glassdoor, and generic pages.

"""

import re
import time
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REMOTE_KEYWORDS = {
    "remote", "work from home", "wfh", "telecommute", "fully remote",
    "work remotely", "distributed", "anywhere",
}

BENEFIT_KEYWORDS = {
    "health insurance", "dental", "vision", "401k", "pto", "paid leave",
    "equity", "stock options", "bonus", "flexible hours", "gym",
}


def _fetch_html(url: str, timeout: int = 10) -> BeautifulSoup | None:
    """Fetch and parse a URL. Returns None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return None


def _clean(text: str | None) -> str:
    """Strip whitespace and collapse blank lines."""
    if not text:
        return ""
    text = re.sub(r"\s{3,}", "\n\n", text.strip())
    return text


def _detect_remote(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in REMOTE_KEYWORDS)


def _detect_benefits(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in BENEFIT_KEYWORDS)


# ─────────────────────────────────────────────
# Site-specific parsers
# ─────────────────────────────────────────────

def _parse_linkedin(soup: BeautifulSoup) -> dict:
    """
    Parse a LinkedIn public job view page.
    Note: LinkedIn public pages are throttled. The Chrome extension bypasses
    this entirely — this scraper targets the public /jobs/view/ URL only.
    """
    result = {}

    title_tag = soup.select_one("h1.top-card-layout__title, h1.t-24")
    result["title"] = _clean(title_tag.get_text()) if title_tag else ""

    desc_tag = soup.select_one(
        "div.description__text, "
        "div.show-more-less-html__markup, "
        "section.description"
    )
    full_desc = _clean(desc_tag.get_text(separator="\n")) if desc_tag else ""
    result["description"] = full_desc

    company_tag = soup.select_one(
        "a.topcard__org-name-link, span.topcard__flavor"
    )
    result["company_profile"] = _clean(company_tag.get_text()) if company_tag else ""

    criteria = soup.select("li.description__job-criteria-item")
    criteria_text = " ".join(c.get_text(" ", strip=True) for c in criteria)

    result["requirements"] = ""
    for tag in soup.select("h3, h4"):
        heading = tag.get_text().lower()
        if "requirement" in heading or "qualification" in heading:
            sibling = tag.find_next_sibling(["ul", "p"])
            if sibling:
                result["requirements"] = _clean(sibling.get_text(separator="\n"))
            break

    result["has_company"] = bool(result["company_profile"])
    result["has_logo"] = bool(soup.select_one("img.artdeco-entity-image"))
    result["has_salary"] = bool(
        soup.select_one("span.compensation__salary") or
        re.search(r"\$[\d,]+|\binr\b|₹[\d,]+|per\s+(?:year|month|hour|annum)", full_desc, re.I)
    )
    result["has_requirements"] = bool(result["requirements"])
    result["has_benefits"] = _detect_benefits(full_desc)
    result["telecommuting"] = _detect_remote(full_desc + " " + criteria_text)

    return result


def _parse_indeed(soup: BeautifulSoup) -> dict:
    result = {}

    title_tag = soup.select_one("h1.jobsearch-JobInfoHeader-title, h1[data-testid='jobsearch-JobInfoHeader-title']")
    result["title"] = _clean(title_tag.get_text()) if title_tag else ""

    desc_tag = soup.select_one("#jobDescriptionText, div.jobsearch-jobDescriptionText")
    full_desc = _clean(desc_tag.get_text(separator="\n")) if desc_tag else ""
    result["description"] = full_desc

    company_tag = soup.select_one("div[data-testid='inlineHeader-companyName'], a[data-testid='inlineHeader-companyName']")
    result["company_profile"] = _clean(company_tag.get_text()) if company_tag else ""

    result["requirements"] = ""
    result["has_company"] = bool(result["company_profile"])
    result["has_logo"] = bool(soup.select_one("img.jobsearch-CompanyAvatar-image"))
    result["has_salary"] = bool(
        soup.select_one("span[data-testid='attribute_snippet_testid']") or
        re.search(r"\$[\d,]+|\binr\b|₹[\d,]+|per\s+(?:year|month|hour|annum)", full_desc, re.I)
    )
    result["has_requirements"] = bool(re.search(r"requirement|qualification", full_desc, re.I))
    result["has_benefits"] = _detect_benefits(full_desc)
    result["telecommuting"] = _detect_remote(full_desc)

    return result


def _parse_naukri(soup: BeautifulSoup) -> dict:
    result = {}

    title_tag = soup.select_one("h1.jd-header-title, h1.styles_jd-header-title__rZwM1")
    result["title"] = _clean(title_tag.get_text()) if title_tag else ""

    desc_tag = soup.select_one("section.styles_job-desc-container__txpYf, div.job-description")
    full_desc = _clean(desc_tag.get_text(separator="\n")) if desc_tag else ""
    result["description"] = full_desc

    company_tag = soup.select_one("a.styles_comp-name__3OA1F, div.comp-name")
    result["company_profile"] = _clean(company_tag.get_text()) if company_tag else ""

    result["requirements"] = ""
    result["has_company"] = bool(result["company_profile"])
    result["has_logo"] = bool(soup.select_one("img.styles_logo__28KWB, img.company-logo"))
    result["has_salary"] = bool(
        soup.select_one("span.salary, i.salary") or
        re.search(r"₹[\d,]+|lpa|lakh|per\s+annum", full_desc, re.I)
    )
    result["has_requirements"] = bool(re.search(r"requirement|qualification", full_desc, re.I))
    result["has_benefits"] = _detect_benefits(full_desc)
    result["telecommuting"] = _detect_remote(full_desc)

    return result


def _parse_glassdoor(soup: BeautifulSoup) -> dict:
    result = {}

    title_tag = soup.select_one("h1[data-test='job-title'], div.e1tk4kwz4 h1")
    result["title"] = _clean(title_tag.get_text()) if title_tag else ""

    desc_tag = soup.select_one("div.jobDescriptionContent, div[data-test='jobDescriptionContent']")
    full_desc = _clean(desc_tag.get_text(separator="\n")) if desc_tag else ""
    result["description"] = full_desc

    company_tag = soup.select_one("a[data-test='employer-name'], div.e1tk4kwz1")
    result["company_profile"] = _clean(company_tag.get_text()) if company_tag else ""

    result["requirements"] = ""
    result["has_company"] = bool(result["company_profile"])
    result["has_logo"] = bool(soup.select_one("img.avatar, div.employerLogoWrapper img"))
    result["has_salary"] = bool(
        soup.select_one("span[data-test='detailSalary']") or
        re.search(r"\$[\d,]+|per\s+(?:year|month|hour)", full_desc, re.I)
    )
    result["has_requirements"] = bool(re.search(r"requirement|qualification", full_desc, re.I))
    result["has_benefits"] = _detect_benefits(full_desc)
    result["telecommuting"] = _detect_remote(full_desc)

    return result


def _parse_generic(soup: BeautifulSoup) -> dict:
    """
    Fallback heuristic parser for unknown job boards.
    Tries to find the title in <h1> and description in the largest text block.
    """
    result = {}

    h1 = soup.find("h1")
    result["title"] = _clean(h1.get_text()) if h1 else ""

    # Find the largest block of text on the page (likely the job description)
    candidates = soup.select("article, main, section, div[class*='desc'], div[class*='job'], div[id*='desc'], div[id*='job']")
    best = ""
    for c in candidates:
        text = c.get_text(separator="\n", strip=True)
        if len(text) > len(best):
            best = text
    result["description"] = _clean(best)

    result["company_profile"] = ""
    result["requirements"] = ""
    result["has_company"] = False
    result["has_logo"] = bool(soup.select_one("img[class*='logo'], img[alt*='logo']"))
    result["has_salary"] = bool(re.search(r"\$[\d,]+|₹[\d,]+|per\s+(?:year|month|hour|annum)|lpa", result["description"], re.I))
    result["has_requirements"] = bool(re.search(r"requirement|qualification", result["description"], re.I))
    result["has_benefits"] = _detect_benefits(result["description"])
    result["telecommuting"] = _detect_remote(result["description"])

    return result


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

PARSER_MAP = {
    "linkedin.com": _parse_linkedin,
    "indeed.com":   _parse_indeed,
    "naukri.com":   _parse_naukri,
    "glassdoor.com": _parse_glassdoor,
    "glassdoor.co.in": _parse_glassdoor,
}


def scrape_job_posting(url: str) -> dict:
    """
    Scrape a job posting URL and return a dict of extracted fields.

    Returns:
        {
            "title": str,
            "description": str,
            "company_profile": str,
            "requirements": str,
            "has_salary": bool,
            "has_logo": bool,
            "has_company": bool,
            "has_requirements": bool,
            "has_benefits": bool,
            "telecommuting": bool,
            "error": str | None,    # set if scraping failed
        }
    """
    empty = {
        "title": "", "description": "", "company_profile": "", "requirements": "",
        "has_salary": False, "has_logo": False, "has_company": False,
        "has_requirements": False, "has_benefits": False, "telecommuting": False,
        "error": None,
    }

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    soup = _fetch_html(url)
    if soup is None:
        empty["error"] = "Could not fetch the URL. Check if it's publicly accessible."
        return empty

    domain = urlparse(url).netloc.replace("www.", "")
    parser_fn = next(
        (fn for host, fn in PARSER_MAP.items() if host in domain),
        _parse_generic,
    )

    try:
        result = parser_fn(soup)
    except Exception as e:
        empty["error"] = f"Parsing failed: {e}"
        return empty

    result["error"] = None
    return result
