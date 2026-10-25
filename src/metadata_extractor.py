"""
Legal metadata extraction — detect act name, year, and section numbers
from raw PDF text so chunks carry lawyer-friendly metadata.

Strategy (in priority order):
  1. Regex the first-page text for a title like "THE X ACT, YYYY"
  2. Fall back to PDF file metadata title (rarely populated)
  3. Fall back to cleaned filename

Also scans each chunk for "Section N" / "Article N" / "Rule N" mentions
so we can attach a `section` field when a chunk is clearly about one.
"""

import re
from pathlib import Path
from typing import Dict, Optional, List


# ── Regex patterns for Indian legal document titles ─────────────
_TITLE_PATTERNS = [
    # "THE COMPANIES ACT, 2013"   /  "The Companies Act, 2013"
    re.compile(r"\bTHE\s+([A-Z][A-Za-z0-9 ,&()'\-]{3,120}?ACT,?\s*(\d{4}))\b", re.I),
    # "THE SEBI ACT, 1992"
    re.compile(r"\bTHE\s+([A-Z][A-Za-z0-9 ,&()'\-]{3,120}?REGULATIONS?,?\s*(\d{4}))\b", re.I),
    # "INDIAN PENAL CODE, 1860"
    re.compile(r"\b(INDIAN\s+[A-Z][A-Za-z0-9 ,&()'\-]{3,120}?CODE,?\s*(\d{4}))\b", re.I),
    # Generic: "XYZ Act 1949"
    re.compile(r"\b([A-Z][A-Za-z0-9 ,&()'\-]{3,120}?ACT,?\s*(\d{4}))\b", re.I),
]

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# Section / Article / Rule / Clause references inside chunk text
_SECTION_RE = re.compile(
    r"\b(?:Section|Sec\.?|Article|Rule|Clause)\s+(\d+[A-Z]?(?:\(\d+\))?)",
    re.I,
)


def _clean_name(raw: str) -> str:
    """Turn 'the companies act, 2013' → 'Companies Act 2013'."""
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"^THE\s+", "", raw, flags=re.I)
    raw = raw.replace(",", "")
    # Title-case each word but keep acronyms (SEBI, RBI) uppercase
    words = []
    for w in raw.split():
        if w.isupper() and len(w) <= 6 and w.isalpha():
            words.append(w)
        else:
            words.append(w.capitalize())
    return " ".join(words)


def detect_act_name(first_page_text: str, filename: str) -> str:
    """Return a clean act name for a PDF.

    Tries regex on the text first, then falls back to filename.
    """
    # 1. Regex the text
    head = first_page_text[:3000]  # only first ~3k chars matter
    for pat in _TITLE_PATTERNS:
        m = pat.search(head)
        if m:
            return _clean_name(m.group(1))

    # 2. Fallback: filename
    stem = Path(filename).stem
    stem = re.sub(r"^\d+", "", stem)          # strip leading digits
    stem = stem.replace("_", " ").replace("-", " ").strip()
    if stem:
        return _clean_name(stem)

    return "Unknown Act"


def detect_year(text: str, act_name: str) -> Optional[int]:
    """Detect the year the act was passed."""
    # Prefer a year inside the act name
    m = _YEAR_RE.search(act_name)
    if m:
        return int(m.group(1))
    # Otherwise first year in the first page
    m = _YEAR_RE.search(text[:2000])
    return int(m.group(1)) if m else None


def detect_section(chunk_text: str) -> Optional[str]:
    """Return the first section/article reference in a chunk, if any."""
    m = _SECTION_RE.search(chunk_text)
    if m:
        return m.group(1)
    return None


def extract_metadata(
    first_page_text: str,
    chunk_text: str,
    filename: str,
) -> Dict:
    """One-shot metadata builder for a single chunk."""
    act_name = detect_act_name(first_page_text, filename)
    year = detect_year(first_page_text, act_name)
    section = detect_section(chunk_text)

    meta = {
        "act_name": act_name,
        "year": year,
        "source_file": Path(filename).name,
    }
    if section:
        meta["section"] = section
    return meta


def clean_filename_for_act(act_name: str) -> str:
    """Turn 'Companies Act 2013' → 'Companies_Act_2013.pdf'."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", act_name).strip("_")
    return f"{safe}.pdf"