"""
Bulk downloader for Indian corporate & commercial law PDFs.

Downloads key acts from official government sources (indiacode.nic.in)
into data/documents/.

Usage:
    python scripts/download_legal_docs.py
"""

import ssl
import sys
from pathlib import Path

import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the InsecureRequestWarning from urllib3
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DOCUMENTS_DIR  # noqa: E402

# ═══════════════════════════════════════════════════════════════════
#   Corporate & Commercial Law PDFs — indiacode.nic.in
# ═══════════════════════════════════════════════════════════════════

LEGAL_PDFS = {
    # ── Core Corporate Law ──────────────────────────────────────
    "Indian_Contract_Act_1872.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2187/1/A1872-09.pdf",

    "Sale_of_Goods_Act_1930.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2387/2/A1930-03.pdf",

    "Indian_Partnership_Act_1932.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2373/1/A1932-09.pdf",

    "Negotiable_Instruments_Act_1881.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2190/1/A1881-26.pdf",

    "Limited_Liability_Partnership_Act_2008.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2107/1/A2009-06.pdf",

    # ── Securities & Capital Markets ────────────────────────────
    "SEBI_Act_1992.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1893/2/A1992-15.pdf",

    "Securities_Contracts_Regulation_Act_1956.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1784/1/A1956-42.pdf",

    "Depositories_Act_1996.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1972/1/A1996-22.pdf",

    # ── Insolvency & Recovery ───────────────────────────────────
    "Insolvency_and_Bankruptcy_Code_2016.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2154/1/A2016-31.pdf",

    "Recovery_of_Debts_Due_to_Banks_Act_1993.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1947/1/A1993-51.pdf",

    "SARFAESI_Act_2002.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2015/1/A2002-54.pdf",

    # ── Banking & Finance ───────────────────────────────────────
    "Banking_Regulation_Act_1949.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1885/3/A1949-10.pdf",

    "Reserve_Bank_of_India_Act_1934.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2373/5/A1934-02.pdf",

    "Foreign_Exchange_Management_Act_1999.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1988/1/A1999-42.pdf",

    "Payment_and_Settlement_Systems_Act_2007.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2097/1/A2007-51.pdf",

    # ── Corporate Governance & Compliance ───────────────────────
    "Prevention_of_Money_Laundering_Act_2002.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2006/3/A2003-15.pdf",

    "Competition_Act_2002.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2010/1/A2003-12.pdf",

    "Arbitration_and_Conciliation_Act_1996.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1978/2/A1996-26.pdf",

    # ── Tax ─────────────────────────────────────────────────────
    "Income_Tax_Act_1961.pdf":
        "https://indiacode.nic.in/bitstream/123456789/24923/1/A1961-43.pdf",

    "GST_Central_Goods_and_Services_Tax_Act_2017.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2170/1/A2017-12.pdf",

    "Indian_Stamp_Act_1899.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2306/3/A1899-02.pdf",

    # ── Property & Transfer ─────────────────────────────────────
    "Transfer_of_Property_Act_1882.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2338/2/A1882-04.pdf",

    "Registration_Act_1908.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2335/1/a1908-16.pdf",

    # ── IP & Technology ─────────────────────────────────────────
    "Information_Technology_Act_2000.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1999/2/A2000-21.pdf",

    "Trademarks_Act_1999.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1993/1/A1999-47.pdf",

    "Patents_Act_1970.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1392/1/A1970-39.pdf",

    "Copyright_Act_1957.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1367/1/A1957-14.pdf",

    # ── Consumer & Employment ───────────────────────────────────
    "Consumer_Protection_Act_2019.pdf":
        "https://indiacode.nic.in/bitstream/123456789/15256/1/A2019-35.pdf",

    "Industrial_Disputes_Act_1947.pdf":
        "https://indiacode.nic.in/bitstream/123456789/2529/1/A1947-14.pdf",

    "Prevention_of_Corruption_Act_1988.pdf":
        "https://indiacode.nic.in/bitstream/123456789/1558/1/A1988-49.pdf",
}


def main():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0
    failed_names = []

    print(f"[INFO] Target directory: {DOCUMENTS_DIR}")
    print(f"[INFO] {len(LEGAL_PDFS)} corporate law documents in queue\n")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
    })

    for filename, url in LEGAL_PDFS.items():
        dest = DOCUMENTS_DIR / filename

        if dest.exists():
            print(f"  [SKIP]     {filename}")
            skipped += 1
            continue

        print(f"  [DOWNLOAD] {filename} … ", end="", flush=True)
        try:
            resp = session.get(url, timeout=60, verify=False, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if len(resp.content) < 5000:
                print(f"WARN (too small: {len(resp.content)} bytes — may be a redirect page)")
                failed += 1
                failed_names.append(filename)
                continue

            dest.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            downloaded += 1

        except Exception as exc:
            print(f"FAILED ({exc})")
            failed += 1
            failed_names.append(filename)

    print(f"\n{'='*55}")
    print(f"  Downloaded : {downloaded}")
    print(f"  Skipped    : {skipped} (already existed)")
    print(f"  Failed     : {failed}")
    print(f"  Total PDFs : {len(list(DOCUMENTS_DIR.glob('*.pdf')))}")
    print(f"{'='*55}")

    if failed_names:
        print(f"\n  Failed files:")
        for fn in failed_names:
            print(f"    - {fn}")

    if downloaded > 0:
        print(f"\n[NEXT STEP] Index the new documents:")
        print(f"  python scripts/auto_ingest.py --skip-scrape")


if __name__ == "__main__":
    main()
