# company_pdf.py (multi-year variant)

import re, requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from ingestion.s3io import put_bytes, hash_bytes

UA = {"User-Agent": "Mozilla/5.0 (compatible; GreenVerifier/0.1)"}
KEYWORDS = ["sustainability", "esg", "csr", "impact", "responsibility", "non-financial", "integrated", "report"]

def _links(html, base):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        yield urljoin(base, a["href"])

def fetch_esg_pdfs_from_site(company_name: str, website: str, s3_bucket: str, max_files: int = 5):
    if not website:
        return []

    s = requests.Session(); s.headers.update(UA)
    out = []

    try:
        r = s.get(website, timeout=30); r.raise_for_status()
    except requests.RequestException:
        print(f"  SITE: homepage fetch failed for {company_name}.")
        return []

    candidates = list(_links(r.text, website))
    pages = [u for u in candidates if any(k in u.lower() for k in KEYWORDS)]
    pages = pages[:12] or candidates[:12]

    seen = set()
    for page in pages:
        try:
            pr = s.get(page, timeout=30); pr.raise_for_status()
        except requests.RequestException:
            continue

        pdfs = [u for u in _links(pr.text, page)
                if u.lower().endswith(".pdf") and u not in seen]
        pdfs.sort(
            key=lambda u: (any(k in u.lower() for k in KEYWORDS),
                           re.search(r"20\d{2}", u) is not None),
            reverse=True
        )

        for pdf_url in pdfs[:max_files]:
            seen.add(pdf_url)
            try:
                fr = s.get(pdf_url, timeout=120); fr.raise_for_status()
            except requests.RequestException:
                continue

            b = fr.content
            sha = hash_bytes(b)
            m = re.search(r"(20\d{2})", pdf_url)
            year = int(m.group(1)) if m else None

            key = f"site/{company_name.replace(' ','_')}/{(year or 'unknown')}_{sha[:10]}.pdf"
            s3_uri = put_bytes(s3_bucket, key, b, "application/pdf")

            out.append({
                "source": "COMPANY_SITE",
                "form_type": "SUS-REPORT",
                "year": year,
                "filing_date": None,
                "source_url": pdf_url,
                "s3_uri": s3_uri,
                "sha256": sha,
                "byte_size": len(b),
                "status": "downloaded",
            })

            if len(out) >= max_files:
                return out

    print(f"  SITE: stored {len(out)} PDFs for {company_name}")
    return out
