import re, requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ingestion.s3io import put_bytes, hash_bytes

UA = {"User-Agent": "Mozilla/5.0 (compatible; GreenVerifier/0.1; +https://example.com)"}

def _session():
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.headers.update(UA)
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def _find_pdf_links(html: str):
    soup = BeautifulSoup(html, "html.parser")
    hrefs = [a.get("href") for a in soup.find_all("a", href=True)]
    pdfs = [h for h in hrefs if h and h.lower().endswith(".pdf")]
    pdfs_sorted = sorted(
        pdfs,
        key=lambda x: (("sustain" in x.lower()) or ("esg" in x.lower()), x.lower()),
        reverse=True,
    )
    return pdfs_sorted

def fetch_latest_gri_pdf(company_name: str, company_profile_url: str, s3_bucket: str):
    if not company_profile_url:
        return None

    sess = _session()
    try:
        r = sess.get(company_profile_url, timeout=45)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  GRI: profile fetch error ({e.__class__.__name__}). Skipping.")
        return None

    pdfs = _find_pdf_links(r.text)
    if not pdfs:
        print("  GRI: no PDFs found on profile page.")
        return None

    pdf_url = pdfs[0] if pdfs[0].startswith("http") else requests.compat.urljoin(company_profile_url, pdfs[0])
    try:
        pr = sess.get(pdf_url, timeout=120)
        pr.raise_for_status()
    except requests.RequestException as e:
        print(f"  GRI: PDF download error ({e.__class__.__name__}). Skipping.")
        return None

    b = pr.content
    sha = hash_bytes(b)

    m = re.search(r"(20\d{2})", pdf_url)
    year = int(m.group(1)) if m else None

    key = f"gri/{company_name.replace(' ','_')}/{(year or 'unknown')}_{sha[:10]}.pdf"
    s3_uri = put_bytes(s3_bucket, key, b, "application/pdf")

    return {
        "source": "GRI",
        "form_type": "SUS-REPORT",
        "year": year,
        "filing_date": None,
        "source_url": pdf_url,
        "s3_uri": s3_uri,
        "sha256": sha,
        "byte_size": len(b),
        "status": "downloaded",
    }
