import requests
from datetime import datetime
from tqdm import tqdm
from config import CFG
from s3io import put_bytes, hash_bytes, polite_sleep

HEADERS = {"User-Agent": CFG["sec"]["user_agent"]}

def _get_ticker_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}

def _get_submissions(cik: str):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def iter_recent_filings_for(ticker: str, forms=("10-K","20-F"), max_per_company=2):
    tmap = _get_ticker_map()
    if ticker is None or ticker.upper() not in tmap:
        return
    cik = tmap[ticker.upper()]
    subs = _get_submissions(cik)
    rec = subs["filings"]["recent"]

    c = 0
    for form, acc, doc, fdate in zip(rec["form"], rec["accessionNumber"], rec["primaryDocument"], rec["filingDate"]):
        if form not in forms:
            continue
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
        yield {"form": form, "url": url, "filing_date": fdate}
        c += 1
        if c >= max_per_company:
            break
        
def download_and_store_edgar(ticker: str, company_id: str, s3_bucket: str):
    out = []
    for item in iter_recent_filings_for(ticker):
        resp = requests.get(item["url"], headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            continue
        b = resp.content
        sha = hash_bytes(b)
        year = int(item["filing_date"][:4])
        key = f"edgar/{ticker}/{item['filing_date']}_{sha[:10]}.html"
        s3_uri = put_bytes(s3_bucket, key, b, "text/html")

        out.append({
            "company_id": company_id,
            "source": "EDGAR",
            "form_type": item["form"],
            "year": year,
            "filing_date": item["filing_date"],
            "source_url": item["url"],
            "s3_uri": s3_uri,
            "sha256": sha,
            "byte_size": len(b),
            "status": "downloaded",
        })
        polite_sleep(0.4)  # be polite to SEC
    return out