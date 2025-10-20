from config import CFG
from db import fetch_companies, upsert_document_row
from edgar import download_and_store_edgar
from gri import fetch_latest_gri_pdf
from company_pdf import fetch_esg_pdfs_from_site

def run(limit_companies=None, do_edgar=True, do_gri=True, do_site_fallback=True):
    companies = fetch_companies(limit=limit_companies)
    for c in companies:
        company_id = c["company_id"]
        name = c["name"]
        ticker = (c["ticker"] or "").upper()
        gri_url = c.get("gri_profile_url")
        website = c.get("website")

        print(f"\n== Processing {name} ==")

        # EDGAR
        if do_edgar and ticker:
            docs = download_and_store_edgar(ticker, company_id, CFG["aws"]["s3_bucket_raw"])
            for row in docs:
                upsert_document_row(row)
            print(f"  EDGAR: stored {len(docs)} docs")

        # GRI then fallback to company site
        meta = None
        if do_gri and gri_url:
            meta = fetch_latest_gri_pdf(name, gri_url, CFG["aws"]["s3_bucket_raw"])
        if not meta and do_site_fallback and website:
            print("  Trying website fallback…")
            site_docs = fetch_esg_pdfs_from_site(name, website, CFG["aws"]["s3_bucket_raw"], max_files=5)
            for row in site_docs:
                row["company_id"] = company_id
                upsert_document_row(row)
            if site_docs:
                print(f"  ESG PDF: stored {len(site_docs)} files")
            else:
                print("  ESG PDF: none found / skipped")

if __name__ == "__main__":
    run(limit_companies=None)
