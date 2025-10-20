import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from ingestion.config import CFG

@contextmanager
def pg_conn():
    conn = psycopg2.connect(
        host=CFG["pg"]["host"],
        dbname=CFG["pg"]["db"],
        user=CFG["pg"]["user"],
        password=CFG["pg"]["password"],
        port=CFG["pg"]["port"],
        connect_timeout=10,
    )
    try:
        yield conn
    finally:
        conn.close()

def fetch_companies(limit=None):
    sql = """
      SELECT company_id, name, ticker, cik, gri_profile_url, website
      FROM companies
      ORDER BY created_at
    """
    if limit:
        sql += " LIMIT %s"
    with pg_conn() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (limit,) if limit else None)
        return cur.fetchall()

def upsert_document_row(row: dict):
    sql = """
    INSERT INTO documents
      (doc_id, company_id, source, form_type, year, filing_date,
       source_url, s3_uri, sha256, byte_size, status)
    VALUES (gen_random_uuid(), %(company_id)s, %(source)s, %(form_type)s, %(year)s, %(filing_date)s,
            %(source_url)s, %(s3_uri)s, %(sha256)s, %(byte_size)s, %(status)s)
    ON CONFLICT (source_url) DO NOTHING;
    """
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, row)
        conn.commit()
