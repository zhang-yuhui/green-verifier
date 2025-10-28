# rag_s3_pipeline.py  (with verbose print logging)
import os
import json
import time
import tempfile
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import boto3
from dotenv import load_dotenv

# ---- LangChain / Transformers bits
from langchain_community.document_loaders import PyMuPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
import torch

# --------------------------------------------------------------------------------------
# 0) Config
# --------------------------------------------------------------------------------------
load_dotenv()  # reads .env next to this file or current working dir

CFG = {
    "aws": {
        "region": os.getenv("AWS_REGION", "eu-central-1"),
        "s3_bucket_raw": os.getenv("S3_BUCKET_RAW"),
        "prefix_edgar": os.getenv("S3_PREFIX_EDGAR", "edgar/"),
        "prefix_site": os.getenv("S3_PREFIX_SITE", "site/"),
        "output_prefix": os.getenv("S3_OUTPUT_PREFIX", "results/"),
    }
}

VERBOSE = True  # <- flip to False to reduce prints

def log(msg: str):
    if VERBOSE:
        print(msg)

def preview(txt: str, n: int = 160) -> str:
    txt = txt.replace("\n", " ").strip()
    return (txt[:n] + "…") if len(txt) > n else txt

# --------------------------------------------------------------------------------------
# 1) LLM (local HF)
# --------------------------------------------------------------------------------------
def build_local_chat(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
    quant_4bit: bool = False,
):
    start = time.time()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[LLM] Loading model: {model_id} | device={device} | dtype={dtype} | 4bit={quant_4bit}")

    quantization_config = None
    if quant_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception as e:
            log(f"[LLM] bitsandbytes unavailable ({e}); running without 4-bit.")
            quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=gen_pipe)
    log(f"[LLM] Model ready in {time.time()-start:.1f}s")
    return ChatHuggingFace(llm=llm)

# --------------------------------------------------------------------------------------
# 2) S3 helpers (supports .pdf, .html, .htm)
# --------------------------------------------------------------------------------------
def list_s3_keys(
    bucket: str,
    prefix: str,
    region: str,
    suffixes: Tuple[str, ...] = (".pdf", ".html", ".htm"),
) -> List[str]:
    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    cnt = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            cnt += 1
            if key.lower().endswith(suffixes):
                keys.append(key)
    log(f"[S3] Prefix '{prefix}': scanned {cnt} objects, matched {len(keys)} ({', '.join(suffixes)})")
    return keys

def download_to_temp(bucket: str, key: str, region: str) -> str:
    s3 = boto3.client("s3", region_name=region)
    ext = "." + key.split(".")[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    s3.download_fileobj(bucket, key, tmp)
    tmp.flush(); tmp.close()
    log(f"[S3] Downloaded s3://{bucket}/{key} -> {tmp.name}")
    return tmp.name

def load_single_s3_doc(bucket: str, key: str, region: str) -> List:
    local_path = download_to_temp(bucket, key, region)
    s3_uri = f"s3://{bucket}/{key}"

    if key.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(local_path)
        kind = "PDF"
    elif key.lower().endswith((".html", ".htm")):
        loader = BSHTMLLoader(local_path, bs_kwargs={"features": "lxml-xml"})
        kind = "HTML"
    else:
        return []

    docs = loader.load()
    log(f"[LOAD] {kind} pages loaded: {len(docs)} | {s3_uri}")

    # normalize prefixes to detect origin
    edgar_pfx = CFG["aws"]["prefix_edgar"].rstrip("/") + "/"
    for d in docs:
        d.metadata["source"] = s3_uri
        d.metadata["file_path"] = s3_uri
        d.metadata["origin"] = "edgar" if key.startswith(edgar_pfx) else "site"
    return docs

def load_docs_from_s3_sources(cfg: Dict[str, Any], max_files: int | None = 5) -> Dict[str, List]:
    bucket = cfg["aws"]["s3_bucket_raw"]
    region = cfg["aws"]["region"]
    prefixes = [cfg["aws"]["prefix_edgar"], cfg["aws"]["prefix_site"]]

    # 1) Collect all matching keys across prefixes
    all_keys: List[str] = []
    for pfx in prefixes:
        all_keys.extend(list_s3_keys(bucket, pfx, region))

    # 2) Make selection deterministic and cap to max_files
    all_keys = sorted(all_keys)
    if max_files is not None:
        all_keys = all_keys[:max_files]

    log(f"[LOAD] Will process {len(all_keys)} file(s):")
    for i, k in enumerate(all_keys, 1):
        log(f"  - [{i}] s3://{bucket}/{k}")

    # 3) Download + load only the selected keys
    grouped_docs: Dict[str, List] = defaultdict(list)
    for key in all_keys:
        docs = load_single_s3_doc(bucket, key, region)
        if not docs:
            continue
        grouped_docs[f"s3://{bucket}/{key}"].extend(docs)

    log(f"[LOAD] Grouped files (selected): {len(grouped_docs)}")
    return dict(grouped_docs)


# --------------------------------------------------------------------------------------
# 3) RAG utilities
# --------------------------------------------------------------------------------------
def build_retriever_from_docs(
    docs,
    embed_model: str = "intfloat/e5-base-v2",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    k: int = 4,
):
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)
    log(f"[CHUNK] chunks={len(splits)} (size={chunk_size}, overlap={chunk_overlap})")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": _device})
    log(f"[EMB] model={embed_model} | device={_device}")
    vs = FAISS.from_documents(splits, embeddings)
    log(f"[INDEX] FAISS built")
    return vs.as_retriever(search_kwargs={"k": k})

def format_docs(docs) -> str:
    return "\n\n".join(f"[{i+1}] " + d.page_content for i, d in enumerate(docs))

def unique_sources(docs) -> List[str]:
    return list(dict.fromkeys([d.metadata.get("source") or d.metadata.get("file_path") for d in docs]))

def build_rag_chain(retriever, chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer using only the provided context. "
                       "If the answer isn't in the context, say you don't know."),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )
    rag_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(answer=rag_from_docs)
        .assign(sources=lambda x: unique_sources(x["context"]))
    )

# --------------------------------------------------------------------------------------
# 4) Batch over S3 + save results (with step-by-step prints)
# --------------------------------------------------------------------------------------
def process_batch_s3(
    cfg: Dict[str, Any],
    question: str,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quant_4bit: bool = False,
    output_filename: str = "batch_results.json",
    max_files: int | None = 5,   # <--- NEW
) -> Dict[str, Any]:
    bucket = cfg["aws"]["s3_bucket_raw"]
    region = cfg["aws"]["region"]
    out_prefix = cfg["aws"]["output_prefix"].strip("/")

    groups = load_docs_from_s3_sources(cfg, max_files = max_files)
    if not groups:
        print("No .pdf/.html/.htm files found under edgar/ or site/")
        return {}

    print(f"Found {len(groups)} file(s) in s3://{bucket}/")
    chat_model = build_local_chat(model_id=model_id, quant_4bit=quant_4bit)

    results: Dict[str, Any] = {}
    for i, (s3_source, docs) in enumerate(groups.items(), 1):
        print(f"\n[{i}/{len(groups)}] Processing: {s3_source}")
        try:
            t0 = time.time()
            retriever = build_retriever_from_docs(docs)

            # ---- Preview what will be fed to the model (top-k retrieved)
            k = retriever.search_kwargs.get("k", 4)
            retrieved = retriever.get_relevant_documents(question)
            log(f"[RETRIEVE] k={k} | got={len(retrieved)}")
            for j, d in enumerate(retrieved, 1):
                src = d.metadata.get("source", "unknown")
                log(f"  - [{j}] {src} | {preview(d.page_content)}")

            # ---- Build chain and run
            rag = build_rag_chain(retriever, chat_model)
            result = rag.invoke(question)

            # ---- Print a short answer preview
            ans_preview = result["answer"]
            if isinstance(ans_preview, (list, dict)):
                ans_preview = json.dumps(ans_preview)  # in case your prompt returns JSON
            log(f"[ANSWER] {preview(ans_preview, 240)}")
            log(f"[TIME] {time.time()-t0:.1f}s for {s3_source}")

            results[s3_source] = {"status": "success", "answer": result["answer"], "sources": result["sources"]}
            print("✓ Completed")
        except Exception as e:
            results[s3_source] = {"status": "error", "error": str(e)}
            print(f"✗ Error: {e}")

    batch_results = {"total_files": len(groups), "results": results}

    s3 = boto3.client("s3", region_name=region)
    out_key = f"{out_prefix}/{output_filename}" if out_prefix else output_filename
    s3.put_object(
        Bucket=bucket,
        Key=out_key,
        Body=json.dumps(batch_results, indent=2, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )
    print(f"\nResults saved to s3://{bucket}/{out_key}")
    return batch_results

# --------------------------------------------------------------------------------------
# 5) CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--quant_4bit", action="store_true")
    parser.add_argument("--output", default="batch_results.json")
    parser.add_argument("--quiet", action="store_true", help="Reduce prints")
    parser.add_argument("--max_files", type=int, default=5, help="Max number of S3 files to process")

    args = parser.parse_args()

    if args.quiet:
        VERBOSE = False  # mute detailed logs

    claim_number = 3
    question = (
        f'You are a label creator of ESG report, read the report and then generate only {claim_number} claims related to the report, '
        'include verbatim evidence, and an answer among "support", "not support", or "don\'t know". Output pure JSON (no markdown).'
    )

    process_batch_s3(
    cfg=CFG,
    question=question,
    model_id=args.model_id,
    quant_4bit=args.quant_4bit,
    output_filename=args.output,
    max_files=args.max_files,   # <--- pass along
)
