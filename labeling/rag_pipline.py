import os
import json
from typing import List, Dict, Any
from pathlib import Path

# 1) Load and split the PDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 2) Embeddings + vector store (embeddings run locally)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 3) Local LLM via Transformers pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 4) RAG chain pieces
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


def get_pdf_files(folder_path: str) -> List[str]:
    """Get all PDF files from a folder."""
    pdf_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(folder_path, file))
    return sorted(pdf_files)


def build_retriever(
    pdf_path: str,
    embed_model: str = "intfloat/e5-base-v2",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    k: int = 4,
):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=embed_model, model_kwargs={"device": "cuda"})
    vs = FAISS.from_documents(splits, embeddings)
    return vs.as_retriever(search_kwargs={"k": k})


def build_local_chat(
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
    quant_4bit: bool = False,
):
    import torch
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    quantization_config = None
    if quant_4bit:
        # requires: pip install bitsandbytes
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,  # needs accelerate
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
    chat = ChatHuggingFace(llm=llm)
    return chat


def format_docs(docs) -> str:
    return "\n\n".join(f"[{i+1}] " + d.page_content for i, d in enumerate(docs))


def unique_sources(docs) -> List[str]:
    return list(
        dict.fromkeys([d.metadata.get("source") or d.metadata.get("file_path") for d in docs])
    )


def build_rag_chain(retriever, chat_model):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer using only the provided context. "
                "If the answer isn't in the context, say you don't know.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    rag_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | chat_model
        | StrOutputParser()
    )

    rag_with_sources = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(answer=rag_from_docs)
        .assign(sources=lambda x: unique_sources(x["context"]))
    )
    return rag_with_sources


def answer_question(
    pdf_path: str,
    question: str,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quant_4bit: bool = False,
) -> Dict[str, Any]:
    """Answer a question for a single PDF."""
    print(f"Processing: {os.path.basename(pdf_path)}")
    retriever = build_retriever(pdf_path)
    chat_model = build_local_chat(model_id=model_id, quant_4bit=quant_4bit)
    rag = build_rag_chain(retriever, chat_model)
    result = rag.invoke(question)
    return {"answer": result["answer"], "sources": result["sources"]}


def process_batch_pdfs(
    folder_path: str,
    question: str,
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quant_4bit: bool = False,
    output_file: str = "batch_results.json",
) -> Dict[str, Any]:
    """Process multiple PDFs with the same question."""
    
    pdf_files = get_pdf_files(folder_path)
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return {}
    
    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    
    results = {}
    batch_results = {
        "total_files": len(pdf_files),
        "results": results,
    }
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
            result = answer_question(pdf_path, question, model_id=model_id, quant_4bit=quant_4bit)
            
            file_name = os.path.basename(pdf_path)
            results[file_name] = {
                "status": "success",
                "answer": result["answer"],
                "sources": result["sources"],
            }
            
            print(f"✓ Completed")
            
        except Exception as e:
            print(f"✗ Error processing {os.path.basename(pdf_path)}: {str(e)}")
            file_name = os.path.basename(pdf_path)
            results[file_name] = {
                "status": "error",
                "error": str(e),
            }
    
    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nResults saved to: {output_file}")
    return batch_results


def print_batch_results(batch_results: Dict[str, Any]) -> None:
    """Pretty print batch results."""
    print("\n" + "="*80)
    print("BATCH PROCESSING RESULTS")
    print("="*80)
    print(f"Total Files Processed: {batch_results['total_files']}\n")
    
    for file_name, result in batch_results['results'].items():
        print(f"\n{'─'*80}")
        print(f"File: {file_name}")
        print(f"{'─'*80}")
        
        if result['status'] == 'success':
            print(f"Answer:\n{result['answer']}\n")
            print(f"Sources: {', '.join(result['sources'])}")
        else:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local RAG over batch of PDFs using Hugging Face + LangChain")
    
    parser.add_argument(
        "--folder",
        default="./pdfs",
        help="Path to folder containing PDF files (default: ./pdfs)"
    )
    parser.add_argument(
        "--model_id",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model repo (e.g., Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-Instruct-v0.3, meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--quant_4bit",
        action="store_true",
        help="Enable 4-bit quantization (GPU + bitsandbytes)"
    )
    parser.add_argument(
        "--output",
        default="batch_results.json",
        help="Output JSON file for results (default: batch_results.json)"
    )
    
    args = parser.parse_args()
    
    # Create folder if it doesn't exist
    if not os.path.exists(args.folder):
        print(f"Creating folder: {args.folder}")
        os.makedirs(args.folder)
        print(f"Please add your PDF files to: {args.folder}")
    
    claim_number = 3
    question = f"""
    You are a label creator of ESG report, read the report and then generate **only {claim_number}** of claims relate to the report,
    with evidence from pdf, and also a answer from "support", "not support", or "don't know", output your answer in json format
    no markdown grammar
    """
    
    # Process batch of PDFs
    batch_results = process_batch_pdfs(
        folder_path=args.folder,
        question=question,
        model_id=args.model_id,
        quant_4bit=args.quant_4bit,
        output_file=args.output
    )
    
    # Print results
    print_batch_results(batch_results)