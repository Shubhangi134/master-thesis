import os
import json
import re
import pandas as pd
import warnings
import logging
import ftfy
import shutil
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict
import faiss
import tiktoken


from openai import AzureOpenAI, AsyncAzureOpenAI
from openai import OpenAI as StandardOpenAI, AsyncOpenAI as StandardAsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory

# Retrieval 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ragas 
from ragas import evaluate
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    Faithfulness
)
from ragas.run_config import RunConfig
from datasets import Dataset

# Modern structured interfaces required by Ragas collections metrics

from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import SimpleParser

from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logging.getLogger("ragas").setLevel(logging.DEBUG)

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ==========================================
# 2. CONFIGURATION SWITCH
# ==========================================
USE_AZURE = bool(os.getenv("ENDPOINT"))

CONFIG = {
    # --- AZURE SETTINGS ---
    "azure_api_key": os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
    "azure_endpoint": os.getenv("ENDPOINT"),
    "azure_api_version": os.getenv("API_VERSION"), 
    
    # Deployment Names
    "azure_gen_deployment": os.getenv("MODEL_NAME"),
    "azure_judge_deployment": os.getenv("MODEL_NAME"),
    "azure_embed_deployment": "text-embedding-ada-002",
    
    # --- LOCAL SETTINGS ---
    "pdf_dir": r"raw_data/pdfs",
    "excel_path": r"raw_data/dataset/Questions_Answer.xlsx",
    "ollama_model": "mistral-large-3:675b-cloud",
    "local_embed_model": os.getenv("LOCAL_EMBED_MODEL", "mxbai-embed-large:latest"),
    "ollama_embed_model": os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest"),
    "reuse_llm_outputs": False,
    "llm_cache_path": "experiment_cache.json",
    "max_rows": 5,
        
    # --- RAGAS RUNTIME ---
    "ragas_timeout": 600,
    "ragas_max_workers": 2,
    "ragas_batch_size": 5,
    "ragas_max_wait": 30,
    "ragas_max_retries": 15,

    # Generic
    "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
    "overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 40)),
    "whoosh_index_dir": "whoosh_pdf_index",
    "rebuild_index": os.getenv("REBUILD_INDEX", True),
    "generation_prompt": None,  # Optional custom prompt template using {context} and {question}

    # Critical: Ollama Base URL for OpenAI compatibility
    "ollama_base_url": "http://localhost:11434/v1", 
    "ollama_model": "mistral-large-3:675b-cloud", # Your specific model name
    "ollama_api_key": "ollama", # Dummy key required by client

    # --- Hybrid Retrieval ---
    "faiss_index_path": "faiss_index.bin",
    "faiss_metadata_path": "faiss_metadata.json",
    "dense_top_k": int(os.getenv("DENSE_TOP_K", 40)),
    "hybrid_top_k": int(os.getenv("HYBRID_TOP_K", 5)),
    "rrf_k": int(os.getenv("RRF_K", 60)),
    "rebuild_dense_index": os.getenv("REBUILD_DENSE_INDEX", True),
    "embed_tpm_limit": int(os.getenv("EMBED_TPM_LIMIT", 200000)),
    "embed_rpm_limit": int(os.getenv("EMBED_RPM_LIMIT", 60)),
    "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", 32)),
}

DEFAULT_GENERATION_PROMPT = """You are an expert in automotive safety standards.
Answer the question in brief based ONLY on the following context.
Do not give explanations or additional information.
Output is expected to be in few words and not sentences.
If multiple answers are possible, separate them by semicolons or & symbol.
If the answer is not in the context, say "I do not know".

Context:
{context}

Question: {question}"""


def _get_int_config(name, default):
    """
    Safely coerce CONFIG entries to int with a fallback.
    """
    try:
        value = CONFIG.get(name, default)
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def log_step(name: str):
    """
    Print a uniform step marker for easier tracing.
    """
    print(f"[STEP] {name}")


def _get_tokenizer(model_name: str):
    """
    Choose a tokenizer for counting tokens ahead of embedding requests.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def _count_tokens(texts, tokenizer):
    if not tokenizer:
        return 0
    total = 0
    for text in texts:
        try:
            total += len(tokenizer.encode(text))
        except Exception:
            continue
    return total


def _batch_iterable(items, batch_size: int):
    """
    Yield fixed-size batches from a sequence while preserving order.
    """
    if batch_size is None or batch_size <= 0:
        yield list(items)
        return
    items = list(items)
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


class _TPMThrottle:
    """
    Simple per-process TPM throttler using a 60-second window.
    """
    def __init__(self, tpm_limit: int):
        self.tpm_limit = max(int(tpm_limit), 0) if tpm_limit is not None else 0
        self.window_start = time.time()
        self.tokens_used = 0

    def enforce(self, token_count: int):
        if self.tpm_limit <= 0 or token_count <= 0:
            return
        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= 60:
            self.window_start = now
            self.tokens_used = 0
        if self.tokens_used + token_count > self.tpm_limit:
            sleep_for = max(0.0, 60 - elapsed)
            if sleep_for > 0:
                print(f"[EMBED][THROTTLE] Sleeping {sleep_for:.2f}s to respect {self.tpm_limit} TPM")
                time.sleep(sleep_for)
            self.window_start = time.time()
            self.tokens_used = 0
        self.tokens_used += token_count


class _RPMThrottle:
    """
    Simple per-process RPM throttler using a 60-second window.
    """
    def __init__(self, rpm_limit: int):
        self.rpm_limit = max(int(rpm_limit), 0) if rpm_limit is not None else 0
        self.window_start = time.time()
        self.calls_made = 0

    def enforce(self):
        if self.rpm_limit <= 0:
            return
        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= 60:
            self.window_start = now
            self.calls_made = 0
        if self.calls_made >= self.rpm_limit:
            sleep_for = max(0.0, 60 - elapsed)
            if sleep_for > 0:
                print(f"[EMBED][RATE] Sleeping {sleep_for:.2f}s to respect {self.rpm_limit} RPM")
                time.sleep(sleep_for)
            self.window_start = time.time()
            self.calls_made = 0
        self.calls_made += 1


class AzureEmbedder:
    """
    Lightweight embedder wrapper for Azure OpenAI embeddings with an encode interface.
    """
    def __init__(self, deployment, endpoint, api_key, api_version, tpm_limit: int, rpm_limit: int, batch_size: int):
        self.deployment = deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            timeout=180.0
        )
        self.tokenizer = _get_tokenizer(deployment)
        self.throttle = _TPMThrottle(tpm_limit)
        self.rate_limiter = _RPMThrottle(rpm_limit)
        self.batch_size = max(int(batch_size), 1) if batch_size else None

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        all_vectors = []
        for batch in _batch_iterable(texts, self.batch_size):
            token_count = _count_tokens(batch, self.tokenizer)
            if token_count:
                print(f"[EMBED][AZURE] tokens={token_count} batch_size={len(batch)}")
            self.throttle.enforce(token_count)
            self.rate_limiter.enforce()
            resp = self.client.embeddings.create(model=self.deployment, input=batch)
            all_vectors.extend(item.embedding for item in resp.data)
        arr = np.array(all_vectors, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
        return arr


class LocalOllamaEmbedder:
    """
    Embedder wrapper for Ollama's OpenAI-compatible embeddings API.
    """
    def __init__(self, model, base_url, api_key, tpm_limit: int, rpm_limit: int, batch_size: int):
        self.model = model
        self.client = StandardOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=180.0
        )
        self.tokenizer = _get_tokenizer(model)
        self.throttle = _TPMThrottle(tpm_limit)
        self.rate_limiter = _RPMThrottle(rpm_limit)
        self.batch_size = max(int(batch_size), 1) if batch_size else None

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        all_vectors = []
        for batch in _batch_iterable(texts, self.batch_size):
            token_count = _count_tokens(batch, self.tokenizer)
            if token_count:
                print(f"[EMBED][OLLAMA] tokens={token_count} batch_size={len(batch)}")
            self.throttle.enforce(token_count)
            self.rate_limiter.enforce()
            resp = self.client.embeddings.create(model=self.model, input=batch)
            all_vectors.extend(item.embedding for item in resp.data)
        arr = np.array(all_vectors, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
        return arr

# ==========================================
# 3. MODEL FACTORY (ALL llm_factory)
# ==========================================
def get_models():
    print(f"Initializing Models... Mode: {'AZURE NATIVE' if USE_AZURE else 'LOCAL OLLAMA (via Factory)'}")
    
    ragas_judge = None
    ragas_embeddings = None
    generator_client = None 

    if USE_AZURE:
        # --- A. AZURE MODE ---
        # 1. Generator (Native Azure Client)
        generator_client = AzureOpenAI(
            api_key=CONFIG["azure_api_key"],
            api_version=CONFIG["azure_api_version"],
            azure_endpoint=CONFIG["azure_endpoint"],
            timeout=180.0
        )
        
        # 2. Judge (llm_factory with Azure Async Client)
        ragas_client = AsyncAzureOpenAI(
            api_key=CONFIG["azure_api_key"],
            api_version=CONFIG["azure_api_version"],
            azure_endpoint=CONFIG["azure_endpoint"],
            timeout=180.0
        )
        ragas_judge = llm_factory(
            model=CONFIG["azure_judge_deployment"],
            client=ragas_client
        )
        
        # 3. Embeddings (Ragas Native Azure)
        ragas_embeddings = RagasOpenAIEmbeddings(
            model=CONFIG["azure_embed_deployment"],
            client=ragas_client
        )
        
    else:
        # --- B. LOCAL MODE (Ollama via OpenAI-Compatible API) ---
        # 1. Generator (Standard OpenAI Client pointing to Ollama)
        generator_client = StandardOpenAI(
            base_url=CONFIG["ollama_base_url"],
            api_key=CONFIG["ollama_api_key"],
            timeout=180.0
        )
        
        # 2. Judge (llm_factory with Standard Async Client pointing to Ollama)
        ragas_client = StandardAsyncOpenAI(
            base_url=CONFIG["ollama_base_url"],
            api_key=CONFIG["ollama_api_key"],
            timeout=180.0
        )
        
        ragas_judge = llm_factory(
            model=CONFIG["ollama_model"],
            client=ragas_client
        )
        
        # 3. Embeddings (Ollama OpenAI-compatible embeddings)
        ragas_embeddings = RagasOpenAIEmbeddings(
            model=CONFIG.get("ollama_embed_model") or CONFIG["local_embed_model"],
            client=ragas_client
        )

    return {
        "generator_client": generator_client,
        "ragas_judge": ragas_judge,
        "ragas_embeddings": ragas_embeddings
    }

# ==========================================
# 4. GENERATION FUNCTION
# ==========================================
def generate_answer(client, context, question, prompt_template=None):
    """
    Generate an answer using the provided LLM client and context.

    prompt_template: Optional string with {context} and {question} placeholders.
    Falls back to CONFIG["generation_prompt"] or DEFAULT_GENERATION_PROMPT.
    """
    template = prompt_template or CONFIG.get("generation_prompt") or DEFAULT_GENERATION_PROMPT
    try:
        prompt = template.format(context=context, question=question)
    except Exception:
        prompt = DEFAULT_GENERATION_PROMPT.format(context=context, question=question)

    # Unified call for both Azure and Ollama (since both use OpenAI Client structure now)
    try:
        model_name = CONFIG["azure_gen_deployment"] if USE_AZURE else CONFIG["ollama_model"]
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Generation Error: {str(e)}"

# ==========================================
# 5. RETRIEVER SETUP
# ==========================================
def clean_pdf_text(text):
    """
    Normalize PDF text by removing headers/footers and collapsing whitespace.
    """
    if not text:
        return ""
    # Normalize line breaks
    normalized = text.replace("\r", "\n")
    cleaned_lines = []
    for line in normalized.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^page\s+\d+(\s*of\s*\d+)?$", stripped.lower()):
            continue
        if len(stripped) <= 40 and stripped.replace(" ", "").isupper():
            continue
        if re.match(r"^\d+$", stripped):
            continue
        cleaned_lines.append(stripped)
    collapsed = " ".join(cleaned_lines)
    collapsed = re.sub(r"\s+", " ", collapsed)
    return collapsed.strip()


@dataclass
class RetrievedDoc:
    page_content: str
    metadata: Dict


class WhooshBM25Retriever:
    def __init__(self, index_dir: str, k: int):
        self.index = open_dir(index_dir)
        self.k = k
        self.parser = SimpleParser("content", schema=self.index.schema)

    def invoke(self, query: str) -> List[RetrievedDoc]:
        if not query:
            return []
        print(f"[RETRIEVE][BM25] query='{query}' top_k={self.k}")
        try:
            parsed = self.parser.parse(query)
        except Exception:
            parsed = self.parser.parse(re.sub(r"[^\w\s]", " ", str(query)))
        with self.index.searcher() as searcher:
            hits = searcher.search(parsed, limit=self.k)
            results = []
            for rank, hit in enumerate(hits):
                doc_id = str(hit.get("doc_id", rank))
                results.append(
                    RetrievedDoc(
                        page_content=hit.get("content", ""),
                        metadata={
                            "source_file": hit.get("source_file", ""),
                            "doc_id": doc_id,
                            "rank": rank,
                            "score": float(hit.score) if hasattr(hit, "score") else None,
                        },
                    )
                )
            print(f"[RETRIEVE][BM25] returned={len(results)}")
            return results


def reciprocal_rank_fusion(result_sets: List[List[RetrievedDoc]], k: int = 60, max_results: int = None):
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    """
    fused = {}
    for result_set in result_sets:
        for rank, doc in enumerate(result_set):
            doc_id = doc.metadata.get("doc_id")
            if doc_id is None:
                continue
            key = str(doc_id)
            stored = fused.get(key, {"doc": doc, "score": 0.0})
            stored["score"] += 1.0 / (k + rank + 1)
            fused[key] = stored
    ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    combined = []
    for item in ordered:
        doc = item["doc"]
        doc.metadata["rrf_score"] = item["score"]
        combined.append(doc)
        if max_results and len(combined) >= max_results:
            break
    return combined


class DenseFAISSRetriever:
    def __init__(self, index, metadata_map: Dict[str, Dict], embedder, k: int):
        self.index = index
        self.metadata_map = metadata_map
        self.embedder = embedder
        self.k = k

    def invoke(self, query: str) -> List[RetrievedDoc]:
        if not query:
            return []
        print(f"[RETRIEVE][DENSE] query='{query}' top_k={self.k}")
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, self.k)
        results = []
        for rank, idx in enumerate(indices[0]):
            doc_id = str(idx)
            meta = self.metadata_map.get(doc_id)
            if not meta:
                continue
            results.append(
                RetrievedDoc(
                    page_content=meta.get("content", ""),
                    metadata={
                        "source_file": meta.get("source_file", ""),
                        "doc_id": doc_id,
                        "rank": rank,
                        "score": float(scores[0][rank]) if scores is not None else None,
                        },
                    )
                )
        print(f"[RETRIEVE][DENSE] returned={len(results)}")
        return results


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, rrf_k: int, top_k: int):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k

    def invoke(self, query: str) -> List[RetrievedDoc]:
        print(f"[RETRIEVE][HYBRID] query='{query}' hybrid_top_k={self.top_k} rrf_k={self.rrf_k}")
        result_sets = []
        if self.bm25_retriever:
            result_sets.append(self.bm25_retriever.invoke(query))
        if self.dense_retriever:
            result_sets.append(self.dense_retriever.invoke(query))
        if not result_sets:
            return []
        if len(result_sets) == 1:
            fused = result_sets[0][: self.top_k]
        else:
            fused = reciprocal_rank_fusion(result_sets, k=self.rrf_k, max_results=self.top_k)
        print(f"[RETRIEVE][HYBRID] fused_returned={len(fused)}")
        return fused


def prepare_pdf_chunks(pdf_dir):
    if not os.path.exists(pdf_dir):
        print("PDF Directory not found!")
        return []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["overlap"])

    all_chunks = []
    for filename in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            docs = loader.load()
            normalized_docs = []
            for doc in docs:
                doc.metadata["source_file"] = filename
                normalized_docs.append(doc)
            split_docs = text_splitter.split_documents(normalized_docs)
            for chunk in split_docs:
                text = chunk.page_content
                text = clean_pdf_text(text)
                text = ftfy.fix_text(text, fix_encoding=True, fix_entities=True)
                chunk.page_content = text
            all_chunks.extend(split_docs)
        except Exception:
            continue
    return all_chunks


def build_whoosh_index(chunks, index_dir: str, k: int):
    if not chunks:
        return None
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        source_file=TEXT(stored=True),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )
    index = create_in(index_dir, schema)
    writer = index.writer()
    for idx, chunk in enumerate(chunks):
        writer.add_document(
            doc_id=str(idx),
            source_file=str(chunk.metadata.get("source_file", "")),
            content=chunk.page_content,
        )
    writer.commit()
    return WhooshBM25Retriever(index_dir, k)


def build_dense_retriever(chunks, embedder, index_path: str, metadata_path: str, k: int):
    if not chunks:
        return None
    vectors = embedder.encode(
        [chunk.page_content for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir = os.path.dirname(index_path)
    if index_dir:
        os.makedirs(index_dir, exist_ok=True)
    meta_dir = os.path.dirname(metadata_path)
    if meta_dir:
        os.makedirs(meta_dir, exist_ok=True)

    faiss.write_index(index, index_path)
    metadata_payload = []
    for idx, chunk in enumerate(chunks):
        metadata_payload.append({
            "doc_id": str(idx),
            "source_file": str(chunk.metadata.get("source_file", "")),
            "content": chunk.page_content,
        })
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
    metadata_map = {str(item["doc_id"]): item for item in metadata_payload}
    return DenseFAISSRetriever(index, metadata_map, embedder, k)


def load_dense_retriever(embedder, index_path: str, metadata_path: str, k: int):
    if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
        return None
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_payload = json.load(f)
        if not isinstance(metadata_payload, list):
            return None
        metadata_map = {str(item.get("doc_id")): item for item in metadata_payload}
        return DenseFAISSRetriever(index, metadata_map, embedder, k)
    except Exception as exc:
        print(f"Failed to load FAISS index, rebuilding: {exc}")
        return None


def load_and_build_retriever(pdf_dir):
    index_dir = CONFIG.get("whoosh_index_dir", "whoosh_pdf_index")
    rebuild_bm25 = CONFIG.get("rebuild_index", True)
    rebuild_dense = CONFIG.get("rebuild_dense_index", False)
    faiss_index_path = CONFIG.get("faiss_index_path", "faiss_index.bin")
    faiss_metadata_path = CONFIG.get("faiss_metadata_path", "faiss_metadata.json")

    bm25_top_k = _get_int_config("top_k", 5)
    dense_top_k = _get_int_config("dense_top_k", bm25_top_k)
    hybrid_top_k = _get_int_config("hybrid_top_k", bm25_top_k)
    rrf_k = _get_int_config("rrf_k", 60)
    embed_tpm_limit = _get_int_config("embed_tpm_limit", 120000)
    embed_rpm_limit = _get_int_config("embed_rpm_limit", 60)
    embed_batch_size = _get_int_config("embed_batch_size", 16)

    bm25_ready = exists_in(index_dir)
    dense_ready = os.path.exists(faiss_index_path) and os.path.exists(faiss_metadata_path)
    need_chunks = rebuild_bm25 or rebuild_dense or not bm25_ready or not dense_ready

    chunks = prepare_pdf_chunks(pdf_dir) if need_chunks else []
    if need_chunks and not chunks:
        return None

    bm25_retriever = None
    if rebuild_bm25 or not bm25_ready:
        bm25_retriever = build_whoosh_index(chunks, index_dir, bm25_top_k)
    else:
        bm25_retriever = WhooshBM25Retriever(index_dir, bm25_top_k)

    dense_retriever = None
    embedder = None
    if USE_AZURE:
        try:
            log_step("Init Azure embeddings")
            embedder = AzureEmbedder(
                deployment=CONFIG["azure_embed_deployment"],
                endpoint=CONFIG["azure_endpoint"],
                api_key=CONFIG["azure_api_key"],
                api_version=CONFIG["azure_api_version"],
                tpm_limit=embed_tpm_limit,
                rpm_limit=embed_rpm_limit,
                batch_size=embed_batch_size,
            )
        except Exception as exc:
            print(f"Failed to load Azure embeddings: {exc}")
            embedder = None
    else:
        try:
            log_step("Init Ollama embeddings")
            embedder = LocalOllamaEmbedder(
                model=CONFIG.get("ollama_embed_model") or CONFIG["local_embed_model"],
                base_url=CONFIG["ollama_base_url"],
                api_key=CONFIG["ollama_api_key"],
                tpm_limit=embed_tpm_limit,
                rpm_limit=embed_rpm_limit,
                batch_size=embed_batch_size,
            )
        except Exception as exc:
            print(f"Failed to load Ollama embeddings: {exc}")
            embedder = None

    if embedder:
        if rebuild_dense or not dense_ready:
            if not chunks:
                chunks = prepare_pdf_chunks(pdf_dir)
            dense_retriever = build_dense_retriever(chunks, embedder, faiss_index_path, faiss_metadata_path, dense_top_k)
        else:
            dense_retriever = load_dense_retriever(embedder, faiss_index_path, faiss_metadata_path, dense_top_k)

    if not bm25_retriever and not dense_retriever:
        return None
    if not bm25_retriever:
        return dense_retriever
    return HybridRetriever(bm25_retriever, dense_retriever, rrf_k, hybrid_top_k)


def _normalize_doc_label(name):
    """
    Normalize document identifiers for robust comparison.
    """
    if not name:
        return ""
    value = os.path.basename(str(name).strip()).lower()
    return value[:-4] if value.endswith(".pdf") else value


def _parse_target_documents(raw_value):
    """
    Split the target PDF column into individual document names.
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []
    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"[;,]", text)
    return [part.strip() for part in parts if part.strip()]


def _compute_retriever_metrics(target_docs, retrieved_docs):
    """
    Compute precision, recall, and F1 for retrieved documents vs. ground-truth targets.
    """
    target_set = {_normalize_doc_label(doc) for doc in target_docs}
    retrieved_set = {_normalize_doc_label(doc) for doc in retrieved_docs}
    target_set.discard("")
    retrieved_set.discard("")
    if not retrieved_set:
        return 0.0, 0.0, 0.0
    if not target_set:
        return 0.0, 0.0, 0.0
    tp = len(target_set & retrieved_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(retrieved_set)
    recall = tp / len(target_set)
    if precision == 0 or recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _ensure_retriever_metrics(record):
    """
    Return precision/recall/F1 stored on a record or recompute if missing.
    """
    prec = record.get('retriever_precision')
    rec = record.get('retriever_recall')
    f1 = record.get('retriever_f1')
    if prec is not None and rec is not None and f1 is not None:
        return prec, rec, f1
    target_docs = _parse_target_documents(record.get('target_pdf', ''))
    found_files = record.get('found_files', [])
    return _compute_retriever_metrics(target_docs, found_files)


# ==========================================
# 6. EXPERIMENT LOOP
# ==========================================
def run_experiment(excel_path, retriever, generator_client):
    if not os.path.exists(excel_path):
        print("Excel file not found!")
        return []
        
    df = pd.read_excel(excel_path)
    max_rows = CONFIG.get("max_rows")
    if isinstance(max_rows, int) and max_rows > 0:
        df = df.head(max_rows)
    print(f"Loaded {len(df)} questions.")
    results = []
    
    for index, row in df.iterrows():
        # Column Mapping
        q_col = 'Questions' if 'Questions' in df.columns else 'Question'
        gt_col = 'Ground_truth_answer' if 'Ground_truth_answer' in df.columns else 'Answer'
        pdf_col = 'Relevant_pdfs' if 'Relevant_pdfs' in df.columns else 'Source_PDF'
        
        q = row[q_col]
        gt_raw = row[gt_col]
        raw_pdf_value = row.get(pdf_col, "")
        pdf_target = "" if pd.isna(raw_pdf_value) else str(raw_pdf_value)
        target_docs = _parse_target_documents(raw_pdf_value)
        
        # Semicolon Logic
        gt_formatted = str(gt_raw)

        print(f"Processing Q{index+1}...")
        
        # Retrieve
        docs = retriever.invoke(q)
        ctx_list = [d.page_content for d in docs]
        found_files = [d.metadata.get("source_file", "") for d in docs]
        
        # Generate
        ans = generate_answer(generator_client, "\n\n".join(ctx_list), q)
            
        retriever_precision, retriever_recall, retriever_f1 = _compute_retriever_metrics(
            target_docs,
            found_files
        )
        
        results.append({
            "question": q,
            "answer": ans,
            "contexts": ctx_list,
            "ground_truth": gt_formatted,
            "target_pdf": pdf_target,
            "found_files": found_files,
            "retriever_f1": retriever_f1,
            "retriever_precision": retriever_precision,
            "retriever_recall": retriever_recall,
        })

    return results


def load_cached_experiment(cache_path):
    """
    Load previously generated experiment data so we can skip re-querying LLMs.
    """
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            print(f"Loaded {len(data)} cached responses from {cache_path}.")
            return data
    except Exception as exc:
        print(f"Failed to load cache {cache_path}: {exc}")
    return None


def save_experiment_cache(experiment_data, cache_path):
    """
    Persist experiment data for future reuse of LLM responses.
    """
    if not cache_path:
        return
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)
        print(f"Cached {len(experiment_data)} responses to {cache_path}.")
    except Exception as exc:
        print(f"Failed to save cache {cache_path}: {exc}")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Init
    log_step("Initialize models and retrievers")
    models = get_models()
    retriever = load_and_build_retriever(CONFIG["pdf_dir"])
    if not retriever: exit()

    # 2. Run
    log_step("Load cache or run experiment")
    cache_path = CONFIG.get("llm_cache_path")
    experiment_data = None
    if CONFIG.get("reuse_llm_outputs"):
        experiment_data = load_cached_experiment(cache_path)

    if experiment_data is None:
        log_step("Run experiment")
        experiment_data = run_experiment(CONFIG["excel_path"], retriever, models["generator_client"])
        if not experiment_data: exit()
        if CONFIG.get("reuse_llm_outputs"):
            save_experiment_cache(experiment_data, cache_path)
    if not experiment_data:
        exit()

    # 3. Prepare Dataset
    log_step("Prepare Ragas dataset")
    ragas_ds = Dataset.from_dict({
        "question": [str(x["question"]) for x in experiment_data],
        "answer": [str(x["answer"]).lower() if x.get("answer") else "" for x in experiment_data],
        "contexts": [x["contexts"] for x in experiment_data],
        "ground_truth": [
            str(x["ground_truth"]).lower() if x.get("ground_truth") else "" for x in experiment_data
        ]
    })
    ragas_metrics = [
        ContextPrecision(llm=models["ragas_judge"]),
        Faithfulness(llm=models["ragas_judge"]),
        AnswerCorrectness(
            llm=models["ragas_judge"],
            embeddings=models["ragas_embeddings"],
        )
    ]
    ragas_run_config = RunConfig(
        timeout=CONFIG["ragas_timeout"],
        max_workers=CONFIG["ragas_max_workers"],
        max_retries=CONFIG["ragas_max_retries"],
        max_wait=CONFIG["ragas_max_wait"]
    )

    # 4. Evaluate
    log_step("Run evaluation")
    scores = evaluate(
        dataset=ragas_ds,
        metrics=ragas_metrics,
        run_config=ragas_run_config,
        batch_size=CONFIG["ragas_batch_size"]
    )

    # 5. Save
    log_step("Save results")
    df_out = scores.to_pandas()
    retriever_precisions = []
    retriever_recalls = []
    retriever_f1s = []
    for record in experiment_data:
        prec, rec, f1 = _ensure_retriever_metrics(record)
        retriever_precisions.append(prec)
        retriever_recalls.append(rec)
        retriever_f1s.append(f1)
    df_out['retriever_precision'] = retriever_precisions
    df_out['retriever_recall'] = retriever_recalls
    df_out['retriever_f1'] = retriever_f1s
    df_out['target_pdf'] = [x['target_pdf'] for x in experiment_data]
    df_out['found_files'] = [str(x['found_files']) for x in experiment_data]

    fname = "Results_Native_Azure.csv" if USE_AZURE else "Results_Local.csv"
    while True:
        try:
            df_out.to_csv(fname, index=False)
            break
        except PermissionError:
            input(f"Please close {fname} and press Enter to retry...")

    print(f"\nSaved to {fname}")
    print(f"Avg Retriever F1: {df_out['retriever_f1'].mean():.2f}")
    print(scores)
