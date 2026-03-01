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
from typing import List, Dict, Tuple
from urllib.parse import unquote
import faiss
import tiktoken
from tqdm import tqdm
from sentence_transformers import CrossEncoder

# --- 1. IMPORTS ---

from openai import AzureOpenAI, AsyncAzureOpenAI
from openai import OpenAI as StandardOpenAI, AsyncOpenAI as StandardAsyncOpenAI


from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory

# Retrieval 
from langchain_community.document_loaders import PyPDFLoader
# Ragas 
from ragas import evaluate
from ragas.metrics import (
    AnswerCorrectness,
    ContextPrecision,
    Faithfulness,
    # ExactMatch,
    # _ChrfScore
)
from ragas.run_config import RunConfig
from datasets import Dataset, load_dataset


from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import SimpleParser
from whoosh.query import Term, Or

from dotenv import load_dotenv

from query_expansion_helper import expand_query_for_bm25, expand_query_for_dense

from hopping import make_hopping_invoke


load_dotenv(".env")

# Enable verbose Ragas logging for easier debugging
logging.basicConfig(level=logging.INFO)
logging.getLogger("ragas").setLevel(logging.DEBUG)

# Suppress Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ==========================================
# 2. CONFIGURATION SWITCH
# ==========================================
USE_AZURE = bool(os.getenv("ENDPOINT"))
DEFAULT_WIKI_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wiki_data")

CONFIG = {
    "doc_mode": os.getenv("DOC_MODE", "pdf").lower(),  # pdf | frames
    
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
    "frames_dir": os.getenv("FRAMES_DIR", r"wiki_data"),
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
    "chunk_size": int(os.getenv("CHUNK_SIZE", 800)),
    "overlap": int(os.getenv("CHUNK_OVERLAP", 150)),
    "chunk_size_pdf": int(os.getenv("CHUNK_SIZE_PDF", os.getenv("CHUNK_SIZE", 1300))),
    "chunk_overlap_pdf": int(os.getenv("CHUNK_OVERLAP_PDF", os.getenv("CHUNK_OVERLAP", 250))),
    "chunk_size_wiki": int(os.getenv("CHUNK_SIZE_WIKI", os.getenv("CHUNK_SIZE", 1100))),
    "chunk_overlap_wiki": int(os.getenv("CHUNK_OVERLAP_WIKI", os.getenv("CHUNK_OVERLAP", 200))),
    "top_k": int(os.getenv("SPARSE_TOP_K", os.getenv("RETRIEVER_TOP_K", 40))),
    "whoosh_index_dir": "whoosh_pdf_index",
    "whoosh_frames_index_dir": os.getenv("WHOOSH_FRAMES_INDEX_DIR", "whoosh_wiki_index"),
    "whoosh_limit_mb": int(os.getenv("WHOOSH_LIMIT_MB", 1024)),
    "wiki_index_filename": os.getenv("WIKI_INDEX_FILENAME", "index.json"),
    "wiki_cache_dir": os.getenv("WIKI_CACHE_DIR", DEFAULT_WIKI_CACHE_DIR),
    "rebuild_index": int(os.getenv("REBUILD_SPARSE_INDEX", os.getenv("REBUILD_INDEX", 1))),
    "rebuild_frames_index": int(os.getenv("REBUILD_FRAMES_INDEX", os.getenv("REBUILD_SPARSE_INDEX", os.getenv("REBUILD_INDEX", 1)))),
    "generation_prompt": None,  # Optional custom prompt template using {context} and {question}

    # Critical: Ollama Base URL for OpenAI compatibility
    "ollama_base_url": "http://localhost:11434/v1", 
    "ollama_model": "gpt-oss:120b-cloud", # Your specific model name
    "ollama_api_key": "ollama", # Dummy key required by client

    # --- Hybrid Retrieval ---
    "faiss_index_path": "faiss_index.bin",
    "faiss_metadata_path": "faiss_metadata.json",
    "faiss_frames_index_path": os.getenv("FAISS_FRAMES_INDEX_PATH", "faiss_wiki_index.bin"),
    "faiss_frames_metadata_path": os.getenv("FAISS_FRAMES_METADATA_PATH", "faiss_wiki_metadata.json"),
    "dense_top_k": int(os.getenv("DENSE_TOP_K", 40)),
    "hybrid_top_k": int(os.getenv("HYBRID_TOP_K", 5)),
    "rrf_top_k": int(os.getenv("RRF_TOP_K", 60)),
    "rebuild_dense_index": int(os.getenv("REBUILD_DENSE_INDEX", True)),
    "rebuild_dense_frames_index": int(os.getenv("REBUILD_DENSE_FRAMES_INDEX", os.getenv("REBUILD_DENSE_INDEX", True))),
    "embed_tpm_limit": int(os.getenv("EMBED_TPM_LIMIT", 200000)),
    "embed_rpm_limit": int(os.getenv("EMBED_RPM_LIMIT", 60)),
    "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", 64)),

    # --- RERANKER / LLM RERANKING SETTINGS ---
    "enable_reranker": os.getenv("ENABLE_RERANKER", "1"),
    "cross_encoder_model": os.getenv("CROSS_ENCODER_MODEL_NAME"),
    "rerank_batch_size": int(os.getenv("RERANK_BATCH_SIZE", 5)),

    # Query Expansion
    "abbrev_map_path": os.getenv("ABBREV_MAP_PATH", "abbreviations.json"),
    "bm25_enable_query_expansion": int(os.getenv("BM25_ENABLE_QUERY_EXPANSION", 0)),
    "bm25_max_query_expansions": int(os.getenv("BM25_MAX_QUERY_EXPANSIONS", 10)),
    "dense_enable_query_expansion": int(os.getenv("DENSE_ENABLE_QUERY_EXPANSION", 0)),

    # --- HOPPING (Query-Rewrite Multi-hop) ---
    "enable_hopping": int(os.getenv("ENABLE_HOPPING", 0)),
    "hop_max_hops": int(os.getenv("HOP_MAX_HOPS", 2)),
    "hop_evidence_docs": int(os.getenv("HOP_EVIDENCE_DOCS", 8)),
    "hop_evidence_chars": int(os.getenv("HOP_EVIDENCE_CHARS", 1200)),
    "hop_query_max_tokens": int(os.getenv("HOP_QUERY_MAX_TOKENS", 40)),

    # --- FRAMES / WIKIPEDIA ---
    "frames_dataset": os.getenv("FRAMES_DATASET", "google/frames-benchmark"),
    "frames_split": os.getenv("FRAMES_SPLIT", "test"),
    "frames_max_wiki_titles": int(os.getenv("FRAMES_MAX_WIKI_TITLES", 0)),

    # Question selection
    "question_range": os.getenv("QUESTION_RANGE", None),  # e.g., "1-10" (1-based, inclusive)

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

# Frames-specific concise QA prompt
FRAMES_GENERATION_PROMPT = """Answer the question based ONLY on the provided context.
Keep it short and factual. If not answerable from context, say "I don't know".

Context:
{context}

Question: {question}"""

_cross_encoder_model = None
CROSS_ENCODER_DEFAULT = "cross-encoder/ms-marco-MiniLM-L6-v2"
LOCAL_CROSS_ENCODER_DIR = os.path.join("models", "reranker-ms-marco-MiniLM-L6-v2")

def _get_int_config(name, default):
    """
    Safely coerce CONFIG entries to int with a fallback.
    """
    try:
        value = CONFIG.get(name, default)
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _get_bool_config(name, default=False):
    """
    Safely coerce CONFIG entries to bool with a fallback.
    """
    value = CONFIG.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value) if value is not None else bool(default)


def _parse_question_range(value: str | None):
    """
    Parse QUESTION_RANGE env (e.g., "1-10" or "5") into (start, end) 1-based inclusive.
    Returns None if not provided or invalid.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if "-" in text:
            start_s, end_s = text.split("-", 1)
            start = int(start_s)
            end = int(end_s)
        else:
            start = end = int(text)
        if start <= 0 or end <= 0 or end < start:
            return None
        return start, end
    except Exception:
        return None


def _apply_question_range(seq: List, qrange):
    """
    Apply (start, end) 1-based inclusive slice to a list-like sequence.
    """
    if not qrange or not seq:
        return seq
    start, end = qrange
    start_idx = max(0, start - 1)
    end_idx = min(len(seq), end)
    return seq[start_idx:end_idx]


def log_config_overview():
    """
    Log key configuration settings for visibility at startup.
    """
    local_ce_dir = os.path.join("models", "reranker-ms-marco-MiniLM-L6-v2")
    logging.info("[CONFIG] Mode: %s", "AZURE" if USE_AZURE else "LOCAL/Ollama")
    logging.info(
        "[CONFIG] Retrieval: top_k=%s dense_top_k=%s hybrid_top_k=%s rrf_top_k=%s",
        _get_int_config("top_k", 40),
        _get_int_config("dense_top_k", 40),
        _get_int_config("hybrid_top_k", 5),
        _get_int_config("rrf_top_k", 60),
    )
    logging.info("[CONFIG] Doc mode: %s", CONFIG.get("doc_mode", "pdf"))
    logging.info("[CONFIG] PDF directory: %s", CONFIG.get("pdf_dir"))
    logging.info("[CONFIG] Frames directory: %s", CONFIG.get("frames_dir"))
    logging.info("[CONFIG] Wiki cache dir: %s", CONFIG.get("wiki_cache_dir", DEFAULT_WIKI_CACHE_DIR))
    logging.info("[CONFIG] Whoosh (pdf/frames): %s | %s", CONFIG.get("whoosh_index_dir"), CONFIG.get("whoosh_frames_index_dir"))
    logging.info("[CONFIG] FAISS (pdf/frames): %s | %s", CONFIG.get("faiss_index_path"), CONFIG.get("faiss_frames_index_path"))
    logging.info(
        "[CONFIG] Reranker: enabled=%s batch_size=%s configured_model=%s local_default=%s",
        _get_bool_config("enable_reranker", True),
        _get_int_config("rerank_batch_size", 5),
        CONFIG.get("cross_encoder_model") or "(not set)",
        local_ce_dir,
    )


log_config_overview()

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


def _load_wiki_cache_index(index_filename: str) -> Dict[str, str]:
    """
    Load wiki_cache_dir/index.json mapping of (link/title -> filename).
    Returns dict: normalized_key -> absolute_file_path.
    Supports:
      - {"<link>": "<file>", ...}
      - [{"link": "<link>", "filename": "<file>"}, ...]
    Also stores a derived title key for each link: _url_to_title(link)
    """
    cache_dir = CONFIG.get("wiki_cache_dir", DEFAULT_WIKI_CACHE_DIR)
    index_path = os.path.join(cache_dir, index_filename)
    if not os.path.exists(index_path):
        logging.warning("Wiki cache index not found at %s", index_path)
        return {}

    try:
        with open(index_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logging.warning("Failed reading wiki cache index (%s): %s", index_path, exc)
        return {}

    pairs: List[Tuple[str, str]] = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(k, str) and isinstance(v, str):
                pairs.append((k, v))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            link = item.get("link") or item.get("url")
            fname = item.get("filename") or item.get("file")
            if isinstance(link, str) and isinstance(fname, str):
                pairs.append((link, fname))

    norm_to_path: Dict[str, str] = {}
    for link, fname in pairs:
        abs_path = os.path.join(cache_dir, fname)

        # Key by link (primary)
        norm_to_path[_normalize_title(link)] = abs_path

        # Also key by title derived from link
        derived_title = _url_to_title(link)
        if derived_title:
            norm_to_path[_normalize_title(derived_title)] = abs_path

    return norm_to_path


def _load_frames_articles_from_dir(frames_dir: str) -> List[Dict]:
    """
    Load plain-text articles from frames_dir.
    Prefers index.json for titles; otherwise uses filename stem.
    """
    articles: List[Dict] = []
    cache_index = _load_wiki_cache_index(CONFIG.get("wiki_index_filename", "index.json"))
    if cache_index:
        seen_paths = set()
        for path in cache_index.values():
            if not path or path in seen_paths or not os.path.exists(path):
                continue
            seen_paths.add(path)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            articles.append({
                "title": os.path.splitext(os.path.basename(path))[0],
                "text": text,
                "source_file": os.path.basename(path),
            })
        if articles:
            return articles

    # Fallback: load every .txt file
    if not os.path.exists(frames_dir):
        return []
    for fname in os.listdir(frames_dir):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(frames_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        articles.append({
            "title": os.path.splitext(fname)[0],
            "text": text,
            "source_file": fname,
        })
    return articles


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
        total = len(texts) if hasattr(texts, "__len__") else None
        with tqdm(total=total, desc="[EMBED][AZURE]", unit="vec", disable=total is None) as pbar:
            for batch in _batch_iterable(texts, self.batch_size):
                token_count = _count_tokens(batch, self.tokenizer)
                self.throttle.enforce(token_count)
                self.rate_limiter.enforce()
                resp = self.client.embeddings.create(model=self.deployment, input=batch)
                all_vectors.extend(item.embedding for item in resp.data)
                if total is not None:
                    pbar.update(len(batch))
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
        total = len(texts) if hasattr(texts, "__len__") else None
        with tqdm(total=total, desc="[EMBED][OLLAMA]", unit="vec", disable=total is None) as pbar:
            for batch in _batch_iterable(texts, self.batch_size):
                token_count = _count_tokens(batch, self.tokenizer)
                self.throttle.enforce(token_count)
                self.rate_limiter.enforce()
                resp = self.client.embeddings.create(model=self.model, input=batch)
                all_vectors.extend(item.embedding for item in resp.data)
                if total is not None:
                    pbar.update(len(batch))
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
        
        # 3. Embeddings (Ollama/OpenAI-compatible embeddings) for Ragas
        ragas_embeddings = RagasOpenAIEmbeddings(
            model=CONFIG.get("ollama_embed_model") or CONFIG["ollama_model"],
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
    if prompt_template:
        template = prompt_template
    else:
        if str(CONFIG.get("doc_mode", "pdf")).strip().lower() == "frames":
            template = CONFIG.get("generation_prompt") or FRAMES_GENERATION_PROMPT
        else:
            template = CONFIG.get("generation_prompt") or DEFAULT_GENERATION_PROMPT
    try:
        prompt = template.format(context=context, question=question)
    except Exception:
        prompt = DEFAULT_GENERATION_PROMPT.format(context=context, question=question)

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
        ans = response.choices[0].message.content
        ans = ftfy.fix_text(ans, normalization="NFKC")
        ans = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", ans)
        return ans
    except Exception as e:
        return f"Generation Error: {str(e)}"

# ==========================================
# 5. RETRIEVER SETUP
# ==========================================
def clean_encoding_artifacts(text: str) -> str:
    if not text:
        return ""

    # 1. Common mojibake replacements (fast & explicit)
    replacements = {
        "â€¢": "•",
        "Â©": "©",
        "Â®": "®",
        "Â": "",        # stray Latin-1 marker
        "â€“": "–",
        "â€”": "—",
        "â€œ": '"',
        "â€": '"',
        "â€˜": "'",
        "â€™": "'",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # 2. Remove isolated control characters
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)

    # 3. Collapse weird spacing
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_pdf_text(text):
    """
    PDF cleanup:
      - Fix hyphenation across line breaks (inter-\nnational -> international)
      - Remove obvious page headers/footers and repeated separators
      - Collapse broken spacing ("E x a m p l e" -> "Example")
      - Preserve punctuation/casing
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r", "\n")

    # Fix hyphenation at line breaks
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Remove obvious header/footer/separator lines
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # page headers/footers like "Page 3" or "3"
        if re.match(r"^page\s+\d+(\s*of\s*\d+)?$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\d+$", stripped):
            continue
        if re.match(r"^[\.\-\=]{5,}$", stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Collapse broken spacing like "E x a m p l e"
    text = re.sub(r"(?<=\b\w) (?!\s)(?=\w\b)", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = ftfy.fix_text(text.strip(), normalization="NFKC")
    return text


def clean_wiki_text(text: str) -> str:
    """
    Lightweight cleaning for plain-text Wikipedia (preserve punctuation/casing).
    """
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return ftfy.fix_text(text.strip(), normalization="NFKC")


def chunk_text_fixed(text: str, *, chunk_size_chars: int, overlap_chars: int) -> List[Dict]:
    """
    Deterministic fixed-size character chunking with overlap.
    Returns list of dicts: {"text": str, "char_start": int, "char_end": int}
    """
    if not text:
        return []
    chunk_size = max(1, int(chunk_size_chars))
    overlap = max(0, int(overlap_chars))
    step = max(1, chunk_size - overlap)
    chunks = []
    n = len(text)
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({
            "text": text[start:end],
            "char_start": start,
            "char_end": end
        })
        idx += 1
        start += step
    return chunks


def _normalize_title(value: str) -> str:
    """
    Normalize wikipedia titles/links for comparison.
    """
    if not value:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _url_to_title(url: str) -> str:
    if not url:
        return ""
    marker = "/wiki/"
    if marker in url:
        url = url.split(marker, 1)[-1]
    url = url.split("#", 1)[0].split("?", 1)[0]
    url = unquote(url)
    return url.replace("_", " ").strip()


def _extract_links(row: Dict) -> List[str]:
    links = set()
    for key, value in row.items():
        if key.startswith("wikipedia_link_") and isinstance(value, str) and value.startswith("http"):
            links.add(value)

    raw_links = row.get("wiki_links")
    if isinstance(raw_links, str) and raw_links.strip().startswith("["):
        try:
            import ast
            parsed = ast.literal_eval(raw_links)
            if isinstance(parsed, list):
                links.update([x for x in parsed if isinstance(x, str) and x.startswith("http")])
        except (ValueError, SyntaxError):
            pass
    return list(links)

def prepare_pdf_chunks(pdf_dir):
    """
    Load PDFs, clean text, and fixed-size character chunk.
    """
    if not os.path.exists(pdf_dir):
        print("PDF Directory not found!")
        return []

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDFs.")

    chunk_size = _get_int_config("chunk_size_pdf", CONFIG["chunk_size"])
    overlap = _get_int_config("chunk_overlap_pdf", CONFIG["overlap"])

    all_chunks = []
    for filename in tqdm(pdf_files, desc="PDF files", unit="file"):
        try:
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            docs = loader.load()
            global_chunk_id = 0
            for doc in docs:
                source_file = filename
                page_num = doc.metadata.get("page")
                raw = doc.page_content or ""
                cleaned = clean_pdf_text(raw)
                for chunk_idx, ch in enumerate(chunk_text_fixed(cleaned, chunk_size_chars=chunk_size, overlap_chars=overlap)):
                    text = ch["text"]
                    char_start = ch["char_start"]
                    char_end = ch["char_end"]
                    chunk_id = f"{_normalize_doc_label(source_file)}-p{page_num if page_num is not None else 'n'}-{chunk_idx}"
                    all_chunks.append(
                        SimpleDocument(
                            page_content=text,
                            metadata={
                                "source_file": source_file,
                                "section": "",
                                "page": page_num,
                                "chunk_id": chunk_id,
                                "char_start": char_start,
                                "char_end": char_end,
                            },
                        )
                    )
                    global_chunk_id += 1

        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue

    print(f"Total chunks prepared: {len(all_chunks)}")
    return all_chunks


def prepare_frames_chunks(frames_dir: str):
    """
    Load Wikipedia/plain text files, clean, and fixed-size char chunk for indexing.
    """
    articles = _load_frames_articles_from_dir(frames_dir)
    if not articles:
        print("Frames directory not found or empty!")
        return []

    chunk_size = _get_int_config("chunk_size_wiki", CONFIG["chunk_size"])
    overlap = _get_int_config("chunk_overlap_wiki", CONFIG["overlap"])
    chunks: List[SimpleDocument] = []
    for art in tqdm(articles, desc="Wiki articles", unit="article"):
        clean_text = clean_wiki_text(art.get("text", ""))
        source = art.get("title") or art.get("source_file")
        section = art.get("section", "")
        for idx, chunk in enumerate(chunk_text_fixed(clean_text, chunk_size_chars=chunk_size, overlap_chars=overlap)):
            chunk_id = f"{_normalize_doc_label(source)}-{idx}"
            chunks.append(
                SimpleDocument(
                    page_content=chunk["text"],
                    metadata={
                        "source_file": source,
                        "section": section,
                        "page": "",
                        "chunk_id": chunk_id,
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    },
                )
            )

    print(f"Prepared {len(chunks)} frames text chunks from {len(articles)} articles.")
    return chunks


def fetch_wikipedia_articles(links: List[str], titles: List[str]) -> Dict[str, str]:
    """
    Read ONLY from wiki_cache_dir/index.json mapping (link/title -> filename).
    No API calls or cache writes.
    Returns dict keyed by title with raw text values.
    """
    cache_index = _load_wiki_cache_index(CONFIG.get("wiki_index_filename", "index.json"))
    articles: Dict[str, str] = {}
    missing = 0

    # Primary: use links (index keyed by link/title)
    for link in links:
        key = _normalize_title(link)
        path = cache_index.get(key)
        if not path or not os.path.exists(path):
            missing += 1
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            title = _url_to_title(link) or os.path.splitext(os.path.basename(path))[0]
            articles[title] = text
        except Exception:
            missing += 1
            continue

    # Fallback: titles
    for title in titles:
        key = _normalize_title(title)
        path = cache_index.get(key)
        if not path or not os.path.exists(path):
            continue
        if title in articles:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            articles[title] = text
        except Exception:
            continue

    print(f"Loaded {len(articles)} articles from cache (wiki_cache_dir). Missing {missing} links.")
    return articles


def build_documents(articles: Dict[str, str]) -> List["SimpleDocument"]:
    chunk_size = _get_int_config("chunk_size_wiki", CONFIG["chunk_size"])
    overlap = _get_int_config("chunk_overlap_wiki", CONFIG["overlap"])
    documents: List[SimpleDocument] = []
    for title, text in articles.items():
        clean_text = clean_wiki_text(text)
        for idx, chunk in enumerate(chunk_text_fixed(clean_text, chunk_size_chars=chunk_size, overlap_chars=overlap)):
            chunk_id = f"{_normalize_doc_label(title)}-{idx}"
            documents.append(
                SimpleDocument(
                    page_content=chunk["text"],
                    metadata={
                        "source_file": title,
                        "section": "",
                        "page": "",
                        "chunk_id": chunk_id,
                        "char_start": chunk["char_start"],
                        "char_end": chunk["char_end"],
                    },
                )
            )
    print(f"Prepared {len(documents)} chunks from {len(articles)} cached wiki articles.")
    return documents


# ==========================================
# FRAMES HELPERS
# ==========================================
def load_frames_dataset():
    print("Loading FRAMES benchmark...")
    try:
        return load_dataset(CONFIG["frames_dataset"], split=CONFIG["frames_split"])
    except Exception as exc:
        print(f"Failed to load FRAMES dataset: {exc}")
        return []


def build_frames_questions(ds, limit=None) -> List[Dict]:
    sampled = ds
    if limit and limit > 0 and len(ds) > limit:
        sampled = ds.select(range(limit))

    questions = []
    for row in sampled:
        prompt = row.get("Prompt", "")
        answer = row.get("Answer", "")
        links = _extract_links(row)
        titles = [_url_to_title(link) for link in links]
        titles = [t for t in titles if t]

        if not prompt or not links or not titles:
            continue

        questions.append(
            {
                "query": prompt,
                "answer": answer,
                "target_links": links,
                "target_titles": titles,
            }
        )
    return questions


def run_frames_experiment(questions: List[Dict], retriever, generator_client):
    if not questions:
        return []

    questions_eval = questions
    print(f"Evaluating {len(questions_eval)} FRAMES questions.")

    results = []
    for row in tqdm(questions_eval, desc="FRAMES questions", unit="q"):
        query = row["query"]
        target_titles = row.get("target_titles", [])
        docs, debug_info = retriever.invoke(query)

        contexts = [d.page_content for d in docs]
        found_titles = [d.metadata.get("source_file", "") for d in docs]

        ans = generate_answer(generator_client, "\n\n".join(contexts), query)

        precision, recall, f1 = _compute_retriever_metrics(target_titles, found_titles)

        results.append(
            {
                "question": query,
                "answer": ans,
                "contexts": contexts,
                "ground_truth": row.get("answer", ""),
                "target_titles": target_titles,
                "found_files": found_titles,
                "retriever_precision": precision,
                "retriever_recall": recall,
                "retriever_f1": f1,
                "debug": debug_info,
            }
        )

    return results



@dataclass
class RetrievedDoc:
    page_content: str
    metadata: Dict


@dataclass
class SimpleDocument:
    page_content: str
    metadata: Dict


class WhooshBM25Retriever:
    def __init__(self, index_dir: str, k: int):
        self.index = open_dir(index_dir)
        self.k = k
        self.parser = SimpleParser("content", schema=self.index.schema)

        self.enable_query_expansion = CONFIG.get("bm25_enable_query_expansion", 0)
        self.max_query_expansions = CONFIG.get("bm25_max_query_expansions", 10)

    def invoke(self, query: str, allowed_sources: set[str] | None = None) -> List[RetrievedDoc]:
        if not query:
            return [], {}
        print(f"[RETRIEVE][BM25] query='{query}' top_k={self.k}")

        debug = {}
        bm25_query = query
        if self.enable_query_expansion:
            bm25_query, qe_debug = expand_query_for_bm25(
                user_query=query,
                abbrev_map_path=CONFIG.get("abbrev_map_path"),
                max_expansions=self.max_query_expansions
            )
            debug["query_expansions"] = qe_debug

        normalized_sources = None
        if allowed_sources:
            normalized_sources = {
                _normalize_doc_label(src)
                for src in allowed_sources
                if src and str(src).strip()
            }
            if not normalized_sources:
                normalized_sources = None
        
        try:
            parsed = self.parser.parse(bm25_query)
        except Exception:
            parsed = self.parser.parse(re.sub(r"[^\w\s]", " ", str(bm25_query)))
        
        source_filter = None
        if normalized_sources:
            terms = [Term("source_norm", tok) for tok in normalized_sources]
            if terms:
                source_filter = Or(terms) if len(terms) > 1 else terms[0]
        with self.index.searcher() as searcher:
            hits = searcher.search(parsed, limit=self.k, filter=source_filter)
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

            debug["bm25_query_used"] = bm25_query
            debug["num_hits"] = len(results)

            return results, debug


def reciprocal_rank_fusion(result_sets: List[List[RetrievedDoc]], rrf_top_k: int):
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    """
    fused = {}
    rrf_constant = 60  # RRF parameter
    for result_set in result_sets:
        for rank, doc in enumerate(result_set):
            if rank < 0:
                continue
            doc_id = doc.metadata.get("doc_id")
            # doc_id = doc.metadata.get("source_file")
            if doc_id is None:
                continue
            key = str(doc_id)
            stored = fused.get(key, {"doc": doc, "score": 0.0})
            stored["score"] += 1.0 / (rrf_constant + rank + 1)
            fused[key] = stored
    ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    
    if rrf_top_k:
        ordered = ordered[:rrf_top_k]

    combined = []
    for item in ordered:
        doc = item["doc"]
        doc.metadata["rrf_score"] = item["score"]
        combined.append(doc)
        
    return combined


class DenseFAISSRetriever:
    def __init__(self, index, metadata_map: Dict[str, Dict], embedder, k: int):
        self.index = index
        self.metadata_map = metadata_map
        self.embedder = embedder
        self.k = k

        self.enable_query_expansion = CONFIG.get("dense_enable_query_expansion", 0)

    def invoke(self, query: str, allowed_sources: set[str] | None = None) -> List[RetrievedDoc]:
        if not query:
            return []

        print(f"[RETRIEVE][DENSE] query='{query[:20]}' top_k={self.k}")

        debug = {}
        dense_query = query
        if self.enable_query_expansion:
            dense_query, qe_debug = expand_query_for_dense(
                user_query=query,
                abbrev_map_path=CONFIG.get("abbrev_map_path"),
            )
            debug["query_expansions"] = qe_debug

        # Normalize allowed sources once
        allowed_tokens = None
        if allowed_sources:
            allowed_tokens = {
                _normalize_doc_label(src)
                for src in allowed_sources
                if src and str(src).strip()
            }
            if not allowed_tokens:
                allowed_tokens = None

        # Embed query
        query_vec = self.embedder.encode(
            [dense_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_vec = np.asarray(query_vec, dtype="float32")

        # FAISS search
        scores, indices = self.index.search(query_vec, self.k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            doc_id = str(idx)
            meta = self.metadata_map.get(doc_id)
            if not meta:
                continue

            # --- SOURCE FILTER (CRITICAL FIX) ---
            if allowed_tokens:
                if meta.get("source_norm", "") not in allowed_tokens:
                    continue

            results.append(
                RetrievedDoc(
                    page_content=meta.get("content", ""),
                    metadata={
                        "doc_id": doc_id,
                        "source_file": meta.get("source_file", ""),
                        "section": meta.get("section", ""),
                        "page": meta.get("page", ""),
                        "char_start": meta.get("char_start"),
                        "char_end": meta.get("char_end"),
                        "rank": rank,
                        "score": float(scores[0][rank]),
                    },
                )
            )

        print(f"[RETRIEVE][DENSE] returned={len(results)}")
        debug["dense_query_used"] = dense_query
        debug["num_hits"] = len(results)
        return results, debug


class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, rrf_top_k: int, top_k: int):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_top_k = rrf_top_k
        self.top_k = top_k
        self.enable_reranker = _get_bool_config("enable_reranker", True)

    def invoke(self, query: str, allowed_sources: set[str] | None = None) -> List[RetrievedDoc]:
        print(f"[RETRIEVE][HYBRID] query='{query[:20]}' hybrid_top_k={self.top_k} rrf_top_k={self.rrf_top_k}")
        result_sets = []
        debug_info = {}
        if self.bm25_retriever:
            result, debug_ = self.bm25_retriever.invoke(query, allowed_sources=allowed_sources)
            debug_info["bm25_debug"] = debug_
            result_sets.append(result)
        if self.dense_retriever:
            result, debug_ = self.dense_retriever.invoke(query, allowed_sources=allowed_sources)
            debug_info["dense_debug"] = debug_
            result_sets.append(result)
        if not result_sets:
            return [], debug_info
        if len(result_sets) == 1:
            fused = result_sets[0][: self.top_k]
        else:
            fused = reciprocal_rank_fusion(result_sets, rrf_top_k=self.rrf_top_k)
        print(f"[RETRIEVE][HYBRID] fused_returned={len(fused)}")
        if not self.enable_reranker:
            print("[RERANK] Disabled via ENABLE_RERANKER; returning fused results.")
            return fused[: self.top_k], debug_info
        reranked = rerank_with_cross_encoder(
            query=query,
            retrieved_docs=fused,
            top_k=self.top_k,
            batch_size=CONFIG["rerank_batch_size"],
        )
        return reranked, debug_info
    

def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        model_name = _resolve_cross_encoder_model()
        _cross_encoder_model = CrossEncoder(model_name)
    return _cross_encoder_model


def _resolve_cross_encoder_model():
    """
    Resolve cross-encoder model path/name with preference for local copy.
    """
    configured = CONFIG.get("cross_encoder_model")
    candidates = [configured, LOCAL_CROSS_ENCODER_DIR, CROSS_ENCODER_DEFAULT]
    chosen = None
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            chosen = candidate
            break
    chosen = chosen or configured or CROSS_ENCODER_DEFAULT
    logging.info(f"[RERANK][CROSS_ENCODER] Using model: {chosen}")
    return chosen


def rerank_with_cross_encoder(
    query: str,
    retrieved_docs: List[RetrievedDoc],
    batch_size: int = 5,
    top_k: int = 10,
) -> List[RetrievedDoc]:
    """
    Rerank retrieved documents with a cross-encoder for better local relevance scoring.
    """
    if not retrieved_docs:
        return []

    model = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = model.predict(pairs, batch_size=batch_size)

    for doc, score in zip(retrieved_docs, scores):
        doc.metadata["rerank_score"] = float(score)

    reranked = sorted(retrieved_docs, key=lambda d: d.metadata["rerank_score"], reverse=True)

    seen = set()
    final_docs = []
    for doc in reranked:
        key = (doc.metadata.get("source_file", ""), doc.page_content[:100])
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if len(final_docs) >= top_k:
            break

    return final_docs


def build_whoosh_index(chunks, index_dir: str, k: int, limit_mb: int | None = None):
    if not chunks:
        return None
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    print(f"[WHOOSH] (re)building index at {index_dir} with {len(chunks)} chunks")
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        source_file=ID(stored=True),
        source_norm=ID(stored=True),
        content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    )
    index = create_in(index_dir, schema)
    writer = index.writer(limitmb=limit_mb or 1024)
    for idx, chunk in enumerate(chunks):
        src = str(chunk.metadata.get("source_file", ""))
        doc_id = str(chunk.metadata.get("chunk_id", idx))
        writer.add_document(
            doc_id=doc_id,
            source_file=src,
            source_norm=_normalize_doc_label(src),
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
        src = str(chunk.metadata.get("source_file", ""))
        doc_id = str(chunk.metadata.get("chunk_id", idx))
        metadata_payload.append({
            "doc_id": doc_id,
            "source_file": src,
            "source_norm": _normalize_doc_label(src),
            "section": chunk.metadata.get("section", ""),
            "page": chunk.metadata.get("page", ""),
            "char_start": chunk.metadata.get("char_start"),
            "char_end": chunk.metadata.get("char_end"),
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


def load_and_build_retriever(_generator_client=None):
    mode = str(CONFIG.get("doc_mode", "pdf")).lower()
    if mode == "frames":
        raise RuntimeError("Frames retriever is built inside run_frames_workflow.")

    retriever = _load_and_build_pdf_retriever(CONFIG.get("pdf_dir"))

    if retriever is None:
        raise RuntimeError("Failed to initialize any retriever. Check logs for details.")
    
    if int(CONFIG.get("enable_hopping", 0)):
        base_invoke = retriever.invoke
        model_name = CONFIG["azure_gen_deployment"] if USE_AZURE else CONFIG["ollama_model"]
        retriever.invoke = make_hopping_invoke(
            base_invoke,
            generator_client=_generator_client or models["generator_client"],
            model_name=model_name,
            max_hops=int(CONFIG.get("hop_max_hops", 2)),
            evidence_max_docs=int(CONFIG.get("hop_evidence_docs", 8)),
            evidence_max_chars=int(CONFIG.get("hop_evidence_chars", 1200)),
            query_max_tokens=int(CONFIG.get("hop_query_max_tokens", 40)),
        )

    return retriever


def _load_and_build_pdf_retriever(pdf_dir):
    index_dir = CONFIG.get("whoosh_index_dir", "whoosh_pdf_index")
    rebuild_bm25 = CONFIG.get("rebuild_index", True)
    rebuild_dense = CONFIG.get("rebuild_dense_index", False)
    faiss_index_path = CONFIG.get("faiss_index_path", "faiss_index.bin")
    faiss_metadata_path = CONFIG.get("faiss_metadata_path", "faiss_metadata.json")

    bm25_top_k = _get_int_config("top_k", 5)
    dense_top_k = _get_int_config("dense_top_k", bm25_top_k)
    hybrid_top_k = _get_int_config("hybrid_top_k", bm25_top_k)
    rrf_top_k = _get_int_config("rrf_top_k", 60)
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
        if not rebuild_bm25 and not bm25_ready:
            print(f"[WHOOSH] building missing PDF index at {index_dir}")
        bm25_retriever = build_whoosh_index(chunks, index_dir, bm25_top_k, CONFIG.get("whoosh_limit_mb"))
    else:
        print(f"[WHOOSH] reusing PDF index at {index_dir}")
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
                model=CONFIG.get("ollama_embed_model"),
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
    return HybridRetriever(bm25_retriever, dense_retriever, rrf_top_k, hybrid_top_k)


def _load_and_build_frames_retriever(frames_dir: str, documents: List[SimpleDocument] | None = None):
    index_dir = CONFIG.get("whoosh_frames_index_dir", "whoosh_wiki_index")
    rebuild_bm25 = CONFIG.get("rebuild_frames_index", True)
    rebuild_dense = CONFIG.get("rebuild_dense_frames_index", False)
    faiss_index_path = CONFIG.get("faiss_frames_index_path", "faiss_wiki_index.bin")
    faiss_metadata_path = CONFIG.get("faiss_frames_metadata_path", "faiss_wiki_metadata.json")

    bm25_top_k = _get_int_config("top_k", 5)
    dense_top_k = _get_int_config("dense_top_k", bm25_top_k)
    hybrid_top_k = _get_int_config("hybrid_top_k", bm25_top_k)
    rrf_top_k = _get_int_config("rrf_top_k", 60)
    embed_tpm_limit = _get_int_config("embed_tpm_limit", 120000)
    embed_rpm_limit = _get_int_config("embed_rpm_limit", 60)
    embed_batch_size = _get_int_config("embed_batch_size", 16)

    bm25_ready = exists_in(index_dir)
    dense_ready = os.path.exists(faiss_index_path) and os.path.exists(faiss_metadata_path)
    need_chunks = rebuild_bm25 or rebuild_dense or not bm25_ready or not dense_ready

    chunks = documents if (documents is not None and need_chunks) else (prepare_frames_chunks(frames_dir) if need_chunks else [])
    if need_chunks and not chunks:
        return None

    bm25_retriever = None
    if rebuild_bm25 or not bm25_ready:
        if not rebuild_bm25 and not bm25_ready:
            print(f"[WHOOSH] building missing FRAMES index at {index_dir}")
        bm25_retriever = build_whoosh_index(chunks, index_dir, bm25_top_k, CONFIG.get("whoosh_limit_mb"))
    else:
        print(f"[WHOOSH] reusing FRAMES index at {index_dir}")
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
                model=CONFIG.get("ollama_embed_model"),
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
                chunks = prepare_frames_chunks(frames_dir)
            dense_retriever = build_dense_retriever(chunks, embedder, faiss_index_path, faiss_metadata_path, dense_top_k)
        else:
            dense_retriever = load_dense_retriever(embedder, faiss_index_path, faiss_metadata_path, dense_top_k)

    if not bm25_retriever and not dense_retriever:
        return None
    if not bm25_retriever:
        return dense_retriever
    return HybridRetriever(bm25_retriever, dense_retriever, rrf_top_k, hybrid_top_k)


def _normalize_doc_label(name):
    """
    Normalize document identifiers for robust comparison.
    """
    if not name:
        return ""
    value = os.path.basename(str(name).strip()).lower()
    if value.endswith(".pdf") or value.endswith(".txt"):
        value = value.rsplit(".", 1)[0]
    return value


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
    qrange = _parse_question_range(CONFIG.get("question_range"))
    if qrange:
        start, end = qrange
        df = df.iloc[start - 1 : end]
    print(f"Loaded {len(df)} questions.")
    results = []
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="PDF questions", unit="q"):
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

        # Retrieve
        docs, debug_info = retriever.invoke(q)
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
            "debug": debug_info
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


def run_pdf_workflow(retriever, models):
    cache_path = CONFIG.get("llm_cache_path")
    experiment_data = None
    if CONFIG.get("reuse_llm_outputs"):
        experiment_data = load_cached_experiment(cache_path)

    if experiment_data is None:
        log_step("Run experiment")
        experiment_data = run_experiment(CONFIG.get("excel_path"), retriever, models["generator_client"])
        if not experiment_data:
            return []
        if CONFIG.get("reuse_llm_outputs"):
            save_experiment_cache(experiment_data, cache_path)
    return experiment_data or []


def run_frames_workflow(models):
    ds = load_frames_dataset()
    if not ds:
        print("FRAMES dataset unavailable; exiting.")
        return []
    questions_all = build_frames_questions(ds, limit=None)
    if not questions_all:
        print("No FRAMES questions prepared; exiting.")
        return []

    qrange = _parse_question_range(CONFIG.get("question_range"))
    questions_eval = _apply_question_range(questions_all, qrange)

    index_dir = CONFIG.get("whoosh_frames_index_dir", "whoosh_wiki_index")
    # Collect unique links + titles
    all_links: List[str] = []
    all_titles: List[str] = []
    for row in questions_all:
        all_links.extend(row.get("target_links", []))
        all_titles.extend(row.get("target_titles", []))

    unique_links = list({x for x in all_links if x})
    unique_titles = list({t for t in all_titles if t})

    cap = CONFIG.get("frames_max_wiki_titles")
    if cap and cap > 0:
        unique_links = unique_links[:cap]
        unique_titles = unique_titles[:cap]

    rebuild_index = CONFIG.get("rebuild_frames_index", True)
    rebuild_dense = CONFIG.get("rebuild_dense_frames_index", False)
    index_exists = exists_in(index_dir)
    faiss_index_path = CONFIG.get("faiss_frames_index_path", "faiss_wiki_index.bin")
    faiss_metadata_path = CONFIG.get("faiss_frames_metadata_path", "faiss_wiki_metadata.json")
    dense_ready = os.path.exists(faiss_index_path) and os.path.exists(faiss_metadata_path)

    need_rebuild = rebuild_index or rebuild_dense or not index_exists or not dense_ready
    documents: List[SimpleDocument] | None = None
    if need_rebuild:
        articles = fetch_wikipedia_articles(unique_links, unique_titles)
        if not articles:
            print("No cached wiki articles found; cannot build FRAMES retriever.")
            return []
        documents = build_documents(articles)

    retriever = _load_and_build_frames_retriever(
        CONFIG.get("frames_dir") or CONFIG.get("wiki_cache_dir", DEFAULT_WIKI_CACHE_DIR),
        documents=documents,
    )
    if not retriever:
        print("Failed to initialize FRAMES retriever.")
        return []

    # Hopping wrapper for frames as well
    if int(CONFIG.get("enable_hopping", 0)):
        base_invoke = retriever.invoke
        model_name = CONFIG["azure_gen_deployment"] if USE_AZURE else CONFIG["ollama_model"]
        retriever.invoke = make_hopping_invoke(
            base_invoke,
            generator_client=models["generator_client"],
            model_name=model_name,
            max_hops=int(CONFIG.get("hop_max_hops", 2)),
            evidence_max_docs=int(CONFIG.get("hop_evidence_docs", 8)),
            evidence_max_chars=int(CONFIG.get("hop_evidence_chars", 1200)),
            query_max_tokens=int(CONFIG.get("hop_query_max_tokens", 40)),
        )

    cache_path = CONFIG.get("llm_cache_path")
    experiment_data = None
    if CONFIG.get("reuse_llm_outputs"):
        experiment_data = load_cached_experiment(cache_path)
    if experiment_data is None:
        experiment_data = run_frames_experiment(questions_eval, retriever, models["generator_client"])
        if experiment_data and CONFIG.get("reuse_llm_outputs"):
            save_experiment_cache(experiment_data, cache_path)
    return experiment_data or []

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Init
    log_step("Initialize models and retrievers")
    models = get_models()
    mode = str(CONFIG.get("doc_mode", "pdf")).lower()
    print(f"Document mode: {mode.upper()}")
    retriever = None
    if mode != "frames":
        retriever = load_and_build_retriever()
        if not retriever: exit()

    # 2. Run per mode
    log_step("Load cache or run experiment")
    if mode == "frames":
        experiment_data = run_frames_workflow(models)
    else:
        experiment_data = run_pdf_workflow(retriever, models)
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
        ),
        # ExactMatch(),
        # _ChrfScore()
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
    debug = []
    for record in experiment_data:
        prec, rec, f1 = _ensure_retriever_metrics(record)
        retriever_precisions.append(prec)
        retriever_recalls.append(rec)
        retriever_f1s.append(f1)
        debug.append(record.get("debug", {}))
    df_out['retriever_precision'] = retriever_precisions
    df_out['retriever_recall'] = retriever_recalls
    df_out['retriever_f1'] = retriever_f1s
    df_out['target_pdf'] = [x.get('target_pdf') or x.get('target_titles') for x in experiment_data]
    df_out['found_files'] = [str(x.get('found_files')) for x in experiment_data]
    df_out['debug'] = debug

    mode_tag = mode.lower()
    azure_tag = "azure" if USE_AZURE else "local"
    qrange = _parse_question_range(CONFIG.get("question_range"))
    qrange_tag = f"q{qrange[0]}-{qrange[1]}" if qrange else "qall"
    base_name = f"Results_Hybrid_{mode_tag}_{azure_tag}_{qrange_tag}"
    fname_csv = f"{base_name}.csv"
    fname_xlsx = f"{base_name}.xlsx"

    while True:
        try:
            df_out.to_csv(fname_csv, index=False)
            break
        except PermissionError:
            input(f"Please close {fname_csv} and press Enter to retry...")

    while True:
        try:
            df_out.to_excel(fname_xlsx, index=False)
            break
        except PermissionError:
            input(f"Please close {fname_xlsx} and press Enter to retry...")

    print(f"\nSaved to {fname_csv} and {fname_xlsx}")
    print(f"Avg Retriever F1: {df_out['retriever_f1'].mean():.2f}")
    print(scores)
