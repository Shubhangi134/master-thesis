"""
Hybrid BM25 + dense RAG pipeline for FRAMES + Wikipedia (script-relative paths):
1. Load FRAMES benchmark questions.
2. Resolve wiki links/titles; if use_wiki_cache=1 read index.json link->file only (no writes), else fetch via Wikipedia API.
3. Clean and chunk articles, then build Whoosh (BM25) + FAISS (dense) indexes.
4. Run hybrid retrieval (RRF + optional reranker), generate answers, and Ragas AnswerCorrectness.
5. Save results to CSV + XLSX; indexes saved beside this script; Wikipedia cache stays read-only.

Cache/config notes:
- use_wiki_cache: "1"/"true" reads only from wiki_cache_dir; "0"/"false" fetches fresh (no cache read/write).
- wiki_cache_dir defaults to a sibling wiki_data folder; Whoosh/FAISS paths are also anchored to SCRIPT_DIR.
"""
import os
import json
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple
from urllib.parse import unquote
import hashlib
import logging
import uuid

import ftfy
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from dotenv import load_dotenv
import faiss
import tiktoken
from tqdm import tqdm
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import SimpleParser

# OpenAI / Azure clients
from openai import AzureOpenAI
from openai import OpenAI as StandardOpenAI

from sentence_transformers import CrossEncoder

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)

USE_AZURE = bool(os.getenv("ENDPOINT"))

# --- Path anchoring (script-relative defaults) ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

DEFAULT_WIKI_CACHE_DIR = os.path.join(SCRIPT_DIR, "wiki_data")
DEFAULT_WHOOSH_INDEX_DIR = os.path.join(SCRIPT_DIR, "whoosh_wiki_index")
DEFAULT_FAISS_INDEX_PATH = os.path.join(SCRIPT_DIR, "faiss_wiki_index.bin")
DEFAULT_FAISS_METADATA_PATH = os.path.join(SCRIPT_DIR, "faiss_wiki_metadata.json")

CONFIG = {
    # Dataset + sampling
    "frames_dataset": "google/frames-benchmark",
    "frames_split": "test",
    "seed": 42,
    "max_questions": int(os.getenv("FRAMES_MAX_QUESTIONS", 5)),
    "max_wiki_titles": int(os.getenv("MAX_WIKI_TITLES", 10)),  # cap titles/links to fetch; 0 = no cap

    # Wikipedia API / Cache
    "wiki_language": os.getenv("WIKI_LANG", "en"),
    "wiki_batch_size": int(os.getenv("WIKI_BATCH_SIZE", 20)),
    "wiki_sleep": float(os.getenv("WIKI_SLEEP", 0.05)),
    "wiki_cache_dir": os.getenv("WIKI_CACHE_DIR", DEFAULT_WIKI_CACHE_DIR),
    "use_wiki_cache": os.getenv("USE_WIKI_CACHE", "1"),  # NEW

    # Chunking
    "chunk_size": int(os.getenv("CHUNK_SIZE", 300)),
    "overlap": int(os.getenv("CHUNK_OVERLAP", 50)),

    # Retrieval
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 40)),
    "dense_top_k": int(os.getenv("DENSE_TOP_K", 40)),
    "hybrid_top_k": int(os.getenv("HYBRID_TOP_K", 4)),
    "rrf_top_k": int(os.getenv("RRF_TOP_K", 60)),
    "rebuild_whoosh": int(os.getenv("REBUILD_INDEX", 1)) > 0,
    "rebuild_dense": int(os.getenv("REBUILD_DENSE_INDEX", 1)) > 0,
    "whoosh_limit_mb": int(os.getenv("WHOOSH_LIMIT_MB", 1024)),
    "whoosh_index_dir": os.getenv("WHOOSH_INDEX_DIR", DEFAULT_WHOOSH_INDEX_DIR),
    "faiss_index_path": os.getenv("FAISS_INDEX_PATH", DEFAULT_FAISS_INDEX_PATH),
    "faiss_metadata_path": os.getenv("FAISS_METADATA_PATH", DEFAULT_FAISS_METADATA_PATH),

    # Embeddings / generation
    "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", 32)),
    "embed_tpm_limit": int(os.getenv("EMBED_TPM_LIMIT", 1_50_000)),
    "embed_rpm_limit": int(os.getenv("EMBED_RPM_LIMIT", 60)),
    "embed_max_tokens": int(os.getenv("EMBED_MAX_TOKENS", 1024)),
    "azure_api_key": os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
    "azure_endpoint": os.getenv("ENDPOINT"),
    "azure_api_version": os.getenv("API_VERSION"),
    "azure_gen_deployment": os.getenv("MODEL_NAME"),
    "azure_embed_deployment": "text-embedding-ada-002",
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "ollama_api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
    "ollama_model": os.getenv("OLLAMA_MODEL", "mistral-large-3:675b-cloud"),
    "ollama_embed_model": os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest"),
    "local_embed_model": os.getenv("LOCAL_EMBED_MODEL", "mxbai-embed-large:latest"),

    # Context building
    "use_full_article_context": os.getenv("USE_FULL_ARTICLE_CONTEXT", "1"),
    "context_dump_dir": os.getenv("CONTEXT_DUMP_DIR", os.path.join(SCRIPT_DIR, "retriever_contexts")),
    "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", 75_000)),  # 0 = no truncation; applies to combined contexts
    "resume_last": os.getenv("RESUME_LAST", "1"),  # reuse existing results and skip processed queries

    # Reranker
    "enable_reranker": os.getenv("ENABLE_RERANKER", "1"),
    "cross_encoder_model": os.getenv("CROSS_ENCODER_MODEL_NAME"),
    "rerank_batch_size": int(os.getenv("RERANK_BATCH_SIZE", 5)),

    # Evaluation output
    "results_xlsx": os.getenv("RESULTS_XLSX", os.path.join(SCRIPT_DIR, "Frames_Wiki_RAG_hybrid.xlsx")),
}


# --- Helpers ---
def _get_int_config(name: str, default: int) -> int:
    try:
        value = CONFIG.get(name, default)
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _get_bool_config(name: str, default: bool = False) -> bool:
    value = CONFIG.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value) if value is not None else bool(default)


def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _add_suffix_to_path(path: str, suffix: str) -> str:
    root, ext = os.path.splitext(path)
    marker = f"_{suffix}"
    if root.endswith(marker):
        return path
    return f"{root}{marker}{ext}"


def _get_results_paths():
    xlsx_path = CONFIG["results_xlsx"]
    if _get_bool_config("enable_reranker", True):
        xlsx_path = _add_suffix_to_path(xlsx_path, "reranker")
    return xlsx_path


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return ftfy.fix_text(text.strip())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    tokens = _tokenize(text)
    if not tokens:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + chunk_size]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
    return chunks


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


def _url_to_title(url: str) -> str:
    if not url:
        return ""
    marker = "/wiki/"
    if marker in url:
        url = url.split(marker, 1)[-1]
    url = url.split("#", 1)[0].split("?", 1)[0]
    url = unquote(url)
    url = url.replace("_", " ")
    return url.strip()


def _normalize_title(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compute_retriever_metrics(target_titles: List[str], retrieved_titles: List[str]):
    target_set = {_normalize_title(x) for x in target_titles if x}
    retrieved_set = {_normalize_title(x) for x in retrieved_titles if x}
    target_set.discard("")
    retrieved_set.discard("")
    if not target_set or not retrieved_set:
        return 0.0, 0.0, 0.0
    tp = len(target_set & retrieved_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    precision = tp / len(retrieved_set)
    recall = tp / len(target_set)
    f1 = 0.0 if (precision == 0 or recall == 0) else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _batch_iterable(items: Iterable, batch_size: int):
    if batch_size is None or batch_size <= 0:
        yield list(items)
        return
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --- Cache helpers (READ-ONLY, index.json link -> filename mapping) ---
def _load_wiki_cache_index() -> Dict[str, str]:
    """
    Loads wiki_cache_dir/index.json mapping of (link -> filename).
    Returns dict: normalized_key -> absolute_file_path.

    Supports:
      - {"<link>": "<file>", ...}
      - [{"link": "<link>", "filename": "<file>"}, ...]
    Also stores a derived title key for each link: _url_to_title(link)
    """
    cache_dir = CONFIG.get("wiki_cache_dir", DEFAULT_WIKI_CACHE_DIR)
    index_path = os.path.join(cache_dir, "index.json")
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

        # Key by link
        norm_to_path[_normalize_title(link)] = abs_path

        # Also key by derived title
        derived = _url_to_title(link)
        if derived:
            norm_to_path[_normalize_title(derived)] = abs_path

    return norm_to_path


def _read_cached_wiki_by_key(key: str, cache_index: Dict[str, str]) -> Optional[str]:
    """
    Reads cached article text using a key that may be a link OR a title.
    Returns None if missing/unreadable.
    """
    if not key:
        return None
    path = cache_index.get(_normalize_title(key))
    if not path:
        return None
    if not os.path.exists(path):
        logging.warning("Cache index points to missing file: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        logging.warning("Failed reading cached file %s: %s", path, exc)
        return None


# --- Wikipedia ingestion (UPDATED) ---
def load_frames_dataset():
    print("Loading FRAMES benchmark...")
    return load_dataset(CONFIG["frames_dataset"], split=CONFIG["frames_split"])


def build_questions(ds, limit: Optional[int]) -> List[Dict]:
    sampled = ds
    if limit and limit > 0 and len(ds) > limit:
        # Keep FRAMES in original order; just take the first N
        sampled = ds.select(range(limit))
    questions = []
    for row in sampled:
        prompt = row.get("Prompt", "")
        answer = row.get("Answer", "")
        links = _extract_links(row)                 # NEW: keep links for cache lookups
        titles = [_url_to_title(link) for link in links]
        titles = [t for t in titles if t]
        if not prompt or not links or not titles:
            continue
        questions.append(
            {
                "query": prompt,
                "answer": answer,
                "target_links": links,              # NEW
                "target_titles": titles,
            }
        )
    print(f"Prepared {len(questions)} questions (limit={limit or 'all'}).")
    return questions


def fetch_wikipedia_articles(links: List[str], titles: List[str]) -> Dict[str, str]:
    """
    If use_wiki_cache=True:
      - Read ONLY from wiki_cache_dir/index.json link->filename mapping
      - DO NOT call Wikipedia API
      - DO NOT write cache

    If use_wiki_cache=False:
      - Fetch from Wikipedia API using titles
      - DO NOT read cache
      - DO NOT write cache

    Returns dict keyed by title (best-effort).
    """
    use_cache = _get_bool_config("use_wiki_cache", True)

    if use_cache:
        cache_index = _load_wiki_cache_index()
        if not cache_index:
            logging.warning("use_wiki_cache=True but cache index.json is missing/empty.")

        articles: Dict[str, str] = {}
        seen_title_norm = set()
        missing = 0

        # primary lookup by link (index.json requirement)
        for link in links:
            text = _read_cached_wiki_by_key(link, cache_index)
            if not text:
                missing += 1
                continue
            title = _url_to_title(link) or link
            norm = _normalize_title(title)
            if norm in seen_title_norm:
                continue
            seen_title_norm.add(norm)
            articles[title] = text

        if missing:
            logging.warning(
                "Cache mode enabled. %d/%d links missing from cache (no API fetch will occur).",
                missing, len(links)
            )

        print(f"Loaded {len(articles)} articles from cache (use_wiki_cache=True).")
        return articles

    # fresh mode (no cache read/write)
    if not titles:
        return {}

    # de-dup by normalized title
    deduped_titles = []
    seen_norm = set()
    for t in titles:
        norm = _normalize_title(t)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        deduped_titles.append(t)
    titles = deduped_titles

    batch_size = _get_int_config("wiki_batch_size", 20)
    sleep_for = float(CONFIG.get("wiki_sleep") or 0.0)
    lang = CONFIG.get("wiki_language", "en") or "en"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Codex-RAG/1.0 (+https://github.com/)",
        "Accept": "application/json",
    })

    articles: Dict[str, str] = {}
    fetch_batches = list(_batch_iterable(titles, batch_size))
    print(f"Fetching {len(titles)} Wikipedia articles in {len(fetch_batches)} batches (lang={lang}, use_wiki_cache=False).")

    for batch in tqdm(fetch_batches, desc="Fetching Wikipedia", unit="batch"):
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "exlimit": "max",
            "format": "json",
            "redirects": 1,
            "titles": "|".join(batch),
            "formatversion": 2
        }
        try:
            resp = session.get(api_url, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"Failed to fetch batch: {exc}")
            continue

        if resp.status_code != 200:
            print(f"Failed to fetch batch: HTTP {resp.status_code}")
            if resp.status_code in (403, 429):
                time.sleep(max(1.0, sleep_for * 2))
            continue

        try:
            payload = resp.json()
        except ValueError:
            print("Failed to parse JSON response from Wikipedia.")
            continue

        pages = payload.get("query", {}).get("pages", [])
        for page in pages:
            title = page.get("title")
            text = page.get("extract")
            if title and text:
                articles[title] = text

        if sleep_for:
            time.sleep(sleep_for)

    print(f"Fetched {len(articles)} articles from Wikipedia (no cache read/write).")
    return articles


def build_documents(articles: Dict[str, str]):
    chunk_size = _get_int_config("chunk_size", 500)
    overlap = _get_int_config("overlap", 50)
    documents = []
    seen_titles = set()
    for title, text in tqdm(articles.items(), desc="Chunking articles", unit="article"):
        norm_title = _normalize_title(title)
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        clean_text = _clean_text(text)
        chunks = _chunk_text(clean_text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            documents.append(
                {
                    "doc_id": f"{norm_title}-{idx}",
                    "title": title,
                    "content": chunk,
                }
            )
    print(f"Prepared {len(documents)} chunks.")
    return documents


# --- (rest of your code unchanged until evaluation, but evaluation must stop reading cache paths) ---

# Remove/ignore old cache naming helpers (kept only if used elsewhere)
def _cache_path_for_title(title: str) -> str:
    title = _normalize_title(title)
    cache_dir = CONFIG.get("wiki_cache_dir", DEFAULT_WIKI_CACHE_DIR)
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", title).strip("_")
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    fname = f"{base[:80]}_{digest}.txt" if base else f"article_{digest}.txt"
    return os.path.join(cache_dir, fname)


def _legacy_cache_path_for_title(title: str) -> str:
    cache_dir = CONFIG.get("wiki_cache_dir", DEFAULT_WIKI_CACHE_DIR)
    base = re.sub(r"[^\w\-_. ]+", "_", _normalize_title(title)).strip(" ._")
    fname = f"{base or 'article'}.txt"
    return os.path.join(cache_dir, fname)


def _cache_paths_for_title(title: str) -> List[str]:
    # Deprecated in new cache scheme; keep for backward compatibility if you still have old cache.
    paths = {_cache_path_for_title(title), _legacy_cache_path_for_title(title)}
    return list(paths)


def _load_cached_articles_from_dir(cache_dir: str) -> Dict[str, str]:
    """
    Deprecated for new cache scheme; do not use for index.json cache.
    Left here only so legacy caches won't crash if referenced.
    """
    articles = {}
    seen_norm = set()
    if not os.path.isdir(cache_dir):
        print(f"Cache directory not found: {cache_dir}")
        return articles
    for fname in os.listdir(cache_dir):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(cache_dir, fname)
        try:
            text = open(path, "r", encoding="utf-8").read()
        except OSError as exc:
            print(f"Failed to read cache file {path}: {exc}")
            continue
        title_raw = os.path.splitext(fname)[0]
        title_raw = re.sub(r"_[0-9a-f]{8}$", "", title_raw)
        title = title_raw.replace("_", " ").strip() or title_raw
        norm = _normalize_title(title)
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        articles[title] = text
    print(f"Loaded {len(articles)} cached articles from {cache_dir}.")
    return articles


class _TPMThrottle:
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
                print(f"[EMBED][TPM] Sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)
            self.window_start = time.time()
            self.tokens_used = 0
        self.tokens_used += token_count


class _RPMThrottle:
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
                print(f"[EMBED][RPM] Sleeping {sleep_for:.2f}s")
                time.sleep(sleep_for)
            self.window_start = time.time()
            self.calls_made = 0
        self.calls_made += 1


def _get_tokenizer(model_name: str):
    if USE_AZURE:
        try:
            return tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                return tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None
    return None


def _count_tokens(texts, tokenizer):
    total = 0
    for text in texts:
        if tokenizer:
            try:
                total += len(tokenizer.encode(text))
                continue
            except Exception:
                pass
        total += len(text.split())
    return total


def _truncate_texts(texts: List[str], tokenizer, max_tokens: int) -> List[str]:
    if max_tokens is None or max_tokens <= 0:
        return list(texts)
    truncated = []
    for text in texts:
        if tokenizer:
            try:
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    tokens = tokens[:max_tokens]
                    text = tokenizer.decode(tokens)
            except Exception:
                pass
        else:
            # Fallback: trim by words to roughly max_tokens when tokenizer is unavailable
            words = text.split()
            if len(words) > max_tokens:
                text = " ".join(words[:max_tokens])
        truncated.append(text)
    return truncated


def _truncate_contexts(contexts: List[str], max_tokens: int) -> List[str]:
    """
    Truncate the combined contexts to a total token budget by cutting each chunk evenly.
    Uses o200k_base (GPT-4.1/4o) tokenizer; falls back to word-based trimming if unavailable.
    """
    if not contexts or max_tokens is None or max_tokens <= 0:
        return contexts

    try:
        encoding = tiktoken.get_encoding("o200k_base")
    except Exception:
        encoding = None

    per_chunk = max(1, max_tokens // len(contexts))
    truncated = []
    for text in contexts:
        if encoding:
            try:
                tokens = encoding.encode(text)
                tokens = tokens[:per_chunk]
                truncated.append(encoding.decode(tokens))
                continue
            except Exception:
                pass
        words = text.split()
        truncated.append(" ".join(words[:per_chunk]))
    return truncated


def _dump_contexts_to_file(contexts: List[str]) -> Optional[str]:
    if not contexts:
        return None
    dump_dir = CONFIG.get("context_dump_dir")
    if not dump_dir:
        return None
    _ensure_dir(dump_dir)
    fname = f"context_{uuid.uuid4().hex}.txt"
    path = os.path.join(dump_dir, fname)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for i, ctx in enumerate(contexts, 1):
                f.write(f"=== Context {i} ===\n")
                f.write(str(ctx))
                f.write("\n\n")
        return fname
    except Exception as exc:
        logging.warning("Failed to write contexts to %s: %s", path, exc)
        return None


_cross_encoder_model = None
CROSS_ENCODER_DEFAULT = "cross-encoder/ms-marco-MiniLM-L6-v2"
LOCAL_CROSS_ENCODER_DIR = os.path.join(SCRIPT_DIR, "models", "reranker-ms-marco-MiniLM-L6-v2")


def _resolve_cross_encoder_model() -> str:
    configured = CONFIG.get("cross_encoder_model")
    candidates = [configured, LOCAL_CROSS_ENCODER_DIR, CROSS_ENCODER_DEFAULT]
    chosen = None
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            chosen = candidate
            break
    chosen = chosen or configured or CROSS_ENCODER_DEFAULT
    logging.info("[RERANK][CROSS_ENCODER] Using model: %s", chosen)
    return chosen


def _get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        _cross_encoder_model = CrossEncoder(_resolve_cross_encoder_model())
    return _cross_encoder_model


def log_config_overview():
    logging.info("[CONFIG] CWD: %s", os.getcwd())
    logging.info("[CONFIG] Script dir: %s", SCRIPT_DIR)
    logging.info("[CONFIG] Mode: %s", "AZURE" if USE_AZURE else "LOCAL/Ollama")
    logging.info(
        "[CONFIG] Retrieval: top_k=%s dense_top_k=%s hybrid_top_k=%s rrf_top_k=%s",
        _get_int_config("top_k", 40),
        _get_int_config("dense_top_k", 40),
        _get_int_config("hybrid_top_k", 4),
        _get_int_config("rrf_top_k", 60),
    )
    logging.info("[CONFIG] Wiki cache dir: %s", CONFIG.get("wiki_cache_dir"))
    logging.info("[CONFIG] use_wiki_cache: %s", _get_bool_config("use_wiki_cache", True))
    logging.info("[CONFIG] Whoosh index dir: %s", CONFIG.get("whoosh_index_dir"))
    logging.info("[CONFIG] FAISS index path: %s", CONFIG.get("faiss_index_path"))
    logging.info("[CONFIG] Context uses full articles: %s", _get_bool_config("use_full_article_context", True))
    logging.info("[CONFIG] Max context tokens (LLM_MAX_TOKENS): %s", _get_int_config("llm_max_tokens", 0) or "disabled")
    logging.info("[CONFIG] Resume last: %s", _get_bool_config("resume_last", False))
    logging.info(
        "[CONFIG] Reranker: enabled=%s batch_size=%s configured_model=%s local_default=%s",
        _get_bool_config("enable_reranker", True),
        _get_int_config("rerank_batch_size", 5),
        CONFIG.get("cross_encoder_model") or "(not set)",
        LOCAL_CROSS_ENCODER_DIR,
    )


log_config_overview()


class AzureEmbedder:
    def __init__(self, deployment, endpoint, api_key, api_version, tpm_limit, rpm_limit, batch_size):
        self.deployment = deployment
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
            timeout=180.0,
        )
        self.tokenizer = _get_tokenizer(deployment)
        self.tpm = _TPMThrottle(tpm_limit)
        self.rpm = _RPMThrottle(rpm_limit)
        self.batch_size = max(int(batch_size), 1) if batch_size else None

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        all_vectors = []
        for batch in _batch_iterable(texts, self.batch_size):
            truncated_batch = _truncate_texts(batch, self.tokenizer, CONFIG.get("embed_max_tokens"))
            token_count = _count_tokens(truncated_batch, self.tokenizer)
            if token_count:
                print(f"[EMBED][AZURE] tokens={token_count} batch={len(truncated_batch)}")
            self.tpm.enforce(token_count)
            self.rpm.enforce()
            resp = self.client.embeddings.create(model=self.deployment, input=truncated_batch)
            all_vectors.extend(item.embedding for item in resp.data)
        arr = np.array(all_vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


class LocalOllamaEmbedder:
    def __init__(self, model, base_url, api_key, tpm_limit, rpm_limit, batch_size):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tokenizer = _get_tokenizer(model)
        self.tpm = _TPMThrottle(tpm_limit)
        self.rpm = _RPMThrottle(rpm_limit)
        self.batch_size = max(int(batch_size), 1) if batch_size else None

    def _embed_via_http(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": texts}
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        all_vectors = []
        for batch in _batch_iterable(texts, self.batch_size):
            truncated_batch = _truncate_texts(batch, self.tokenizer, CONFIG.get("embed_max_tokens"))
            token_count = _count_tokens(truncated_batch, self.tokenizer)
            if token_count:
                print(f"[EMBED][OLLAMA] tokens={token_count} batch={len(truncated_batch)}")
            self.tpm.enforce(token_count)
            self.rpm.enforce()
            try:
                vectors = self._embed_via_http(truncated_batch)
            except requests.HTTPError as exc:
                resp = getattr(exc, "response", None)
                msg_lower = ""
                if resp is not None:
                    try:
                        payload = resp.json()
                        msg_lower = str(payload.get("error", {}).get("message", "")).lower()
                    except Exception:
                        msg_lower = (resp.text or "").lower()
                if "context length" in msg_lower:
                    short_max = min(int(CONFIG.get("embed_max_tokens") or 512), 512)
                    shorter = _truncate_texts(truncated_batch, self.tokenizer, short_max)
                    print(f"[EMBED][OLLAMA] Retrying with max_tokens={short_max} due to context limit.")
                    vectors = self._embed_via_http(shorter)
                else:
                    raise
            all_vectors.extend(vectors)
        arr = np.array(all_vectors, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms


def _get_embedder():
    tpm_limit = _get_int_config("embed_tpm_limit", 120000)
    rpm_limit = _get_int_config("embed_rpm_limit", 60)
    batch_size = _get_int_config("embed_batch_size", 16)
    if USE_AZURE:
        return AzureEmbedder(
            deployment=CONFIG["azure_embed_deployment"],
            endpoint=CONFIG["azure_endpoint"],
            api_key=CONFIG["azure_api_key"],
            api_version=CONFIG["azure_api_version"],
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit,
            batch_size=batch_size,
        )
    return LocalOllamaEmbedder(
        model=CONFIG.get("ollama_embed_model") or CONFIG["local_embed_model"],
        base_url=CONFIG["ollama_base_url"],
        api_key=CONFIG["ollama_api_key"],
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
        batch_size=batch_size,
    )


def _get_generator_client():
    if USE_AZURE:
        return AzureOpenAI(
            api_key=CONFIG["azure_api_key"],
            api_version=CONFIG["azure_api_version"],
            azure_endpoint=CONFIG["azure_endpoint"],
            timeout=180.0,
        )
    return StandardOpenAI(
        base_url=CONFIG["ollama_base_url"],
        api_key=CONFIG["ollama_api_key"],
        timeout=180.0,
    )


def generate_answer(client, context, question):
    template = (
        "Answer the question based ONLY on the context.\n"
        "Explanation is not needed.\n"
        "Do not use your memory. Keep answers short and to the point.\n"
        "if not in context, respond with 'I don't know'.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    prompt = template.format(context=context, question=question)
    model_name = CONFIG["azure_gen_deployment"] if USE_AZURE else CONFIG["ollama_model"]
    attempts = 0
    while attempts < 2:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            msg = str(exc).lower()
            if "rate limit" in msg and attempts == 0:
                print("Rate limit hit during generation; retrying in 60s...")
                time.sleep(60)
                attempts += 1
                continue
            return f"Generation Error: {exc}"


# --- Retrieval primitives (UNCHANGED) ---
def build_whoosh_index(documents: List[Dict]) -> Optional["WhooshBM25Retriever"]:
    if not documents:
        return None
    index_dir = CONFIG["whoosh_index_dir"]
    rebuild = CONFIG.get("rebuild_whoosh", True)
    if exists_in(index_dir) and rebuild is False:
        return WhooshBM25Retriever(open_dir(index_dir), _get_int_config("top_k", 10))
    if os.path.exists(index_dir):
        for root, dirs, files in os.walk(index_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    os.makedirs(index_dir, exist_ok=True)
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
    )
    index = create_in(index_dir, schema)
    writer = index.writer(limitmb=CONFIG.get("whoosh_limit_mb", 1024))
    for doc in documents:
        writer.add_document(doc_id=doc["doc_id"], title=doc["title"], content=doc["content"])
    writer.commit()
    print(f"Whoosh docs indexed: {index.doc_count()}")
    return WhooshBM25Retriever(index, _get_int_config("top_k", 10))


def build_dense_index(documents: List[Dict], embedder) -> Optional["DenseFAISSRetriever"]:
    if not documents:
        return None
    vectors = embedder.encode([doc["content"] for doc in documents], convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, CONFIG["faiss_index_path"])
    metadata_payload = []
    for idx, doc in enumerate(documents):
        item = dict(doc)
        item["faiss_idx"] = idx
        metadata_payload.append(item)
    with open(CONFIG["faiss_metadata_path"], "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, ensure_ascii=False, indent=2)
    print(f"FAISS vectors stored: {len(documents)}")
    metadata_map = {str(idx): item for idx, item in enumerate(metadata_payload)}
    return DenseFAISSRetriever(index, metadata_map, embedder, _get_int_config("dense_top_k", 40))


def load_dense_index(embedder) -> Optional["DenseFAISSRetriever"]:
    if not (os.path.exists(CONFIG["faiss_index_path"]) and os.path.exists(CONFIG["faiss_metadata_path"])):
        return None
    try:
        index = faiss.read_index(CONFIG["faiss_index_path"])
        with open(CONFIG["faiss_metadata_path"], "r", encoding="utf-8") as f:
            documents = json.load(f)
        metadata_map = {}
        for idx, doc in enumerate(documents):
            key = str(doc.get("faiss_idx", idx))
            metadata_map[key] = doc
        return DenseFAISSRetriever(index, metadata_map, embedder, _get_int_config("dense_top_k", 40))
    except Exception as exc:
        print(f"Failed to load FAISS index: {exc}")
        return None


@dataclass
class RetrievedDoc:
    page_content: str
    metadata: Dict


class WhooshBM25Retriever:
    def __init__(self, index, k: int):
        self.index = index
        self.k = k
        self.parser = SimpleParser("content", schema=self.index.schema)

    def invoke(self, query: str) -> List[RetrievedDoc]:
        if not query:
            return []
        with self.index.searcher() as searcher:
            try:
                parsed = self.parser.parse(query)
            except Exception:
                parsed = self.parser.parse(re.sub(r"[^\w\s]", " ", query))
            hits = searcher.search(parsed, limit=self.k)
            results = []
            for rank, hit in enumerate(hits):
                results.append(
                    RetrievedDoc(
                        page_content=hit.get("content", ""),
                        metadata={
                            "title": hit.get("title", ""),
                            "doc_id": hit.get("doc_id", ""),
                            "rank": rank,
                        },
                    )
                )
            return results


class DenseFAISSRetriever:
    def __init__(self, index, metadata_map: Dict[str, Dict], embedder, k: int):
        self.index = index
        self.metadata_map = metadata_map
        self.embedder = embedder
        self.k = k

    def invoke(self, query: str) -> List[RetrievedDoc]:
        if not query:
            return []
        vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        scores, indices = self.index.search(vec, self.k)
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
                        "title": meta.get("title", ""),
                        "doc_id": doc_id,
                        "rank": rank,
                        "score": float(scores[0][rank]) if scores is not None else None,
                    },
                )
            )
        return results


def reciprocal_rank_fusion(result_sets: List[List[RetrievedDoc]], rrf_top_k: int):
    fused = {}
    rrf_constant = 60
    for result_set in result_sets:
        for rank, doc in enumerate(result_set):
            doc_id = doc.metadata.get("doc_id")
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


def rerank_with_cross_encoder(query: str, retrieved_docs: List[RetrievedDoc], batch_size: int = 5, top_k: int = 10) -> List[RetrievedDoc]:
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
        key = (doc.metadata.get("title", ""), doc.page_content[:100])
        if key in seen:
            continue
        seen.add(key)
        final_docs.append(doc)
        if len(final_docs) >= top_k:
            break
    return final_docs


class HybridRetriever:
    def __init__(self, bm25_retriever: Optional[WhooshBM25Retriever], dense_retriever: Optional[DenseFAISSRetriever], rrf_k: int, top_k: int):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.enable_reranker = _get_bool_config("enable_reranker", True)

    def invoke(self, query: str) -> List[RetrievedDoc]:
        result_sets = []
        if self.bm25_retriever:
            result_sets.append(self.bm25_retriever.invoke(query))
        if self.dense_retriever:
            result_sets.append(self.dense_retriever.invoke(query))
        if not result_sets:
            return []
        fused = result_sets[0][: self.top_k] if len(result_sets) == 1 else reciprocal_rank_fusion(result_sets, rrf_top_k=self.rrf_k)
        if not self.enable_reranker:
            logging.info("[RERANK] Disabled; returning fused results.")
            return fused[: self.top_k]
        return rerank_with_cross_encoder(
            query=query,
            retrieved_docs=fused,
            batch_size=_get_int_config("rerank_batch_size", 5),
            top_k=self.top_k,
        )


# --- Evaluation (UPDATED: no legacy cache reads; full-article context uses in-memory articles or cache read-only) ---
def run_evaluation(
    questions: List[Dict],
    retriever: HybridRetriever,
    generator_client=None,
    article_texts: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    results = []
    use_full_article = _get_bool_config("use_full_article_context", True)
    use_cache = _get_bool_config("use_wiki_cache", True)
    cache_index = _load_wiki_cache_index() if (use_cache and use_full_article) else {}
    xlsx_path = _get_results_paths()

    for idx, row in enumerate(tqdm(questions, desc="Evaluating", unit="q")):
        query = row["query"]
        target_titles = row["target_titles"]
        try:
            docs = retriever.invoke(query)
        except Exception as exc:
            logging.warning("Retriever failed for query '%s': %s", query, exc)
            error_result = {
                "query": query,
                "answer": f"Retriever Error: {exc}",
                "ground_truth": row.get("answer", ""),
                "contexts": [],
                "contexts_file": None,
                "target_titles": target_titles,
                "found_titles": [],
                "retriever_precision": 0.0,
                "retriever_recall": 0.0,
                "retriever_f1": 0.0,
            }
            results.append(error_result)
            if xlsx_path:
                _append_result_xlsx(error_result, xlsx_path)
            continue

        deduped = []
        seen_titles = set()

        for doc in docs:
            title = doc.metadata.get("title", "")
            if title and title in seen_titles:
                continue
            deduped.append(doc)
            if title:
                seen_titles.add(title)

        contexts = []
        for d in deduped:
            title = d.metadata.get("title", "")

            if use_full_article and title:
                # Prefer in-memory full article text
                if article_texts and title in article_texts and article_texts[title]:
                    contexts.append(article_texts[title])
                    continue

                # Read-only cache fallback
                if cache_index:
                    cached = _read_cached_wiki_by_key(title, cache_index)
                    if cached:
                        contexts.append(cached)
                        continue

            contexts.append(d.page_content)

        max_ctx_tokens = _get_int_config("llm_max_tokens", 0)
        if max_ctx_tokens > 0 and contexts:
            contexts = _truncate_contexts(contexts, max_ctx_tokens)

        contexts_file = _dump_contexts_to_file(contexts)

        found_titles = [d.metadata.get("title", "") for d in deduped]
        precision, recall, f1 = _compute_retriever_metrics(target_titles, found_titles)

        answer = None
        if generator_client is not None:
            answer = generate_answer(generator_client, "\n\n".join(contexts), query)

        results.append(
            {
                "query": query,
                "answer": answer,
                "ground_truth": row.get("answer", ""),
                "contexts": contexts,
                "contexts_file": contexts_file,
                "target_titles": target_titles,
                "found_titles": found_titles,
                "retriever_precision": precision,
                "retriever_recall": recall,
                "retriever_f1": f1,
            }
        )

        if xlsx_path:
            _append_result_xlsx(results[-1], xlsx_path)
    return results


def save_results(results: List[Dict]):
    if not results:
        return
    df = pd.DataFrame(results)
    xlsx_path = _get_results_paths()
    print(f"Results were written row-by-row to {xlsx_path}")
    print(f"Avg Retriever F1 (this run): {df['retriever_f1'].mean():.2f}")
    if "answer_correctness" in df.columns:
        print(f"Avg Answer Correctness (this run): {df['answer_correctness'].mean():.2f}")


def _load_existing_results(path: str) -> List[Dict]:
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_excel(path)
        return df.to_dict(orient="records")
    except Exception as exc:
        print(f"Failed to load existing results from {path}: {exc}")
        return []


def _append_result_xlsx(row: Dict, path: str):
    row_to_save = dict(row)
    row_to_save["contexts"] = row.get("contexts_file") or row.get("contexts")
    df_new = pd.DataFrame([row_to_save])
    if os.path.exists(path):
        try:
            df_existing = pd.read_excel(path)
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception as exc:
            print(f"Failed to read existing XLSX ({path}); writing new file. Error: {exc}")
    while True:
        try:
            df_new.to_excel(path, index=False)
            break
        except PermissionError:
            input(f"Please close {path} and press Enter to retry...")


def run_pipeline() -> List[Dict]:
    ds = load_frames_dataset()
    questions_all = build_questions(ds, limit=None)
    if not questions_all:
        print("No questions loaded; exiting.")
        return []

    questions_eval = build_questions(ds, limit=CONFIG.get("max_questions"))
    resume = _get_bool_config("resume_last", False)
    existing_results: List[Dict] = []
    processed_queries = set()
    if resume:
        existing_results = _load_existing_results(_get_results_paths())
        if existing_results:
            processed_queries = {str(x.get("query")) for x in existing_results if x.get("query")}
            before = len(questions_eval)
            questions_eval = [q for q in questions_eval if str(q.get("query")) not in processed_queries]
            skipped = before - len(questions_eval)
            print(f"Resume enabled: skipping {skipped} previously processed questions.")
            if not questions_eval:
                print("No new questions to evaluate; reusing existing results.")
                return existing_results
        else:
            print("Resume enabled but no existing results found; evaluating all questions.")

    # Collect unique links + titles (indexing uses the full dataset)
    all_links: List[str] = []
    all_titles: List[str] = []
    for row in questions_all:
        all_links.extend(row.get("target_links", []))
        all_titles.extend(row.get("target_titles", []))

    unique_links = list({x for x in all_links if x})
    unique_titles = list({t for t in all_titles if t})

    rng = random.Random(CONFIG["seed"])
    rng.shuffle(unique_links)
    rng.shuffle(unique_titles)

    cap = CONFIG.get("max_wiki_titles")
    if cap and cap > 0:
        unique_links = unique_links[:cap]
        unique_titles = unique_titles[:cap]

    index_dir = CONFIG.get("whoosh_index_dir", DEFAULT_WHOOSH_INDEX_DIR)
    rebuild_whoosh = CONFIG.get("rebuild_whoosh", True)

    dense_paths_exist = os.path.exists(CONFIG["faiss_index_path"]) and os.path.exists(CONFIG["faiss_metadata_path"])
    rebuild_dense = CONFIG.get("rebuild_dense", True)

    whoosh_exists = exists_in(index_dir)
    need_whoosh = rebuild_whoosh or not whoosh_exists

    # If rebuild_dense is False we skip dense entirely even if files are missing.
    need_dense = rebuild_dense

    need_docs = need_whoosh or need_dense
    documents = None
    articles = None

    if need_docs:
        # NEW: read from cache (index.json) OR fetch from API depending on use_wiki_cache
        articles = fetch_wikipedia_articles(unique_links, unique_titles)
        if not articles:
            print("No articles available (cache empty or API fetch failed). Exiting.")
            return
        documents = build_documents(articles)

    # BM25
    if need_whoosh:
        bm25_retriever = build_whoosh_index(documents)
    else:
        bm25_retriever = WhooshBM25Retriever(open_dir(index_dir), _get_int_config("top_k", 10))
        print(f"Reusing existing Whoosh index at {index_dir} (set REBUILD_INDEX=1 to rebuild).")

    # Dense
    dense_retriever = None
    if rebuild_dense:
        embedder = _get_embedder()
        if embedder:
            if need_dense:
                if documents is None:
                    # Need docs to rebuild dense index
                    articles = fetch_wikipedia_articles(unique_links, unique_titles)
                    if not articles:
                        print("No articles available; cannot build dense index.")
                        return
                    documents = build_documents(articles)
                dense_retriever = build_dense_index(documents, embedder)
            elif dense_paths_exist:
                dense_retriever = load_dense_index(embedder)
    elif dense_paths_exist:
        embedder = _get_embedder()
        if embedder:
            dense_retriever = load_dense_index(embedder)

    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=_get_int_config("rrf_top_k", 60),
        top_k=_get_int_config("hybrid_top_k", 10),
    )

    generator_client = _get_generator_client()

    # Ensure we have article_texts for full-article context even if indexes were reused
    if articles is None:
        # best-effort: load what we can (cache mode: from cache; fresh mode: fetch)
        articles = fetch_wikipedia_articles(unique_links, unique_titles)

    results = run_evaluation(
        questions_eval,
        retriever,
        generator_client=generator_client,
        article_texts=articles,
    )

    if resume and existing_results:
        results = existing_results + results
    return results


def main():
    results = run_pipeline()
    if not results:
        print("No results to save.")
        return
    save_results(results)


if __name__ == "__main__":
    main()
