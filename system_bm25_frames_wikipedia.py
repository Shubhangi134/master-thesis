"""
BM25-only pipeline for FRAMES + Wikipedia:
1. Fetch FRAMES benchmark dataset from Hugging Face.
2. Collect linked Wikipedia articles via the API OR read from local cache (read-only).
3. Chunk and clean articles.
4. Index chunks in Whoosh (BM25) and run evaluation with optional generation + Ragas.

CACHE RULES (updated):
- This code NEVER writes Wikipedia content to cache anymore.
- New config: use_wiki_cache
  - True  -> read ONLY from wiki_cache_dir using index.json (link -> filename mapping)
  - False -> fetch fresh from Wikipedia API (no cache read, no cache write)
"""
import os
import json
import re
import time
import random
import logging
import uuid
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional, Tuple
from urllib.parse import unquote

import ftfy
import pandas as pd
import requests
from datasets import load_dataset
try:
    import tiktoken
except ImportError:
    tiktoken = None
from dotenv import load_dotenv
from tqdm import tqdm
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import SimpleParser

# OpenAI / Azure clients
from openai import AzureOpenAI
from openai import OpenAI as StandardOpenAI

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)

USE_AZURE = bool(os.getenv("ENDPOINT"))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WIKI_CACHE_DIR = os.path.join(SCRIPT_DIR, "wiki_data")

CONFIG = {
    # Dataset + sampling
    "frames_dataset": "google/frames-benchmark",
    "frames_split": "test",
    "seed": 42,
    "max_questions": int(os.getenv("FRAMES_MAX_QUESTIONS", 10)),
    "max_wiki_titles": int(os.getenv("MAX_WIKI_TITLES", 10)),  # cap titles/links to fetch; 0 = no cap

    # Wikipedia API / Cache
    "wiki_language": os.getenv("WIKI_LANG", "en"),
    "wiki_batch_size": int(os.getenv("WIKI_BATCH_SIZE", 20)),
    "wiki_sleep": float(os.getenv("WIKI_SLEEP", 0.05)),
    "wiki_cache_dir": os.getenv("WIKI_CACHE_DIR", DEFAULT_WIKI_CACHE_DIR),
    "use_wiki_cache": os.getenv("USE_WIKI_CACHE", "1"),  # NEW: "1"/"0", "true"/"false", etc.

    # Chunking
    "chunk_size": int(os.getenv("CHUNK_SIZE", 800)),
    "overlap": int(os.getenv("CHUNK_OVERLAP", 150)),

    # Retrieval
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 40)),
    "rebuild_whoosh": int(os.getenv("REBUILD_INDEX", 1)) > 0,
    "whoosh_limit_mb": int(os.getenv("WHOOSH_LIMIT_MB", 1024)),
    "whoosh_index_dir": "whoosh_wiki_index",

    # LLM / embeddings
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

    # Context/token limits
    "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", 30_000)),  # 0 = no truncation; applies to combined contexts
    "resume_last": os.getenv("RESUME_LAST", "1"),  # "1"/"true" to reuse existing results and skip processed queries
    "context_dump_dir": os.getenv("CONTEXT_DUMP_DIR", os.path.join(SCRIPT_DIR, "retriever_contexts")),

    # Evaluation output
    "results_xlsx": "Frames_Wiki_RAG_BM25.xlsx",
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


def _truncate_contexts(contexts: List[str], max_tokens: int) -> List[str]:
    """
    Truncate the combined contexts to a total token budget by cutting each chunk evenly.
    """
    if not contexts or max_tokens is None or max_tokens <= 0:
        return contexts

    if tiktoken is None:
        # Fallback: approximate with words if tiktoken is unavailable.
        per_chunk_words = max(1, max_tokens // len(contexts))
        return [" ".join(text.split()[:per_chunk_words]) for text in contexts]

    # GPT-4.1 uses the o200k_base tokenizer (same as GPT-4o)
    encoding = tiktoken.get_encoding("o200k_base")
    per_chunk = max(1, max_tokens // len(contexts))
    truncated = []
    for text in contexts:
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:per_chunk]
        truncated.append(encoding.decode(truncated_tokens))
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
            for idx, ctx in enumerate(contexts, 1):
                f.write(f"=== Context {idx} ===\n")
                f.write(str(ctx))
                f.write("\n\n")
        return fname
    except Exception as exc:
        logging.warning("Failed to write contexts to %s: %s", path, exc)
        return None


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


def log_config_overview():
    logging.info("[CONFIG] Mode: %s", "AZURE" if USE_AZURE else "LOCAL/Ollama")
    logging.info("[CONFIG] Retrieval (BM25 only): top_k=%s", _get_int_config("top_k", 40))
    logging.info("[CONFIG] Wiki cache dir: %s", CONFIG.get("wiki_cache_dir"))
    logging.info("[CONFIG] use_wiki_cache: %s", _get_bool_config("use_wiki_cache", True))
    logging.info("[CONFIG] Context uses full articles: %s", _get_bool_config("use_full_article_context", True))
logging.info("[CONFIG] Max context tokens (LLM_MAX_TOKENS): %s", _get_int_config("llm_max_tokens", 0) or "disabled")
logging.info("[CONFIG] Resume last: %s", _get_bool_config("resume_last", False))


log_config_overview()

# --- LLM clients ---
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


# --- Cache helpers (READ-ONLY) ---
def _load_wiki_cache_index() -> Dict[str, str]:
    """
    Loads wiki_cache_dir/index.json mapping of (link -> filename).
    Returns dict: normalized_key -> absolute_file_path.
    Supports:
      - {"<link>": "<file>", ...}
      - [{"link": "<link>", "filename": "<file>"}, ...]
    Also stores a derived title key for each link: _url_to_title(link)
    """
    cache_dir = CONFIG.get("wiki_cache_dir", "wiki_data")
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

        # Key by link (primary requirement)
        norm_to_path[_normalize_title(link)] = abs_path

        # Also key by title derived from link (helps later lookups by title)
        derived_title = _url_to_title(link)
        if derived_title:
            norm_to_path[_normalize_title(derived_title)] = abs_path

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
        links = _extract_links(row)  # keep original links (needed for cache index lookups)
        titles = [_url_to_title(link) for link in links]
        titles = [t for t in titles if t]

        if not prompt or not links or not titles:
            continue

        questions.append(
            {
                "query": prompt,
                "answer": answer,
                "target_links": links,     # NEW: for cache lookup
                "target_titles": titles,   # for evaluation and API fetch
            }
        )
    print(f"Prepared {len(questions)} questions (limit={limit or 'all'}).")
    return questions


def fetch_wikipedia_articles(
    links: List[str],
    titles: List[str],
) -> Dict[str, str]:
    """
    If use_wiki_cache=True:
      - Read ONLY from wiki_cache_dir/index.json link->filename mapping
      - DO NOT call Wikipedia API
      - DO NOT write cache

    If use_wiki_cache=False:
      - Fetch from Wikipedia API using titles
      - DO NOT read cache
      - DO NOT write cache

    Returns dict keyed by *title* (best-effort).
    """
    use_cache = _get_bool_config("use_wiki_cache", True)

    if use_cache:
        cache_index = _load_wiki_cache_index()
        if not cache_index:
            logging.warning("use_wiki_cache=True but cache index.json is missing/empty.")

        articles: Dict[str, str] = {}
        missing = 0

        # Primary requirement: index.json is link -> filename mapping, so use links first.
        for link in links:
            text = _read_cached_wiki_by_key(link, cache_index)
            if not text:
                missing += 1
                continue
            title = _url_to_title(link) or link
            articles[title] = text

        if missing:
            logging.warning(
                "Cache mode enabled. %d/%d links missing from cache (no API fetch will occur).",
                missing, len(links)
            )

        print(f"Loaded {len(articles)} articles from cache (use_wiki_cache=True).")
        return articles

    # Fresh mode (no cache read/write)
    titles = [t for t in titles if t]
    if not titles:
        return {}

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
    for title, text in tqdm(articles.items(), desc="Chunking articles", unit="article"):
        clean_text = _clean_text(text)
        chunks = _chunk_text(clean_text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            documents.append(
                {
                    "doc_id": f"{_normalize_title(title)}-{idx}",
                    "title": title,
                    "content": chunk,
                }
            )
    print(f"Prepared {len(documents)} chunks.")
    return documents


# --- Retrieval primitives ---
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


# --- Evaluation (UPDATED: no cache writes; full-article context uses in-memory articles or cache read-only) ---
def run_evaluation(
    questions: List[Dict],
    retriever: WhooshBM25Retriever,
    generator_client=None,
    article_texts: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    results = []
    use_full_article = _get_bool_config("use_full_article_context", True)
    use_cache = _get_bool_config("use_wiki_cache", True)
    cache_index = _load_wiki_cache_index() if (use_cache and use_full_article) else {}
    xlsx_path = CONFIG.get("results_xlsx")

    for _, row in enumerate(tqdm(questions, desc="Evaluating", unit="q")):
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

        # de-dupe by title
        deduped: List[RetrievedDoc] = []
        seen_titles = set()
        for doc in docs:
            title = doc.metadata.get("title", "")
            if title and title in seen_titles:
                continue
            deduped.append(doc)
            if title:
                seen_titles.add(title)

        contexts: List[str] = []
        for d in deduped:
            title = d.metadata.get("title", "")

            if use_full_article and title:
                # Prefer in-memory full article text (works for both cache and fresh).
                if article_texts and title in article_texts and article_texts[title]:
                    contexts.append(article_texts[title])
                    continue

                # Cache read-only fallback (best-effort).
                if cache_index:
                    cached = _read_cached_wiki_by_key(title, cache_index)
                    if cached:
                        contexts.append(cached)
                        continue

            # Fallback to chunk content from Whoosh hit.
            contexts.append(d.page_content)

        max_ctx_tokens = _get_int_config("llm_max_tokens", 0)
        if max_ctx_tokens > 0 and contexts:
            contexts = _truncate_contexts(contexts, max_ctx_tokens)

        contexts_file = _dump_contexts_to_file(contexts)

        found_titles = [d.metadata.get("title", "") for d in deduped]
        precision, recall, f1 = _compute_retriever_metrics(target_titles, found_titles)

        answer = None
        # if generator_client is not None:
        #     answer = generate_answer(generator_client, "\n\n".join(contexts), query)

        results.append(
            {
                "query": query,
                "answer": answer,
                "ground_truth": row.get("answer", ""),
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
    print(f"Results were written row-by-row to {CONFIG['results_xlsx']}")
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
    row_to_save["contexts_file"] = row.get("contexts_file")
    row_to_save.pop("contexts", None)  # remove large contexts field if present
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


def _append_result_xlsx(row: Dict, path: str):
    """
    Append a single result row to the XLSX file without dropping existing data.
    """
    df_new = pd.DataFrame([row])
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
        existing_results = _load_existing_results(CONFIG.get("results_xlsx", ""))
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

    index_dir = CONFIG.get("whoosh_index_dir", "whoosh_wiki_index")
    rebuild_index = CONFIG.get("rebuild_whoosh", True)
    index_exists = exists_in(index_dir)

    # Collect unique links + titles
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

    articles: Dict[str, str] = {}
    bm25_retriever = None

    # Only fetch/chunk when rebuilding or when index is missing
    if rebuild_index or not index_exists:
        articles = fetch_wikipedia_articles(unique_links, unique_titles)
        documents = build_documents(articles)
        bm25_retriever = build_whoosh_index(documents)
    else:
        bm25_retriever = WhooshBM25Retriever(open_dir(index_dir), _get_int_config("top_k", 10))
        print(f"Reusing existing Whoosh index at {index_dir}; set REBUILD_INDEX=1 to rebuild.")

    if not bm25_retriever:
        print("Failed to build or load BM25 index; exiting.")
        return []

    generator_client = _get_generator_client()

    # IMPORTANT: pass `articles` so full-article context works without cache writes
    results = run_evaluation(
        questions_eval,
        bm25_retriever,
        generator_client=generator_client,
        article_texts=articles if articles else None,
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
