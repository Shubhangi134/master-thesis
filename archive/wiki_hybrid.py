import os
import json
import re
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
from urllib.parse import unquote
import hashlib

import ftfy
import numpy as np
import pandas as pd
import requests
from datasets import load_dataset
from datasets import Dataset
from dotenv import load_dotenv
import faiss
import tiktoken
from tqdm import tqdm
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import QueryParser, OrGroup

# OpenAI / Azure clients
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai import OpenAI as StandardOpenAI, AsyncOpenAI as StandardAsyncOpenAI

from ragas import evaluate as ragas_evaluate
from ragas.metrics import AnswerCorrectness
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.run_config import RunConfig

load_dotenv(".env")

USE_AZURE = bool(os.getenv("ENDPOINT"))

CONFIG = {
    # Dataset + sampling
    "frames_dataset": "google/frames-benchmark",
    "frames_split": "test",
    "seed": 42,
    "max_questions": int(os.getenv("FRAMES_MAX_QUESTIONS", 5)),
    "max_wiki_titles": int(os.getenv("MAX_WIKI_TITLES", 0)),  # cap titles to fetch; 0 = no cap

    # Wikipedia API
    "wiki_language": os.getenv("WIKI_LANG", "en"),
    "wiki_batch_size": int(os.getenv("WIKI_BATCH_SIZE", 20)),
    "wiki_sleep": float(os.getenv("WIKI_SLEEP", 0.05)),
    "wiki_cache_dir": os.getenv("WIKI_CACHE_DIR", "wiki_data"),

    # Chunking
    "chunk_size": int(os.getenv("CHUNK_SIZE", 800)),
    "overlap": int(os.getenv("CHUNK_OVERLAP", 150)),

    # Retrieval
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 40)),
    "dense_top_k": int(os.getenv("DENSE_TOP_K", 40)),
    "hybrid_top_k": int(os.getenv("HYBRID_TOP_K", 4)),
    "rrf_k": int(os.getenv("RRF_K", 60)),
    "rebuild_whoosh": int(os.getenv("REBUILD_WHOOSH_INDEX", 1)) > 0,
    "rebuild_dense": int(os.getenv("REBUILD_DENSE_INDEX", 1)) > 0,
    "whoosh_limit_mb": int(os.getenv("WHOOSH_LIMIT_MB", 1024)),
    "whoosh_index_dir": "whoosh_wiki_index",
    "faiss_index_path": "faiss_wiki_index.bin",
    "faiss_metadata_path": "faiss_wiki_metadata.json",
    # Embeddings / generation
    "embed_batch_size": int(os.getenv("EMBED_BATCH_SIZE", 4)),
    "embed_tpm_limit": int(os.getenv("EMBED_TPM_LIMIT", 100_000)),
    "embed_rpm_limit": int(os.getenv("EMBED_RPM_LIMIT", 60)),
    "embed_max_tokens": int(os.getenv("EMBED_MAX_TOKENS", 3000)),
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

    # Evaluation output
    "results_csv": "Frames_Wiki_RAG.csv",
    # Ragas config
    "ragas_timeout": 600,
    "ragas_max_workers": 2,
    "ragas_batch_size": 5,
    "ragas_max_wait": 30,
    "ragas_max_retries": 15,
}


# --- Helpers ---
def _get_int_config(name: str, default: int) -> int:
    try:
        value = CONFIG.get(name, default)
        return int(value if value is not None else default)
    except (TypeError, ValueError):
        return default


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


def _cache_path_for_title(title: str) -> str:
    cache_dir = CONFIG.get("wiki_cache_dir", "wiki_data")
    os.makedirs(cache_dir, exist_ok=True)
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", title).strip("_")
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()[:8]
    fname = f"{base[:80]}_{digest}.txt" if base else f"article_{digest}.txt"
    return os.path.join(cache_dir, fname)


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
            # Fallback
            text = text[:2500]
        truncated.append(text)
    return truncated


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
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
        return arr


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
            vectors = self._embed_via_http(truncated_batch)
            all_vectors.extend(vectors)
        arr = np.array(all_vectors, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
        return arr


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


def _get_ragas_clients():
    if USE_AZURE:
        client = AsyncAzureOpenAI(
            api_key=CONFIG["azure_api_key"],
            api_version=CONFIG["azure_api_version"],
            azure_endpoint=CONFIG["azure_endpoint"],
            timeout=180.0,
        )
        llm = llm_factory(model=CONFIG["azure_gen_deployment"], client=client)
        embeddings = OpenAIEmbeddings(model=CONFIG["azure_embed_deployment"], client=client)
    else:
        client = StandardAsyncOpenAI(
            base_url=CONFIG["ollama_base_url"],
            api_key=CONFIG["ollama_api_key"],
            timeout=180.0,
        )
        llm = llm_factory(model=CONFIG["ollama_model"], client=client)
        embed_model = CONFIG.get("ollama_embed_model") or CONFIG["local_embed_model"]
        embeddings = OpenAIEmbeddings(model=embed_model, client=client)
    return llm, embeddings


def generate_answer(client, context, question):
    template = (
        "Answer the question based ONLY on the context.\n"
        "Explanation is not needed.\n"
        "Do not use use your memory. Keep answers short and to the point.\n"
        "if not in context, respond with 'I don't know'.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    try:
        prompt = template.format(context=context, question=question)
        model_name = CONFIG["azure_gen_deployment"] if USE_AZURE else CONFIG["ollama_model"]
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an answer evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"Generation Error: {exc}"


# --- Wikipedia ingestion ---
def load_frames_dataset():
    print("Loading FRAMES benchmark...")
    return load_dataset(CONFIG["frames_dataset"], split=CONFIG["frames_split"])


def build_questions(ds, limit: Optional[int]) -> List[Dict]:
    sampled = ds
    if limit and limit > 0 and len(ds) > limit:
        sampled = ds.shuffle(seed=CONFIG["seed"]).select(range(limit))
    questions = []
    for row in sampled:
        prompt = row.get("Prompt", "")
        answer = row.get("Answer", "")
        links = _extract_links(row)
        titles = [_url_to_title(link) for link in links]
        titles = [t for t in titles if t]
        if not prompt or not titles:
            continue
        questions.append(
            {
                "query": prompt,
                "answer": answer,
                "target_titles": titles,
            }
        )
    print(f"Prepared {len(questions)} questions (limit={limit or 'all'}).")
    return questions


def fetch_wikipedia_articles(titles: List[str]) -> Dict[str, str]:
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
    articles = {}
    # Load from cache first
    missing_titles = []
    for title in titles:
        cache_path = _cache_path_for_title(title)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    articles[title] = f.read()
            except OSError:
                missing_titles.append(title)
        else:
            missing_titles.append(title)

    cached_count = len(articles)
    if cached_count:
        print(f"Loaded {cached_count} articles from cache.")

    titles_to_fetch = missing_titles
    fetch_batches = list(_batch_iterable(titles_to_fetch, batch_size))
    print(f"Fetching {len(titles_to_fetch)} Wikipedia articles in {len(fetch_batches)} batches (lang={lang}).")
    for batch in tqdm(fetch_batches, desc="Fetching Wikipedia", unit="batch"):
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "format": "json",
            "redirects": 1,
            "titles": "|".join(batch),
            "exintro": 1,
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
                # Respectful pause on rate limit / forbidden
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
                # Cache to disk
                cache_path = _cache_path_for_title(title)
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        f.write(text)
                except OSError as exc:
                    print(f"Failed to cache article '{title}': {exc}")
        if sleep_for:
            time.sleep(sleep_for)
    fetched_count = len(articles) - cached_count
    print(f"Fetched {fetched_count} articles from Wikipedia (cached {cached_count}).")
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
def build_whoosh_index(documents: List[Dict]) -> Optional["WhooshRetriever"]:
    if not documents:
        return None
    index_dir = CONFIG["whoosh_index_dir"]
    rebuild = CONFIG.get("rebuild_whoosh", True)
    if exists_in(index_dir) and rebuild is False:
        return WhooshRetriever(open_dir(index_dir), _get_int_config("top_k", 10))
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
        # Store content so it can be returned as context in evaluation
        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
    )
    index = create_in(index_dir, schema)
    writer = index.writer(limitmb=CONFIG.get("whoosh_limit_mb", 1024))
    for doc in documents:
        writer.add_document(doc_id=doc["doc_id"], title=doc["title"], content=doc["content"])
    writer.commit()
    print(f"Whoosh docs indexed: {index.doc_count()}")
    return WhooshRetriever(index, _get_int_config("top_k", 10))


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


class WhooshRetriever:
    def __init__(self, index, k: int):
        self.index = index
        self.k = k
        self.parser = QueryParser("content", schema=self.index.schema, group=OrGroup.factory(0.9))

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


def reciprocal_rank_fusion(result_sets: List[List[RetrievedDoc]], k: int = 60, max_results: int = None):
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


class HybridRetriever:
    def __init__(self, bm25_retriever: Optional[WhooshRetriever], dense_retriever: Optional[DenseFAISSRetriever], rrf_k: int, top_k: int):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.top_k = top_k

    def invoke(self, query: str) -> List[RetrievedDoc]:
        result_sets = []
        if self.bm25_retriever:
            result_sets.append(self.bm25_retriever.invoke(query))
        if self.dense_retriever:
            result_sets.append(self.dense_retriever.invoke(query))
        if not result_sets:
            return []
        if len(result_sets) == 1:
            return result_sets[0][: self.top_k]
        return reciprocal_rank_fusion(result_sets, k=self.rrf_k, max_results=self.top_k)


# --- Evaluation ---
def run_evaluation(questions: List[Dict], retriever: HybridRetriever, generator_client=None) -> List[Dict]:
    results = []
    for idx, row in enumerate(tqdm(questions, desc="Evaluating", unit="q")):
        query = row["query"]
        target_titles = row["target_titles"]
        docs = retriever.invoke(query)
        deduped = []
        seen_titles = set()
        # Fallback: if any doc is missing page_content, attempt to reload from cache by title
        for doc in docs:
            title = doc.metadata.get("title", "")
            if title and title in seen_titles:
                continue
            if not doc.page_content and title:
                cache_path = _cache_path_for_title(title)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            doc.page_content = f.read()
                    except OSError:
                        pass
            deduped.append(doc)
            if title:
                seen_titles.add(title)
        contexts = []
        for d in deduped:
            title = d.metadata.get("title", "")
            if title:
                cache_path = _cache_path_for_title(title)
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            contexts.append(f.read())
                            continue
                    except OSError:
                        pass
            contexts.append(d.page_content)
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
                "target_titles": target_titles,
                "found_titles": found_titles,
                "retriever_precision": precision,
                "retriever_recall": recall,
                "retriever_f1": f1,
            }
        )
    return results


def save_results(results: List[Dict]):
    if not results:
        return
    df = pd.DataFrame(results)
    while True:
        try:
            df.to_csv(CONFIG["results_csv"], index=False)
            break
        except PermissionError:
            input(f"Please close {CONFIG['results_csv']} and press Enter to retry...")
    print(f"Saved results to {CONFIG['results_csv']}")
    print(f"Avg Retriever F1: {df['retriever_f1'].mean():.2f}")
    if "answer_correctness" in df.columns:
        print(f"Avg Answer Correctness: {df['answer_correctness'].mean():.2f}")
    print(f"Results file: {CONFIG['results_csv']}")


def main():
    ds = load_frames_dataset()
    questions_all = build_questions(ds, limit=None)
    if not questions_all:
        print("No questions loaded; exiting.")
        return
    questions_eval = build_questions(ds, limit=CONFIG.get("max_questions"))

    titles = []
    for row in questions_all:
        titles.extend(row["target_titles"])
    unique_titles = list({t for t in titles if t})
    cap = CONFIG.get("max_wiki_titles")
    if cap and cap > 0:
        unique_titles = unique_titles[:cap]
    random.Random(CONFIG["seed"]).shuffle(unique_titles)

    articles = fetch_wikipedia_articles(unique_titles)
    documents = build_documents(articles)

    bm25_retriever = build_whoosh_index(documents)
    embedder = _get_embedder()
    dense_retriever = None
    if embedder:
        if CONFIG.get("rebuild_dense", True):
            dense_retriever = build_dense_index(documents, embedder)
        else:
            dense_retriever = load_dense_index(embedder)
    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        rrf_k=_get_int_config("rrf_k", 60),
        top_k=_get_int_config("hybrid_top_k", 10),
    )

    generator_client = _get_generator_client()
    results = run_evaluation(questions_eval, retriever, generator_client=generator_client)

    # Ragas AnswerCorrectness evaluation
    try:
        ragas_llm, ragas_embeddings = _get_ragas_clients()
        ragas_ds = Dataset.from_dict({
            "question": [str(x["query"]) for x in results],
            "answer": [str(x["answer"] or "") for x in results],
            "contexts": [x.get("contexts", []) for x in results],
            "ground_truth": [str(x.get("ground_truth") or "") for x in results],
        })
        ragas_run_cfg = RunConfig(
            timeout=CONFIG["ragas_timeout"],
            max_workers=CONFIG["ragas_max_workers"],
            max_retries=CONFIG["ragas_max_retries"],
            max_wait=CONFIG["ragas_max_wait"],
        )
        ragas_scores = ragas_evaluate(
            dataset=ragas_ds,
            metrics=[AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings)],
            run_config=ragas_run_cfg,
            batch_size=CONFIG["ragas_batch_size"],
        )
        ragas_df = ragas_scores.to_pandas()
        if "answer_correctness" in ragas_df.columns and len(ragas_df) == len(results):
            for idx, score in enumerate(ragas_df["answer_correctness"].tolist()):
                results[idx]["answer_correctness"] = score
        print("Ragas AnswerCorrectness (per-sample mean):", ragas_df["answer_correctness"].mean())
    except Exception as exc:
        print(f"Ragas evaluation failed: {exc}")

    save_results(results)


if __name__ == "__main__":
    main()
