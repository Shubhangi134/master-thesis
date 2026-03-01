import os
import json
import re
import pandas as pd
import warnings
import logging
import ftfy
import shutil
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from urllib.parse import unquote
try:
    import tiktoken
except ImportError:
    tiktoken = None

# --- 1. IMPORTS ---

from openai import AzureOpenAI, AsyncAzureOpenAI
from openai import OpenAI as StandardOpenAI, AsyncOpenAI as StandardAsyncOpenAI

from ragas.embeddings import LangchainEmbeddingsWrapper
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
from datasets import Dataset

from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import SimpleParser
from whoosh.query import Term, Or

from dotenv import load_dotenv

from datasets import load_dataset
from tqdm import tqdm

from query_expansion_helper import expand_query_for_bm25

from hopping import make_hopping_invoke

load_dotenv(".env")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WIKI_CACHE_DIR = os.path.join(SCRIPT_DIR, "wiki_data")


logging.basicConfig(level=logging.INFO)
logging.getLogger("ragas").setLevel(logging.DEBUG)

# Suppress Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


USE_AZURE = bool(os.getenv("ENDPOINT"))

CONFIG = {
    # --- MODES ---
    "doc_mode": os.getenv("DOC_MODE", "pdf"),

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
    "local_embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "ollama_embed_model": os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large:latest"),
    "reuse_llm_outputs": False,
    "llm_cache_path": "experiment_cache.json",
        
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
    "top_k": int(os.getenv("SPARSE_TOP_K", os.getenv("RETRIEVER_TOP_K", 5))),
    "whoosh_index_dir": os.getenv("WHOOSH_INDEX_DIR", "whoosh_pdf_index"),
    "whoosh_index_dir_frames": os.getenv("WHOOSH_INDEX_DIR_FRAMES", "whoosh_wiki_index"),
    "rebuild_index": int(os.getenv("REBUILD_SPARSE_INDEX", os.getenv("REBUILD_INDEX", 1))),
    "generation_prompt": None,  # Optional custom prompt template using {context} and {question}

    # Critical: Ollama Base URL for OpenAI compatibility
    "ollama_base_url": "http://localhost:11434/v1", 
    "ollama_model": "gpt-oss:120b-cloud",
    "ollama_api_key": "ollama", # Dummy key required by client

    # Query Expansion
    "abbrev_map_path": os.getenv("ABBREV_MAP_PATH", "abbreviations.json"),
    "bm25_enable_query_expansion": int(os.getenv("BM25_ENABLE_QUERY_EXPANSION", 0)),
    "bm25_max_query_expansions": int(os.getenv("BM25_MAX_QUERY_EXPANSIONS", 10)),
    
    # --- HOPPING (Query-Rewrite Multi-hop) ---
    "enable_hopping": int(os.getenv("ENABLE_HOPPING", 0)),
    "hop_max_hops": int(os.getenv("HOP_MAX_HOPS", 2)),
    "hop_evidence_docs": int(os.getenv("HOP_EVIDENCE_DOCS", 8)),
    "hop_evidence_chars": int(os.getenv("HOP_EVIDENCE_CHARS", 1200)),
    "hop_query_max_tokens": int(os.getenv("HOP_QUERY_MAX_TOKENS", 40)),

    # --- FRAMES / WIKIPEDIA ---
    "frames_dataset": os.getenv("FRAMES_DATASET", "google/frames-benchmark"),
    "frames_split": os.getenv("FRAMES_SPLIT", "test"),
    "frames_max_wiki_titles": int(os.getenv("MAX_WIKI_TITLES", 0)),
    "wiki_cache_dir": os.getenv("WIKI_CACHE_DIR", DEFAULT_WIKI_CACHE_DIR),
    "whoosh_limit_mb": int(os.getenv("WHOOSH_LIMIT_MB", 1024)),

    # Context/token limits
    "llm_max_tokens": int(os.getenv("LLM_MAX_TOKENS", 0)),

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

# Frames-specific concise QA prompt (cache/Wikipedia snippets)
FRAMES_GENERATION_PROMPT = """Answer the question based ONLY on the provided context.
Keep it short and factual. If not answerable from context, say "I don't know".

Context:
{context}

Question: {question}"""


# ------------------------------------------
# Helper config accessors (shared)
# ------------------------------------------
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


def _parse_question_range(value: str | None):
    """
    Parse QUESTION_RANGE env (e.g., "1-10" or "5") into (start, end) 1-based inclusive.
    Returns None if not provided or invalid.
    """
    if not value:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        m = re.match(r"^(\d+)(?:\s*-\s*(\d+))?$", text)
        if not m:
            return None
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if start <= 0 or end <= 0:
            return None
        if end < start:
            start, end = end, start
        return start, end
    except Exception:
        return None


def _apply_question_range(seq: List, qrange):
    """
    Apply (start, end) 1-based inclusive slice to a list-like sequence.
    """
    if not qrange:
        return seq
    start, end = qrange
    start_idx = max(0, start - 1)
    end_idx = max(start_idx, end - 1)
    return seq[start_idx : end_idx + 1]


def log_config_overview():
    logging.info("[CONFIG] Mode: %s", "AZURE" if USE_AZURE else "LOCAL/Ollama")
    logging.info("[CONFIG] Retrieval top_k=%s", _get_int_config("top_k", 5))
    logging.info("[CONFIG] PDF chunks size=%s overlap=%s", CONFIG.get("chunk_size_pdf"), CONFIG.get("chunk_overlap_pdf"))
    logging.info("[CONFIG] WIKI chunks size=%s overlap=%s", CONFIG.get("chunk_size_wiki"), CONFIG.get("chunk_overlap_wiki"))
    logging.info("[CONFIG] Whoosh index dir (pdf/hybrid): %s", CONFIG.get("whoosh_index_dir"))
    logging.info("[CONFIG] Whoosh frames index dir: %s", CONFIG.get("whoosh_index_dir_frames"))
    logging.info("[CONFIG] Question range: %s", CONFIG.get("question_range") or "all")
    logging.info("[CONFIG] LLm max tokens: %s", _get_int_config("llm_max_tokens", 0) or "disabled")

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

        generator_client = StandardOpenAI(
            base_url=CONFIG["ollama_base_url"],
            api_key=CONFIG["ollama_api_key"],
            timeout=180.0
        )
        
        ragas_client = StandardAsyncOpenAI(
            base_url=CONFIG["ollama_base_url"],
            api_key=CONFIG["ollama_api_key"],
            timeout=180.0
        )
        
        ragas_judge = llm_factory(
            model=CONFIG["ollama_model"],
            client=ragas_client,
            max_tokens=1024,
            max_completion_tokens=1024,
        )
        
     
        ragas_embeddings = RagasOpenAIEmbeddings(
            model=CONFIG["ollama_embed_model"],
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
    if not question or not context:
        return "I do not know"

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
      - Fix hyphenation across line breaks
      - Remove obvious headers/footers/separators
      - Collapse broken spacing
      - Preserve punctuation/casing
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^page\s+\d+(\s*of\s*\d+)?$", stripped, re.IGNORECASE):
            continue
        if re.match(r"^\d+$", stripped):
            continue
        if re.match(r"^[\.\-\=]{5,}$", stripped):
            continue
        cleaned_lines.append(stripped)

    text = " ".join(cleaned_lines)
    text = re.sub(r"(?<=\b\w) (?!\s)(?=\w\b)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = ftfy.fix_text(text.strip(), normalization="NFKC")
    return text


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return ftfy.fix_text(text.strip(), normalization="NFKC")


def chunk_text_fixed(text: str, *, chunk_size_chars: int, overlap_chars: int) -> List[Dict]:
    if not text:
        return []
    chunk_size = max(1, int(chunk_size_chars))
    overlap = max(0, int(overlap_chars))
    step = max(1, chunk_size - overlap)
    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(
            {
                "text": text[start:end],
                "char_start": start,
                "char_end": end,
            }
        )
        start += step
    return chunks


def _truncate_contexts(contexts: List[str], max_tokens: int) -> List[str]:
    """
    Truncate combined contexts to a total token budget by evenly cutting each chunk.
    """
    if not contexts or max_tokens is None or max_tokens <= 0:
        return contexts

    if tiktoken is None:
        # Fallback: approximate with words if tiktoken is unavailable.
        per_chunk_words = max(1, max_tokens // len(contexts))
        return [" ".join(text.split()[:per_chunk_words]) for text in contexts]

    encoding = tiktoken.get_encoding("o200k_base")
    per_chunk = max(1, max_tokens // len(contexts))
    truncated = []
    for text in contexts:
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:per_chunk]
        truncated.append(encoding.decode(truncated_tokens))
    return truncated


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

    def invoke(self, query: str, allowed_sources: set[str] | None = None) -> tuple[List[RetrievedDoc], Dict]:
        if not query:
            return [], {}
        
        debug = {}
        bm25_query = query
        if self.enable_query_expansion:
            bm25_query, qe_debug = expand_query_for_bm25(
                user_query=query,
                abbrev_map_path=CONFIG.get("abbrev_map_path"),
                max_expansions=self.max_query_expansions
            )
            debug["query_expansions"] = qe_debug

        try:
            parsed = self.parser.parse(bm25_query)
        except Exception:
            parsed = self.parser.parse(re.sub(r"[^\w\s]", " ", str(bm25_query)))
        
        source_filter = None
        if allowed_sources:
            normalized_sources = {
                _normalize_doc_label(src)
                for src in allowed_sources
                if src and str(src).strip()
            }

            if normalized_sources:
                terms = [
                    Term("source_norm", src) for src in normalized_sources
                ]
                source_filter = Or(terms) if len(terms) > 1 else terms[0] 

        with self.index.searcher() as searcher:
            hits = searcher.search(parsed, limit=self.k, filter=source_filter)
            docs = [
                RetrievedDoc(
                    page_content=hit.get("content", ""),
                    metadata={"source_file": hit.get("source_file", "")},
                )
                for hit in hits
            ]

        debug["bm25_query_used"] = bm25_query
        debug["num_hits"] = len(docs)
        return docs, debug

def load_and_build_retriever(pdf_dir, _generator_client=None):
    retriever = _load_and_build_retriever(pdf_dir)
    if retriever is None:
        raise RuntimeError(
            f"Failed to initialize retriever. Check PDF directory and index configuration."
        )
    
    # wrap retriever.invoke with multi-hop query rewrite
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



def _load_and_build_retriever(pdf_dir):
    index_dir = CONFIG.get("whoosh_index_dir", "whoosh_pdf_index")
    rebuild = CONFIG.get("rebuild_index", True)

    # Fast path: reuse existing index without touching PDFs when rebuild is False.
    if not rebuild and exists_in(index_dir):
        print(f"[WHOOSH] reusing PDF index at {index_dir}; set REBUILD_SPARSE_INDEX=1 to rebuild.")
        return WhooshBM25Retriever(index_dir, int(CONFIG.get("top_k", 5) or 5))

    all_chunks = []
    if not os.path.exists(pdf_dir):
        print("PDF Directory not found!")
        return None

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs.")
    
    for filename in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            docs = loader.load()
            for doc in docs:
                source_file = filename
                page_num = doc.metadata.get("page")
                cleaned = clean_pdf_text(doc.page_content or "")
                for chunk_idx, ch in enumerate(
                    chunk_text_fixed(
                        cleaned,
                        chunk_size_chars=_get_int_config("chunk_size_pdf", CONFIG["chunk_size"]),
                        overlap_chars=_get_int_config("chunk_overlap_pdf", CONFIG["overlap"]),
                    )
                ):
                    text = ch["text"]
                    if not text.strip():
                        continue
                    meta = {
                        "source_file": source_file,
                        "page": page_num,
                        "chunk_id": f"{_normalize_doc_label(source_file)}-p{page_num if page_num is not None else 'n'}-{chunk_idx}",
                        "char_start": ch["char_start"],
                        "char_end": ch["char_end"],
                    }
                    all_chunks.append(
                        SimpleDocument(
                            page_content=text,
                            metadata=meta,
                        )
                    )
        except Exception:
            pass
            
    if not all_chunks:
        return None

    if rebuild and os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)
        print(f"[WHOOSH] (re)building PDF index at {index_dir} with {len(all_chunks)} chunks")
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            source_file=ID(stored=True),
            source_norm=ID(stored=True),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        )
        index = create_in(index_dir, schema)
        writer = index.writer()
        for idx, chunk in enumerate(all_chunks):
            src = str(chunk.metadata.get("source_file", ""))
            writer.add_document(
                doc_id=str(idx),
                source_file=src,
                source_norm=_normalize_doc_label(src),
                content=chunk.page_content,
            )
        writer.commit()
    else:
        index = open_dir(index_dir)
    top_k = CONFIG.get("top_k", 5)
    try:
        k_val = int(top_k)
    except (TypeError, ValueError):
        k_val = 5
    return WhooshBM25Retriever(index_dir, k_val)


def _normalize_doc_label(name):
    """
    Normalize document identifiers for robust comparison.
    """
    if not name:
        return ""
    value = os.path.basename(str(name).strip()).lower()
    return value[:-4] if value.endswith(".pdf") else value


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
    url = url.replace("_", " ")
    return url.strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    return []


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
# 6. FRAMES / WIKIPEDIA HELPERS
# ==========================================
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


def load_frames_dataset():
    print("Loading FRAMES benchmark...")
    try:
        return load_dataset(CONFIG["frames_dataset"], split=CONFIG["frames_split"])
    except Exception as exc:
        print(f"Failed to load FRAMES dataset: {exc}")
        return []


def build_questions(ds, limit: Optional[int]) -> List[Dict]:
    sampled = ds
    if limit and limit > 0 and len(ds) > limit:
        sampled = ds.select(range(limit))

    questions = []
    for row in sampled:
        prompt = row.get("Prompt", "")
        answer = row.get("Answer", "")

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

        titles = [_url_to_title(link) for link in links]
        titles = [t for t in titles if t]

        if not prompt or not links or not titles:
            continue

        questions.append(
            {
                "query": prompt,
                "answer": answer,
                "target_links": list(links),
                "target_titles": titles,
            }
        )
    print(f"Prepared {len(questions)} questions (limit={limit or 'all'}).")
    return questions


def fetch_wikipedia_articles(links: List[str], titles: List[str]) -> Dict[str, str]:
    """
    Read ONLY from wiki_cache_dir/index.json mapping (link/title -> filename).
    No API calls or cache writes.
    """
    cache_index = _load_wiki_cache_index()
    if not cache_index:
        logging.warning("Wiki cache index missing/empty; no articles loaded.")
        return {}

    articles: Dict[str, str] = {}
    missing = 0

    for link in links:
        text = _read_cached_wiki_by_key(link, cache_index)
        if not text:
            missing += 1
            continue
        title = _url_to_title(link) or link
        articles[title] = text

    # Fallback lookup using titles (helps when index keys are titles)
    for title in titles:
        if title in articles:
            continue
        text = _read_cached_wiki_by_key(title, cache_index)
        if text:
            articles[title] = text

    if missing:
        logging.warning(
            "Cache mode only: %d/%d links missing from cache (no API fetch will occur).",
            missing, len(links)
        )

    print(f"Loaded {len(articles)} articles from cache (wiki_cache_dir).")
    return articles


def build_documents(articles: Dict[str, str]) -> List[Dict]:
    chunk_size = _get_int_config("chunk_size_wiki", CONFIG["chunk_size"])
    overlap = _get_int_config("chunk_overlap_wiki", CONFIG["overlap"])
    documents = []
    for title, text in tqdm(articles.items(), desc="Chunking articles", unit="article"):
        clean_text = _clean_text(text)
        for idx, chunk in enumerate(
            chunk_text_fixed(
                clean_text,
                chunk_size_chars=chunk_size,
                overlap_chars=overlap,
            )
        ):
            documents.append(
                {
                    "doc_id": f"{_normalize_title(title)}-{idx}",
                    "source_file": title,
                    "content": chunk["text"],
                    "section": "",
                    "page": "",
                    "chunk_id": f"{_normalize_title(title)}-{idx}",
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                }
            )
    print(f"Prepared {len(documents)} chunks.")
    return documents


def build_whoosh_index_frames(documents: List[Dict]) -> Optional["WhooshBM25Retriever"]:
    if not documents:
        return None

    index_dir = CONFIG.get("whoosh_index_dir_frames", "whoosh_wiki_index")
    rebuild = CONFIG.get("rebuild_index", True)

    if exists_in(index_dir) and rebuild is False:
        print(f"[WHOOSH] reusing FRAMES index at {index_dir}; set REBUILD_SPARSE_INDEX=1 to rebuild.")
        return WhooshBM25Retriever(index_dir, _get_int_config("top_k", 10))

    if os.path.exists(index_dir):
        for root, dirs, files in os.walk(index_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.makedirs(index_dir, exist_ok=True)
    print(f"[WHOOSH] (re)building FRAMES index at {index_dir} with {len(documents)} chunks")
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        source_file=ID(stored=True),
        source_norm=ID(stored=True),
        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
    )
    index = create_in(index_dir, schema)
    writer = index.writer(limitmb=CONFIG.get("whoosh_limit_mb", 1024))
    for doc in documents:
        src = str(doc.get("source_file", ""))
        writer.add_document(
            doc_id=doc["doc_id"],
            source_file=src,
            source_norm=_normalize_doc_label(src),
            content=doc["content"]
        )
    writer.commit()

    print(f"Whoosh docs indexed: {index.doc_count()}")
    return WhooshBM25Retriever(index_dir, _get_int_config("top_k", 10))


def _compute_retriever_metrics_titles(target_titles: List[str], retrieved_titles: List[str]):
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

        print(f"Processing Q{index+1}...")
        
        # Retrieve
        docs, debug_info = retriever.invoke(q)
        ctx_list = [d.page_content for d in docs]
        found_files = [d.metadata.get("source_file", "") for d in docs]

        max_ctx_tokens = _get_int_config("llm_max_tokens", 0)
        ctx_list = _truncate_contexts(ctx_list, max_ctx_tokens)
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
            "debug": json.dumps(debug_info)
        })

    return results


def run_frames_experiment(
    questions: List[Dict],
    retriever,
    generator_client,
    article_texts: Optional[Dict[str, str]] = None,
):
    if not questions:
        return []

    results = []
    for row in tqdm(questions, desc="FRAMES questions", unit="q"):
        query = row["query"]
        target_titles = row["target_titles"]
        debug_info = {}
        try:
            res = retriever.invoke(query)
            if isinstance(res, tuple) and len(res) == 2:
                docs, debug_info = res
            else:
                docs = res
        except Exception as exc:
            logging.warning("Retriever failed for query '%s': %s", query, exc)
            results.append(
                {
                    "question": query,
                    "answer": f"Retriever Error: {exc}",
                    "contexts": [],
                    "ground_truth": row.get("answer", ""),
                    "target_pdf": "; ".join(target_titles),
                    "found_files": [],
                    "retriever_precision": 0.0,
                    "retriever_recall": 0.0,
                    "retriever_f1": 0.0,
                    "debug": "{}",
                }
            )
            continue

        # de-dupe by source_file/title
        deduped: List[RetrievedDoc] = []
        seen_sources = set()
        for doc in docs:
            src = doc.metadata.get("source_file", "")
            if src and src in seen_sources:
                continue
            deduped.append(doc)
            if src:
                seen_sources.add(src)

        contexts: List[str] = []
        for d in deduped:
            title = d.metadata.get("source_file", "")
            if article_texts and title and title in article_texts:
                contexts.append(article_texts[title])
                continue
            contexts.append(d.page_content)

        max_ctx_tokens = _get_int_config("llm_max_tokens", 0)
        contexts = _truncate_contexts(contexts, max_ctx_tokens)

        found_titles = [d.metadata.get("source_file", "") for d in deduped]
        precision, recall, f1 = _compute_retriever_metrics_titles(target_titles, found_titles)

        ans = generate_answer(generator_client, "\n\n".join(contexts), query)

        results.append(
            {
                "question": query,
                "answer": ans,
                "contexts": contexts,
                "ground_truth": row.get("answer", ""),
                "target_pdf": "; ".join(target_titles),
                "found_files": found_titles,
                "retriever_precision": precision,
                "retriever_recall": recall,
                "retriever_f1": f1,
                "debug": json.dumps(debug_info),
            }
        )

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


def run_pdf_workflow(models):
    retriever = load_and_build_retriever(CONFIG["pdf_dir"], _generator_client=models["generator_client"])
    if not retriever:
        return []

    cache_path = CONFIG.get("llm_cache_path")
    experiment_data = None
    if CONFIG.get("reuse_llm_outputs"):
        experiment_data = load_cached_experiment(cache_path)

    if experiment_data is None:
        experiment_data = run_experiment(CONFIG["excel_path"], retriever, models["generator_client"])
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
    questions_all = build_questions(ds, limit=None)
    if not questions_all:
        print("No questions loaded; exiting.")
        return []

    qrange = _parse_question_range(CONFIG.get("question_range"))
    questions_eval = _apply_question_range(questions_all, qrange)

    index_dir = CONFIG.get("whoosh_index_dir_frames", "whoosh_wiki_index")
    rebuild_index = CONFIG.get("rebuild_index", True)
    index_exists = exists_in(index_dir)

    articles: Dict[str, str] = {}

    if rebuild_index or not index_exists:
        # Collect unique links + titles for article fetch / cache lookup
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

        articles = fetch_wikipedia_articles(unique_links, unique_titles)
        documents = build_documents(articles)
        bm25_retriever = build_whoosh_index_frames(documents)
    else:
        bm25_retriever = WhooshBM25Retriever(index_dir, _get_int_config("top_k", 10))
        print(f"Reusing existing Whoosh index at {index_dir}; set REBUILD_SPARSE_INDEX=1 to rebuild.")

    if not bm25_retriever:
        print("Failed to build or load BM25 index; exiting.")
        return []

    # wrap retriever.invoke with multi-hop query rewrite (frames path)
    if int(CONFIG.get("enable_hopping", 0)):
        base_invoke = bm25_retriever.invoke
        model_name = CONFIG["azure_gen_deployment"] if USE_AZURE else CONFIG["ollama_model"]
        bm25_retriever.invoke = make_hopping_invoke(
            base_invoke,
            generator_client=models["generator_client"],
            model_name=model_name,
            max_hops=int(CONFIG.get("hop_max_hops", 2)),
            evidence_max_docs=int(CONFIG.get("hop_evidence_docs", 8)),
            evidence_max_chars=int(CONFIG.get("hop_evidence_chars", 1200)),
            query_max_tokens=int(CONFIG.get("hop_query_max_tokens", 40)),
        )

    generator_client = models["generator_client"]
    return run_frames_experiment(
        questions_eval,
        bm25_retriever,
        generator_client=generator_client,
        article_texts=articles if articles else None,
    )

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    log_config_overview()
    models = get_models()
    doc_mode = str(CONFIG.get("doc_mode", "pdf")).strip().lower()
    print(f"Document mode: {doc_mode.upper()}")

    if doc_mode == "frames":
        experiment_data = run_frames_workflow(models)
    else:
        experiment_data = run_pdf_workflow(models)

    if not experiment_data:
        exit()

    # 3. Prepare Dataset
    print("Preparing Ragas Dataset...")
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
    print("Running Evaluation...")
    scores = evaluate(
        dataset=ragas_ds,
        metrics=ragas_metrics,
        run_config=ragas_run_config,
        batch_size=CONFIG["ragas_batch_size"]
    )

    # 5. Save
    df_out = scores.to_pandas()
    retriever_precisions = []
    retriever_recalls = []
    retriever_f1s = []
    bm25_debug_infos = []
    for record in experiment_data:
        prec, rec, f1 = _ensure_retriever_metrics(record)
        retriever_precisions.append(prec)
        retriever_recalls.append(rec)
        retriever_f1s.append(f1)
        bm25_debug_infos.append(record.get("debug", "{}"))
    df_out['retriever_precision'] = retriever_precisions
    df_out['retriever_recall'] = retriever_recalls
    df_out['retriever_f1'] = retriever_f1s
    df_out['target_pdf'] = [x['target_pdf'] for x in experiment_data]
    df_out['found_files'] = [str(x['found_files']) for x in experiment_data]
    df_out['debug'] = bm25_debug_infos

    mode_tag = doc_mode.lower()
    azure_tag = "azure" if USE_AZURE else "local"
    qrange = _parse_question_range(CONFIG.get("question_range"))
    if qrange:
        qrange_tag = f"q{qrange[0]}-{qrange[1]}"
    else:
        qrange_tag = "qall"
    base_name = f"Results_BM25_{mode_tag}_{azure_tag}_{qrange_tag}"
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
