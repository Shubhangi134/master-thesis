import os
import json
import re
import pandas as pd
import warnings
import logging
import ftfy
import shutil
from dataclasses import dataclass
from typing import List, Dict

# --- 1. IMPORTS ---
# Native Azure Client for Generation (This works fine)
from openai import AzureOpenAI, AsyncAzureOpenAI
from openai import OpenAI as StandardOpenAI, AsyncOpenAI as StandardAsyncOpenAI

# UNIVERSAL WRAPPERS (The Fix)
# We will wrap LangChain objects for Ragas evaluation to avoid ImportErrors
from ragas.embeddings import LangchainEmbeddingsWrapper
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

# UNIVERSAL WRAPPERS (The Fix)
# Modern structured interfaces required by Ragas collections metrics
from ragas.embeddings import HuggingFaceEmbeddings, LangchainEmbeddingsWrapper

from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import SimpleParser

from dotenv import load_dotenv

load_dotenv(".env")

# Enable verbose Ragas logging for easier debugging
logging.basicConfig(level=logging.INFO)
logging.getLogger("ragas").setLevel(logging.DEBUG)

# Suppress Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
    "local_embed_model": "sentence-transformers/all-MiniLM-L6-v2",
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
    "top_k": int(os.getenv("RETRIEVER_TOP_K", 5)),
    "whoosh_index_dir": "whoosh_pdf_index",
    "rebuild_index": False,
    "generation_prompt": None,  # Optional custom prompt template using {context} and {question}

    # Critical: Ollama Base URL for OpenAI compatibility
    "ollama_base_url": "http://localhost:11434/v1", 
    "ollama_model": "mistral-large-3:675b-cloud", # Your specific model name
    "ollama_api_key": "ollama", # Dummy key required by client
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
        # This tricks Ragas into thinking it's using OpenAI, so it works!
        ragas_client = StandardAsyncOpenAI(
            base_url=CONFIG["ollama_base_url"],
            api_key=CONFIG["ollama_api_key"],
            timeout=180.0
        )
        
        ragas_judge = llm_factory(
            model=CONFIG["ollama_model"],
            client=ragas_client
        )
        
        # 3. Embeddings (Local HuggingFace - Keeps it fast/free)
        # We stick to Wrapper here because llm_factory doesn't handle HF embeddings
        embed_model_raw = HuggingFaceEmbeddings(model=CONFIG["local_embed_model"])
        ragas_embeddings = LangchainEmbeddingsWrapper(embed_model_raw)

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
        try:
            parsed = self.parser.parse(query)
        except Exception:
            parsed = self.parser.parse(re.sub(r"[^\w\s]", " ", str(query)))
        with self.index.searcher() as searcher:
            hits = searcher.search(parsed, limit=self.k)
            return [
                RetrievedDoc(
                    page_content=hit.get("content", ""),
                    metadata={"source_file": hit.get("source_file", "")},
                )
                for hit in hits
            ]


def load_and_build_retriever(pdf_dir):
    all_chunks = []
    if not os.path.exists(pdf_dir):
        print("PDF Directory not found!")
        return None

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDFs.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG["chunk_size"], chunk_overlap=CONFIG["overlap"])
    
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
                text = ftfy.fix_text(
                    text,
                    fix_encoding=True,
                    fix_entities=True
                )
                chunk.page_content = text
            all_chunks.extend(split_docs)
        except: pass
            
    if not all_chunks:
        return None

    index_dir = CONFIG.get("whoosh_index_dir", "whoosh_pdf_index")
    rebuild = CONFIG.get("rebuild_index", True)
    if rebuild and os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            source_file=TEXT(stored=True),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        )
        index = create_in(index_dir, schema)
        writer = index.writer()
        for idx, chunk in enumerate(all_chunks):
            writer.add_document(
                doc_id=str(idx),
                source_file=str(chunk.metadata.get("source_file", "")),
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


def _normalize_answer_text(value):
    """
    Normalize text for token comparison (lowercase, keep periods, strip other punctuation).
    """
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[^\w\s\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    models = get_models()
    retriever = load_and_build_retriever(CONFIG["pdf_dir"])
    if not retriever: exit()

    # 2. Run
    cache_path = CONFIG.get("llm_cache_path")
    experiment_data = None
    if CONFIG.get("reuse_llm_outputs"):
        experiment_data = load_cached_experiment(cache_path)

    if experiment_data is None:
        experiment_data = run_experiment(CONFIG["excel_path"], retriever, models["generator_client"])
        if not experiment_data: exit()
        if CONFIG.get("reuse_llm_outputs"):
            save_experiment_cache(experiment_data, cache_path)
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
        )
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
