import os
import sys
import re
import time
import json
import argparse
from collections import Counter
import pandas as pd

try:
    from openai import AzureOpenAI, OpenAI
except Exception:
    AzureOpenAI = None
    OpenAI = None

# ---------- CONFIG (edit if you want) ----------
QA_XLSX = "Questions_Answer.xlsx"
OUT_JSONL = "bm25_llm_results.jsonl"
TOP_K = 5
MODEL = "gpt-4.1"           # fallback model name for non-Azure
TEMPERATURE = 0.0
MAX_TOKENS = 512
CALL_ALL_DEFAULT = False
RUN_QUERY_IMPORTS = [
    "bm25_query",
    "bm25_query_tool",
    "query_tool",
    "run_query_module",
    "bm25_query"
]

run_query = None
for modname in RUN_QUERY_IMPORTS:
    try:
        mod = __import__(modname)
        if hasattr(mod, "run_query"):
            run_query = getattr(mod, "run_query")
            break
    except Exception:
        pass

# As a fallback, try to import from the current file's sibling if the user placed run_query there.
if run_query is None:
    # try importing by searching files in current directory for run_query
    this_dir = os.path.dirname(os.path.abspath(__file__))
    for fname in os.listdir(this_dir):
        if not fname.endswith(".py") or fname == os.path.basename(__file__):
            continue
        module_name = fname[:-3]
        try:
            mod = __import__(module_name)
            if hasattr(mod, "run_query"):
                run_query = getattr(mod, "run_query")
                break
        except Exception:
            pass

if run_query is None:
    print("ERROR: Could not find `run_query` function in nearby modules.")
    print("Please place this script alongside the file that defines run_query or")
    print("adjust the import at the top of this script.")
    sys.exit(1)

ENDPOINT = os.getenv("ENDPOINT")  # Azure endpoint if present
DEPLOYMENT = os.getenv("DEPLOYMENT")  # Azure deployment name
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")  # either Azure or OpenAI key
API_VERSION = os.getenv("API_VERSION")  # Azure API version
MODEL_NAME_ENV = os.getenv("MODEL_NAME")  # optional override for model name

IS_AZURE = bool(ENDPOINT)

# Validate credentials
if IS_AZURE:
    if not (ENDPOINT and DEPLOYMENT and API_KEY and API_VERSION):
        print("ERROR: For Azure you must set ENDPOINT, DEPLOYMENT, API_KEY and API_VERSION environment variables.")
        sys.exit(1)
else:
    if not API_KEY:
        print("ERROR: OPENAI API key not found. Set either API_KEY or OPENAI_API_KEY (or set ENDPOINT for Azure).")
        sys.exit(1)

# If MODEL_NAME env var is provided, use it for non-Azure (or as a fallback)
if MODEL_NAME_ENV:
    MODEL = MODEL_NAME_ENV

# Create client(s)
client = None
if IS_AZURE:
    if AzureOpenAI is None:
        raise RuntimeError("AzureOpenAI client not available. Please install openai>=1.x that exposes AzureOpenAI / OpenAI classes.")
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    )
else:
    if OpenAI is None:
        raise RuntimeError("OpenAI client class not available. Please install openai>=1.x that exposes AzureOpenAI / OpenAI classes.")
    client = OpenAI(api_key=API_KEY)

# --- helpers for normalization and metrics ---
_tok_re = re.compile(r"[A-Za-z0-9\-_]+")

def normalize_text(t):
    if t is None:
        return ""
    t = re.sub(r"\s+", " ", t.strip().lower())
    return t

def tokens_for_eval(text):
    return [t for t in _tok_re.findall(normalize_text(text))]

def f1_score(pred, gold):
    p_tokens = tokens_for_eval(pred)
    g_tokens = tokens_for_eval(gold)
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_tokens)
    rec = num_same / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

def exact_match(pred, gold):
    return 1 if normalize_text(pred) == normalize_text(gold) else 0

# Extract chunk ids from model answer (if model follows the prompt and lists EVIDENCE)
_cite_re = re.compile(r"chunk[_\-\s]?id[:\s]*([A-Za-z0-9\-_\.]+)", flags=re.I)
_used_re = re.compile(r"\[used[:\s]*([^\]]+)\]", flags=re.I)
_chunk_token_re = re.compile(r"(chunk[_\-\d]+)", flags=re.I)

def extract_cited_chunk_ids(answer_text):
    if not answer_text:
        return []
    ids = set()
    for m in _cite_re.finditer(answer_text):
        ids.add(m.group(1).strip())
    for m in _used_re.finditer(answer_text):
        parts = re.split(r"[,;\s]+", m.group(1).strip())
        for p in parts:
            if p:
                ids.add(p.strip())
    for m in _chunk_token_re.finditer(answer_text):
        ids.add(m.group(1))
    return list(ids)

# Build strict prompt: forces refusal when no answer in passages
def build_strict_prompt(question, passages):
    """
    passages: list of dicts with keys: chunk_id, text
    """
    lines = []
    lines.append("You are a precise assistant. Use ONLY the passages below to answer the question.")
    lines.append("If the answer is not contained in the passages, reply EXACTLY: NO_ANSWER_IN_DOCUMENTS")
    lines.append("When you answer, do these three things (in this order):")
    lines.append("1) Give a short, concise answer (1-3 sentences).")
    lines.append("2) On a new line, write: EVIDENCE: [used: chunk_123, chunk_456] (list the chunk_id(s) you used).")
    lines.append("3) On the next line, write: CONFIDENCE: HIGH  or  CONFIDENCE: LOW")
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    lines.append("")
    lines.append("Passages:")
    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (chunk_id: {p.get('chunk_id')})")
        # include full chunk text (or truncated if extremely long)
        txt = p.get("text") or p.get("content") or ""
        lines.append(txt)
        lines.append("")
    lines.append("Now answer following the instructions above.")
    return "\n".join(lines)

# Call GPT using the instantiated client (supports AzureOpenAI and OpenAI)
def call_gpt(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model_name=None):
    model_to_use = model_name or (DEPLOYMENT if IS_AZURE else MODEL)
    # Build args
    kwargs = {
        "model": model_to_use,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # call client
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        try:
            if IS_AZURE:
                resp = client.create_chat_completion(model=model_to_use, messages=kwargs["messages"], max_tokens=max_tokens, temperature=temperature)
            else:
                resp = client.create_chat_completion(model=model_to_use, messages=kwargs["messages"], max_tokens=max_tokens, temperature=temperature)
        except Exception as e2:
            raise

    text = None
    total_tokens = None
    try:
        choices = resp.get("choices") if isinstance(resp, dict) else getattr(resp, "choices", None)
        if choices:
            first = choices[0]
            msg = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
            if msg:
                text = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            else:
                text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
    except Exception:
        text = None
    if not text:
        try:
            choices = resp.choices if hasattr(resp, "choices") else None
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                if msg:
                    text = getattr(msg, "content", None)
                else:
                    text = getattr(first, "text", None)
        except Exception:
            pass

    if text is None:
        # final fallback: stringified response
        text = str(resp)
    try:
        if isinstance(resp, dict):
            total_tokens = resp.get("usage", {}).get("total_tokens")
        else:
            usage = getattr(resp, "usage", None)
            if usage:
                total_tokens = getattr(usage, "total_tokens", None)
    except Exception:
        total_tokens = None

    return text, total_tokens

def main(call_all=False, top_k=TOP_K):
    if not os.path.exists(QA_XLSX):
        print(f"ERROR: QA file not found: {QA_XLSX}")
        sys.exit(1)

    df = pd.read_excel(QA_XLSX)
    cols = list(df.columns)
    # detect common column names
    qid_col = "qid" if "qid" in cols else cols[0]
    question_col = "question" if "question" in cols else (cols[1] if len(cols) > 1 else cols[0])
    answer_col = "answer" if "answer" in cols else (cols[2] if len(cols) > 2 else cols[1] if len(cols) > 1 else cols[0])

    # ensure strings
    df[qid_col] = df[qid_col].astype(str)
    df[question_col] = df[question_col].astype(str)
    df[answer_col] = df[answer_col].astype(str)

    outfh = open(OUT_JSONL, "w", encoding="utf-8")

    total_f1 = 0.0
    total_em = 0
    n = 0
    num_called = 0
    num_refusals = 0

    print(f"Running BM25+LLM evaluation on {len(df)} queries (TOP_K={top_k}) ...")
    for _, row in df.iterrows():
        n += 1
        qid = str(row[qid_col])
        question = row[question_col]
        ref_answer = row[answer_col]

        # Call your run_query (it handles index opening)
        try:
            retrieval = run_query(question, k=top_k)
        except Exception as e:
            print(f"Warning: run_query failed for qid {qid}: {e}")
            retrieval = {"results": [], "confident": False, "top_score": 0.0}

        results = retrieval.get("results", []) or []
        top_score = retrieval.get("top_score", 0.0)
        confident_flag = retrieval.get("confident", False)

        # Decide whether to call LLM
        should_call = call_all or confident_flag

        # If there are no results but call_all=True, still pass empty passages (model will refuse)
        passages = []
        for r in results[:top_k]:
            passages.append({
                "chunk_id": r.get("chunk_id"),
                "text": r.get("text") or r.get("content") or ""
            })

        if not should_call:
            # Do not call LLM; production-safe refusal
            model_answer = "NO_ANSWER_IN_DOCUMENTS"
            model_tokens = 0
            model_runtime = 0.0
            num_refusals += 1
        else:
            # Build strict prompt and call LLM
            prompt = build_strict_prompt(question, passages)
            start = time.time()
            try:
                model_answer, model_tokens = call_gpt(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model_name=(DEPLOYMENT if IS_AZURE else MODEL))
            except Exception as e:
                model_answer = f"[LLM_ERROR] {e}"
                model_tokens = None
            model_runtime = time.time() - start
            num_called += 1

        # Parse cited chunk ids (if any)
        cited_chunk_ids = extract_cited_chunk_ids(model_answer)

        # Evaluate against reference:
        f1 = f1_score(model_answer, ref_answer)
        em = exact_match(model_answer, ref_answer)

        total_f1 += f1
        total_em += em

        out = {
            "qid": qid,
            "question": question,
            "reference_answer": ref_answer,
            "retrieval_top_k": results,
            "top_score": top_score,
            "confident_by_retriever": confident_flag,
            "called_llm": should_call,
            "model_answer_raw": model_answer,
            "model_cited_chunk_ids": cited_chunk_ids,
            "model_tokens": model_tokens,
            "model_runtime_s": model_runtime,
            "f1": f1,
            "exact_match": em
        }
        outfh.write(json.dumps(out, ensure_ascii=False) + "\n")

    outfh.close()

    mean_f1 = total_f1 / n if n else 0.0
    mean_em = total_em / n if n else 0.0
    print("\n--- SUMMARY ---")
    print(f"Queries processed: {n}")
    print(f"LLM called for: {num_called} queries ({num_called/n:.2%})")
    print(f"Explicit retriever refusals (no LLM call): {num_refusals} ({num_refusals/n:.2%})")
    print(f"Mean F1: {mean_f1:.4f}")
    print(f"Exact Match rate: {mean_em:.4f}")
    print(f"Results written to: {OUT_JSONL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 + LLM end-to-end evaluation")
    parser.add_argument("--call-all", action="store_true", help="Call LLM for every query regardless of retriever confidence")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top-K contexts to send to LLM")
    args = parser.parse_args()
    main(call_all=args.call_all, top_k=args.top_k)
