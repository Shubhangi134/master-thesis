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

# ---------- CONFIG (minimal required changes) ----------
QA_XLSX = "Questions_Answer.xlsx"
OUT_JSONL = "bm25_llm_results.jsonl"

TOP_K = 8                   # ↑ increased from 5 → improves recall
MODEL = "gpt-4.1"
TEMPERATURE = 0.0
MAX_TOKENS = 800            # ↑ increased from 512 → longer automotive answers
ABS_THRESHOLD = 5.0         # ↑ major improvement → prevents bad LLM calls
RATIO_THRESHOLD = 3.0
USE_RATIO_CHECK = False

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

if run_query is None:
    print("ERROR: Could not find run_query().")
    sys.exit(1)

# ENV
ENDPOINT = os.getenv("ENDPOINT")
DEPLOYMENT = os.getenv("DEPLOYMENT")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_VERSION = os.getenv("API_VERSION")
MODEL_NAME_ENV = os.getenv("MODEL_NAME")
IS_AZURE = bool(ENDPOINT)

if IS_AZURE:
    if not (ENDPOINT, DEPLOYMENT, API_KEY, API_VERSION):
        print("Azure vars missing.")
        sys.exit(1)
else:
    if not API_KEY:
        print("API key missing.")
        sys.exit(1)

if MODEL_NAME_ENV:
    MODEL = MODEL_NAME_ENV

client = None
if IS_AZURE:
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=API_KEY,
    )
else:
    client = OpenAI(api_key=API_KEY)

# ----------------- NORMALIZATION -----------------

_tok_re = re.compile(r"[A-Za-z0-9\-_]+")

def normalize_text(t):
    if t is None:
        return ""
    return re.sub(r"\s+", " ", t.strip().lower())

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

# ------------- cite extraction -----------------
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

# ----------------- IMPROVED PROMPT -----------------

def build_strict_prompt(question, passages):
    """
    passages: list of dicts with keys: chunk_id, text
    """
    lines = []
    lines.append("You are a precise assistant. Use ONLY the passages below to answer the question.")
    lines.append("You MAY combine information across multiple passages if needed.")  # NEW
    lines.append("If the passages do not provide enough information, reply: NOT_ENOUGH_CONTEXT")  # NEW
    lines.append("")
    lines.append("When you answer, do these steps:")
    lines.append("1) Give a concise answer (1–3 sentences).")
    lines.append("2) On a new line, write: EVIDENCE: [used: chunk_123, chunk_456]")
    lines.append("3) On the next line, write: CONFIDENCE: HIGH or CONFIDENCE: LOW")
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    lines.append("")
    lines.append("Passages:")

    for i, p in enumerate(passages, start=1):
        lines.append(f"[{i}] (chunk_id: {p.get('chunk_id')})")
        txt = p.get("text") or ""
        lines.append(txt)
        lines.append("")

    lines.append("Now answer following the instructions above.")
    return "\n".join(lines)

# ---------------- LLM CALL ----------------------

def call_gpt(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, model_name=None):
    model_to_use = model_name or (DEPLOYMENT if IS_AZURE else MODEL)

    try:
        resp = client.chat.completions.create(
            model=model_to_use,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        return f"[LLM_ERROR] {e}", None

    try:
        text = resp.choices[0].message["content"]
    except:
        text = str(resp)

    try:
        total_tokens = resp.usage.total_tokens
    except:
        total_tokens = None

    return text, total_tokens

# ----------------- MAIN -------------------------

def main(call_all=False, top_k=TOP_K):
    if not os.path.exists(QA_XLSX):
        print(f"ERROR: QA file not found: {QA_XLSX}")
        sys.exit(1)

    df = pd.read_excel(QA_XLSX)
    cols = list(df.columns)

    qid_col = "qid" if "qid" in cols else cols[0]
    question_col = "question" if "question" in cols else cols[1]
    answer_col = "answer" if "answer" in cols else cols[2]

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
        qid = row[qid_col]
        question = row[question_col]
        ref_answer = row[answer_col]

        # Retrieval
        try:
            retrieval = run_query(question, k=top_k)
        except Exception as e:
            retrieval = {"results": [], "confident": False, "top_score": 0.0}

        results = retrieval.get("results", [])
        top_score = retrieval.get("top_score", 0.0)
        confident_flag = retrieval.get("confident", False)

        should_call = call_all or confident_flag

        # Build passages
        passages = [{
            "chunk_id": r.get("chunk_id"),
            "text": r.get("text") or ""
        } for r in results[:top_k]]

        if not should_call:
            model_answer = "NOT_ENOUGH_CONTEXT"
            model_tokens = 0
            model_runtime = 0.0
            num_refusals += 1
        else:
            prompt = build_strict_prompt(question, passages)
            start = time.time()
            model_answer, model_tokens = call_gpt(prompt)
            model_runtime = time.time() - start
            num_called += 1

        cited_chunk_ids = extract_cited_chunk_ids(model_answer)

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

    mean_f1 = total_f1 / n
    mean_em = total_em / n

    print("\n--- SUMMARY ---")
    print(f"Queries processed: {n}")
    print(f"LLM called for: {num_called} queries ({num_called/n:.2%})")
    print(f"Retriever refusals: {num_refusals} ({num_refusals/n:.2%})")
    print(f"Mean F1: {mean_f1:.4f}")
    print(f"Exact Match rate: {mean_em:.4f}")
    print(f"Results written to: {OUT_JSONL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--call-all", action="store_true")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    main(call_all=args.call_all, top_k=args.top_k)
