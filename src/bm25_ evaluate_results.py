import os, sys, json, math, csv, re
from collections import Counter, defaultdict
from statistics import mean
from datetime import datetime

# CONFIG (edit if needed)
LLM_RESULTS = "bm25_llm_results.jsonl"
RETRIEVAL_RESULTS = "bm25_retrieval_results.jsonl"   # optional
GROUND_TRUTH = "ground_truth.json"                 # optional: qid -> [chunk_id,...]
QA_XLSX = "Questions_Answer.xlsx"                  # fallback to get reference answers if not in LLM results
OUT_SUMMARY = "eval_summary.json"
OUT_PER_QUERY_JSONL = "eval_per_query.jsonl"
OUT_PER_QUERY_CSV = "eval_per_query.csv"
OUT_RETRIEVAL_CSV = "retrieval_metrics.csv"

# Tokenization for F1/EM (simple)
_tok_re = re.compile(r"[A-Za-z0-9\-_']+")

def normalize_text(t):
    if t is None: return ""
    return re.sub(r"\s+", " ", str(t).strip().lower())

def tokens(text):
    return _tok_re.findall(normalize_text(text))

def f1_score(pred, gold):
    p = tokens(pred)
    g = tokens(gold)
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0: return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)

def exact_match(pred, gold):
    return 1 if normalize_text(pred) == normalize_text(gold) else 0

def dcg(rels):
    # binary rels list ordered by rank
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def compute_retrieval_metrics_for_query(retrieved_chunk_ids, gold_set, Klist=[1,3,5,10]):
    """Given ordered retrieved list of chunk_ids and gold set, compute Precision@K, Recall@K, RR, nDCG@K for Klist"""
    metrics = {}
    for K in Klist:
        topk = retrieved_chunk_ids[:K]
        hits = sum(1 for cid in topk if cid in gold_set)
        precision = hits / K
        recall = hits / (len(gold_set) if gold_set else 1)
        metrics[f"prec@{K}"] = precision
        metrics[f"recall@{K}"] = recall
        # nDCG binary
        rels = [1 if cid in gold_set else 0 for cid in topk]
        idcg = dcg(sorted(rels, reverse=True))
        ndcg = dcg(rels) / idcg if idcg > 0 else 0.0
        metrics[f"ndcg@{K}"] = ndcg
    # reciprocal rank
    rr = 0.0
    for i, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in gold_set:
            rr = 1.0 / i
            break
    metrics["rr"] = rr
    return metrics

def load_jsonl(path):
    arr = []
    if not os.path.exists(path): 
        return arr
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip(): continue
            arr.append(json.loads(line))
    return arr

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)

def main():
    # load llm results
    if not os.path.exists(LLM_RESULTS):
        print(f"ERROR: {LLM_RESULTS} not found. Run the end-to-end evaluator first.")
        sys.exit(1)
    llm_rows = load_jsonl(LLM_RESULTS)
    print(f"Loaded {len(llm_rows)} LLM result rows from {LLM_RESULTS}.")

    # load retrieval results if present
    retrieval_rows = load_jsonl(RETRIEVAL_RESULTS) if os.path.exists(RETRIEVAL_RESULTS) else []
    if retrieval_rows:
        print(f"Loaded {len(retrieval_rows)} retrieval rows from {RETRIEVAL_RESULTS}.")

    # load ground truth if provided
    gold = {}
    if os.path.exists(GROUND_TRUTH):
        with open(GROUND_TRUTH, "r", encoding="utf-8") as fh:
            gold = json.load(fh)
        print(f"Loaded ground truth for {len(gold)} queries from {GROUND_TRUTH}.")
    else:
        print("No ground_truth.json found — retrieval-level metrics & groundedness will be partial/optional.")

    per_query = []
    # accumulators
    f1_list = []
    em_list = []
    called_list = []
    refusal_list = []
    model_times = []
    retrieval_times = []  # if present in jsons
    grounded_flags = []

    # For retrieval metrics aggregation
    Klist = [1,3,5,10]
    retrieval_metrics_acc = {f"prec@{k}": [] for k in Klist}
    retrieval_metrics_acc.update({f"recall@{k}": [] for k in Klist})
    retrieval_metrics_acc.update({f"ndcg@{k}": [] for k in Klist})
    retrieval_rrs = []

    # Build a quick map from qid to retrieval row if available
    retrieval_map = {r["qid"]: r for r in retrieval_rows} if retrieval_rows else {}

    for row in llm_rows:
        qid = str(row.get("qid"))
        question = row.get("question","")
        ref = row.get("reference_answer", "") or row.get("gold_answer","")
        # prefer ref from LLM result if exists; else must find from QA file (not implemented here)
        model_answer = row.get("model_answer_raw") or row.get("gpt_answer") or row.get("model_answer") or ""
        called = bool(row.get("called_llm", False)) or (model_answer and not model_answer.startswith("NO_ANSWER") and not model_answer.startswith("[LLM_ERROR]"))
        called_list.append(1 if called else 0)
        is_refusal = str(model_answer).strip().startswith(("NO_ANSWER_IN_DOCUMENTS", "I don't know", "NO_ANSWER"))
        refusal_list.append(1 if is_refusal else 0)
        f1 = f1_score(model_answer, ref)
        em = exact_match(model_answer, ref)
        f1_list.append(f1)
        em_list.append(em)
        t = row.get("model_runtime_s") or row.get("runtime_s") or None
        if t is not None:
            try:
                model_times.append(float(t))
            except:
                pass

        # retrieval info
        retrieved = row.get("retrieval_top_k") or row.get("retrieved") or []
        retrieved_ids = [r.get("chunk_id") for r in retrieved if r.get("chunk_id")]
        top_score = row.get("top_score", 0.0)
        second_score = row.get("second_score", 0.0)
        conf = row.get("confident_by_retriever", row.get("confident", False))

        # groundedness if gold and model cited chunk ids present
        cited = row.get("model_cited_chunk_ids") or []
        grounded = False
        if gold and qid in gold:
            gold_set = set(gold.get(qid, []))
            grounded = bool(set(cited) & gold_set)
            grounded_flags.append(1 if grounded else 0)
        else:
            # if no gold present but we have retrieval rows mapping to pdf-level,
            # we can approximate: consider grounded if any cited id appears in retrieved top-k
            grounded = bool(set(cited) & set(retrieved_ids))
            grounded_flags.append(1 if grounded else 0)

        per = {
            "qid": qid,
            "question": question,
            "reference_answer": ref,
            "model_answer": model_answer,
            "called_llm": int(called),
            "model_refusal": int(is_refusal),
            "f1": f1,
            "exact_match": em,
            "top_score": top_score,
            "second_score": second_score,
            "confident_by_retriever": int(conf),
            "retrieved_chunk_ids": retrieved_ids,
            "model_cited_chunk_ids": cited,
            "grounded": int(grounded),
            "model_runtime_s": t
        }
        per_query.append(per)

        # if ground truth available, compute retrieval metrics per query using retrieval_map or retrieved_ids
        if gold and qid in gold:
            gold_set = set(gold.get(qid, []))
            # if we have retrieved ids in a separate retrieval_results map, use that for ranking; else use retrieved_ids
            ranking_ids = retrieved_ids
            q_metrics = compute_retrieval_metrics_for_query(ranking_ids, gold_set, Klist=Klist)
            for k in Klist:
                retrieval_metrics_acc[f"prec@{k}"].append(q_metrics[f"prec@{k}"])
                retrieval_metrics_acc[f"recall@{k}"].append(q_metrics[f"recall@{k}"])
                retrieval_metrics_acc[f"ndcg@{k}"].append(q_metrics[f"ndcg@{k}"])
            retrieval_rrs.append(q_metrics["rr"])

    # Aggregation
    n = len(per_query)
    mean_f1 = mean(f1_list) if f1_list else 0.0
    mean_em = mean(em_list) if em_list else 0.0
    call_rate = sum(called_list)/n if n else 0.0
    refusal_rate = sum(refusal_list)/n if n else 0.0
    mean_model_time = mean(model_times) if model_times else None
    grounded_rate = sum(grounded_flags)/n if grounded_flags else None

    summary = {
        "date": datetime.utcnow().isoformat() + "Z",
        "n_queries": n,
        "mean_f1": mean_f1,
        "mean_exact_match": mean_em,
        "llm_call_rate": call_rate,
        "model_refusal_rate": refusal_rate,
        "mean_model_runtime_s": mean_model_time,
        "grounded_rate": grounded_rate
    }

    # retrieval aggregates if computed
    if gold and retrieval_metrics_acc:
        for k in Klist:
            keyp = f"prec@{k}"
            keyr = f"recall@{k}"
            keyn = f"ndcg@{k}"
            summary[f"mean_{keyp}"] = mean(retrieval_metrics_acc[keyp]) if retrieval_metrics_acc[keyp] else None
            summary[f"mean_{keyr}"] = mean(retrieval_metrics_acc[keyr]) if retrieval_metrics_acc[keyr] else None
            summary[f"mean_{keyn}"] = mean(retrieval_metrics_acc[keyn]) if retrieval_metrics_acc[keyn] else None
        summary["mean_rr"] = mean(retrieval_rrs) if retrieval_rrs else None

    # write per-query JSONL and CSV
    with open(OUT_PER_QUERY_JSONL, "w", encoding="utf-8") as fh:
        for p in per_query:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    # CSV header
    csv_fields = ["qid","f1","exact_match","called_llm","model_refusal","top_score","confident_by_retriever","grounded","model_runtime_s"]
    with open(OUT_PER_QUERY_CSV, "w", newline='', encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for p in per_query:
            row = {k: p.get(k) for k in csv_fields}
            writer.writerow(row)

    # retrieval metrics CSV if ground truth
    if gold and retrieval_metrics_acc:
        with open(OUT_RETRIEVAL_CSV, "w", newline='', encoding="utf-8") as fh:
            fieldnames = ["metric", "K", "mean_value"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for k in Klist:
                writer.writerow({"metric": "precision", "K": k, "mean_value": summary.get(f"mean_prec@{k}")})
                writer.writerow({"metric": "recall", "K": k, "mean_value": summary.get(f"mean_recall@{k}")})
                writer.writerow({"metric": "ndcg", "K": k, "mean_value": summary.get(f"mean_ndcg@{k}")})
            if summary.get("mean_rr") is not None:
                writer.writerow({"metric": "mrr", "K": "", "mean_value": summary["mean_rr"]})

    # write summary json
    save_json(summary, OUT_SUMMARY)
    print("Evaluation complete.")
    print("Summary:", json.dumps(summary, indent=2))
    print(f"Wrote per-query JSONL: {OUT_PER_QUERY_JSONL}")
    print(f"Wrote per-query CSV: {OUT_PER_QUERY_CSV}")
    if gold:
        print(f"Wrote retrieval metrics CSV: {OUT_RETRIEVAL_CSV}")

if __name__ == "__main__":
    main()
