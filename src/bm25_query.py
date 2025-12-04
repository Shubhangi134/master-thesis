import os
import textwrap
from whoosh import index
from whoosh.qparser import MultifieldParser, OrGroup

BASE = "../processed_data"
INDEX_DIR = os.path.join(BASE, "bm25_index")

TOP_K = 5               # number of chunks to return
ABS_THRESHOLD = 1.0     # absolute score threshold (tune this)
RATIO_THRESHOLD = 3.0   # optional: top_score should be at least this many times second_score to be confident
USE_RATIO_CHECK = False # set True to enable ratio check (optional)

def run_query(query_text, k=TOP_K, abs_threshold=ABS_THRESHOLD,
              use_ratio=USE_RATIO_CHECK, ratio_threshold=RATIO_THRESHOLD):
    """
    Returns a dict with:
      - 'results': list of result dicts
      - 'confident': bool
      - 'top_score': float
    """
    if not os.path.exists(INDEX_DIR):
        raise Exception("BM25 index not found: " + INDEX_DIR)

    ix = index.open_dir(INDEX_DIR)

    qp = MultifieldParser(["content"], schema=ix.schema, group=OrGroup)
    parsed = qp.parse(query_text)

    results_out = []
    with ix.searcher() as s:
        results = s.search(parsed, limit=k)
        for r in results:
            results_out.append({
                "score": r.score,
                "chunk_id": r["chunk_id"],
                "pdf_file": r["pdf_file"],
                "region": r.get("region", ""),
                "source": r.get("source", ""),        # NEW
                "page_from": r.get("page_from"),     # NEW
                "page_to": r.get("page_to"),         # NEW
                "text": r["content"]
            })

    # No hits
    if not results_out:
        return {"results": [], "confident": False, "top_score": 0.0}

    top_score = results_out[0]["score"]
    second_score = results_out[1]["score"] if len(results_out) > 1 else 0.0

    # Absolute threshold check
    if top_score < abs_threshold:
        return {"results": [], "confident": False, "top_score": top_score}

    # Optional ratio check
    if use_ratio:
        eps = 1e-9
        ratio = top_score / max(second_score, eps)
        if ratio < ratio_threshold:
            return {"results": [], "confident": False, "top_score": top_score}

    return {"results": results_out, "confident": True, "top_score": top_score}


def main():
    print("BM25 Query Tool (interactive).")
    print("Press ENTER on an empty line to exit.")
    print(f"ABS_THRESHOLD={ABS_THRESHOLD} | USE_RATIO_CHECK={USE_RATIO_CHECK} | RATIO_THRESHOLD={RATIO_THRESHOLD}")

    while True:
        q = input("\nEnter question: ").strip()
        if q == "":
            print("Exiting.")
            break

        out = run_query(q, k=TOP_K)

        if not out["confident"]:
            print(f"No confident answer found (top score {out['top_score']:.4f} < threshold {ABS_THRESHOLD}).")
            continue

        results = out["results"]
        print("\nTop Results:")
        print("=" * 100)
        for i, r in enumerate(results, start=1):
            print(f"[{i}] score={r['score']:.4f} | pdf={r['pdf_file']} | region={r['region']} | source={r['source']} | pages={r['page_from']}-{r['page_to']}")

            snippet = (r["text"] or "")[:600].replace("\n", " ")
            print(textwrap.fill(snippet, width=100))
            print("-" * 100)


if __name__ == "__main__":
    main()
