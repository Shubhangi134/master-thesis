# inspect_index_distribution.py
from whoosh import index
from collections import Counter
IDX = "../processed_data/bm25_index"

ix = index.open_dir(IDX)
cntr = Counter()
with ix.searcher() as s:
    for doc in s.all_stored_fields():
        pdf = doc.get("pdf_file") or "UNKNOWN"
        cntr[pdf] += 1

# print summary
total_docs = sum(cntr.values())
unique_pdfs = len([p for p in cntr.keys() if p and p != "UNKNOWN"])
print("Total docs in index:", total_docs)
print("Distinct pdf_file values:", unique_pdfs)
print("Top 10 pdfs by chunk count:")
for pdf, count in cntr.most_common(10):
    print(f"  {pdf}: {count}")
