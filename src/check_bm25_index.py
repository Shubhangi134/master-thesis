from whoosh import index
import os

INDEX_DIR = "../processed_data/bm25_index"

if not os.path.exists(INDEX_DIR):
    print("Index folder not found:", INDEX_DIR)
else:
    try:
        ix = index.open_dir(INDEX_DIR)
    except Exception as e:
        print("Failed to open index:", e)
    else:
        try:
            with ix.searcher() as s:
                print("doc_count_all():", s.doc_count_all())
                # print first 3 documents' chunk_ids
                it = s.all_stored_fields()
                i = 0
                for doc in it:
                    print("sample stored doc:", {k: doc.get(k) for k in ("chunk_id","pdf_file")})
                    i += 1
                    if i >= 3:
                        break
        except Exception as e:
            print("Error while reading index:", e)
