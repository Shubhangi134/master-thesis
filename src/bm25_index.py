import os
import json
import shutil
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import RegexTokenizer, LowercaseFilter

BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "processed_data"))
CHUNKS_DIR = os.path.join(BASE, "chunks")
INDEX_DIR = os.path.join(BASE, "bm25_index")

# Set to True to rebuild the index from scratch
REBUILD = False

# tokenizer for technical documents
tokenizer = RegexTokenizer() | LowercaseFilter()

# UPDATED SCHEMA (added source, page_from, page_to)
schema = Schema(
    chunk_id=ID(stored=True, unique=True),
    pdf_file=STORED,
    region=STORED,
    source=STORED,
    page_from=STORED,
    page_to=STORED,
    content=TEXT(stored=True, analyzer=tokenizer)
)

# INDEX CREATION / OPENING
def ensure_index(rebuild=False):
    if rebuild and os.path.exists(INDEX_DIR):
        print("Rebuild requested — removing existing index...")
        shutil.rmtree(INDEX_DIR)

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        ix = index.create_in(INDEX_DIR, schema)
        print("Created new index.")
        return ix

    try:
        ix = index.open_dir(INDEX_DIR)
        print("Opened existing index.")
        return ix
    except Exception:
        try:
            shutil.rmtree(INDEX_DIR)
        except Exception:
            pass
        os.makedirs(INDEX_DIR)
        ix = index.create_in(INDEX_DIR, schema)
        print("Recreated index after error.")
        return ix


# LOAD CHUNKS
def load_chunks():
    if not os.path.exists(CHUNKS_DIR):
        raise Exception("Chunks directory not found: " + CHUNKS_DIR)

    for fname in sorted(os.listdir(CHUNKS_DIR)):
        if not fname.endswith("_chunks.json"):
            continue

        path = os.path.join(CHUNKS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print("Warning: failed to read", fname, ":", e)
            continue

        for c in data.get("chunks", []):
            yield c


# INDEXING
def index_all(ix):
    writer = ix.writer(limitmb=512)
    total = 0
    updated = 0

    for c in load_chunks():
        total += 1
        text = (c.get("text") or "").strip()
        if not text:
            continue

        try:
            writer.update_document(
                chunk_id=c["chunk_id"],
                pdf_file=c.get("pdf_file", ""),
                region=c.get("region", ""),
                source=c.get("source", ""),               # NEW
                page_from=c.get("page_from"),            # NEW
                page_to=c.get("page_to"),                # NEW
                content=text
            )
            updated += 1
        except Exception as e:
            print("Failed to index chunk:", c.get("chunk_id"), str(e))

    try:
        writer.commit()
    except Exception as e:
        writer.cancel()
        print("Commit failed; writer rolled back:", e)
        raise

    return total, updated


def main():
    ix = ensure_index(rebuild=REBUILD)

    print("Indexing chunks...")
    total, updated = index_all(ix)

    print("Chunks seen:", total)
    print("Chunks added/updated:", updated)
    print("Done.")


if __name__ == "__main__":
    main()
