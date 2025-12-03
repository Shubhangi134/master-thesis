import json
import re
from hashlib import sha256
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shutil
import pdfplumber
from tqdm import tqdm

RAW_PDF_DIR = Path("../raw_data/pdfs")
PROCESSED_DATA_DIR = Path("../processed_data")

RAW_TEXT_DIR = PROCESSED_DATA_DIR / "raw_text"
CLEAN_TEXT_DIR = PROCESSED_DATA_DIR / "clean_text"
CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"
META_DIR = PROCESSED_DATA_DIR / "metadata"
PROCESSED_INDEX = PROCESSED_DATA_DIR / "processed_index.json"

for d in (RAW_TEXT_DIR, CLEAN_TEXT_DIR, CHUNKS_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 300
OVERLAP = 50
WORKERS = max(1, cpu_count() - 1)
FORCE = False  # set True in code to force reprocessing

# Manual filename -> region mapping
PDF_REGION_MAP = {
    "Automotive_SPICE_PAM_v40.pdf": "International",
    "ISO_26262_1_2018.pdf": "International",
    "ISO_26262_12_2018.pdf": "International",
    "ISO_16750_1_2023.pdf": "International",
    "ISO_7637.pdf": "International",
    "IS_IEC_61508_1998.pdf": "International",
    "astm_D4814_20A.pdf": "USA",
    "AIS_021.pdf": "India",
    "AIS_022.pdf": "India",
    "AIS_012.pdf": "India",
    "AIS_015.pdf": "India",
    "AIS_016.pdf": "India",
    "AIS_038.pdf": "India",
    "AIS_060.pdf": "India",
    "AIS_127.pdf": "India",
    "AIS_189.pdf": "India",
    "IS15140_2018.pdf": "India",
    "IS15139_2002.pdf": "India",
    "SAE_J3016_202104.pdf": "USA",
}

# Patterns for page-number detection
_PAGE_NUM_PATTERNS = [
    re.compile(r'^\s*page\s*\d+\s*$', re.I),
    re.compile(r'^\s*\(?\d+\s*/\s*\d+\)?\s*$'),
    re.compile(r'^\s*\(?\d+\)?\s*$'),
    re.compile(r'^\s*\d+\s*[-–]\s*\d+\s*$'),
    re.compile(r'^\s*\d+\s*$'),
]


def load_processed_index():
    if PROCESSED_INDEX.exists():
        try:
            return json.loads(PROCESSED_INDEX.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_processed_index(idx):
    PROCESSED_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_file_hash(path, block_size=65536):
    h = sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def _looks_like_page_number(s):
    if not s:
        return False
    s = s.strip()
    for p in _PAGE_NUM_PATTERNS:
        if p.match(s):
            return True
    return False


def _frequent_strings(cands, min_fraction=0.35, min_occurrences=2):
    freq = {}
    n = len(cands) or 1
    for s in cands:
        s = s.strip()
        if not s:
            continue
        freq[s] = freq.get(s, 0) + 1
    chosen = set()
    threshold = max(min_occurrences, int(n * min_fraction))
    for s, count in freq.items():
        if count >= threshold:
            chosen.add(s)
    return chosen


def remove_repeated_headers_footers(page_texts):
    first_lines = []
    last_lines = []
    for t in page_texts:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        first_lines.append(lines[0] if lines else "")
        last_lines.append(lines[-1] if lines else "")
    headers = _frequent_strings(first_lines)
    footers = _frequent_strings(last_lines)
    cleaned = []
    for t in page_texts:
        lines = [ln for ln in t.splitlines()]
        if lines and lines[0].strip() in headers:
            lines = lines[1:]
        if lines and lines[-1].strip() in footers:
            lines = lines[:-1]
        cleaned.append("\n".join(lines))
    removed = {"headers": list(headers), "footers": list(footers)}
    return cleaned, removed


def extract_text_from_pdf(pdf_path):
    page_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            page_texts.append(t)
    joined = "\n\n".join(page_texts)
    return joined, page_texts


def clean_text(text, page_texts=None):
    diagnostics = {"header_footer_removed": {}, "page_number_lines_removed": 0}
    if page_texts:
        cleaned_pages, removed = remove_repeated_headers_footers(page_texts)
    else:
        cleaned_pages, removed = [text], {"headers": [], "footers": []}

    cleaned_pages2 = []
    pn_count = 0
    for p in cleaned_pages:
        lines = [ln for ln in p.splitlines()]
        out_lines = []
        for ln in lines:
            if _looks_like_page_number(ln.strip()):
                pn_count += 1
                continue
            out_lines.append(ln)
        page_text = "\n".join(out_lines)
        page_text = re.sub(r'\s+\n', '\n', page_text)
        page_text = re.sub(r'\n\s+', '\n', page_text)
        page_text = re.sub(r'\r\n?', '\n', page_text)
        page_text = re.sub(r'[ \t]{2,}', ' ', page_text)
        page_text = re.sub(r'\n{3,}', '\n\n', page_text)
        cleaned_pages2.append(page_text.strip())

    diagnostics["header_footer_removed"] = removed
    diagnostics["page_number_lines_removed"] = pn_count
    joined = "\n\n".join([p for p in cleaned_pages2 if p.strip()])
    joined = re.sub(r' {2,}', ' ', joined)
    return joined.strip(), diagnostics


def infer_region_from_text(cleaned_text):
    t = cleaned_text.lower()
    if "iso" in t or "international organization for standardization" in t:
        return "International"
    if "iec" in t or "international electrotechnical commission" in t:
        return "International"
    if "sae international" in t or "society of automotive engineers" in t:
        return "USA"
    if "astm" in t:
        return "USA"
    if "bureau of indian standards" in t or "bis" in t or "automotive industry standard" in t or "ais " in t:
        return "India"
    return "Unknown"


def get_region(pdf_path, cleaned_text):
    if pdf_path.name in PDF_REGION_MAP:
        return PDF_REGION_MAP[pdf_path.name]
    if pdf_path.stem in PDF_REGION_MAP:
        return PDF_REGION_MAP[pdf_path.stem]
    return infer_region_from_text(cleaned_text)


def deterministic_chunk_id(file_hash, chunk_index):
    return sha256(f"{file_hash}|{chunk_index}".encode("utf-8")).hexdigest()


def chunk_text_with_metadata(text, pdf_filename, file_hash, chunk_size, overlap, region):
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        cid = deterministic_chunk_id(file_hash, chunk_index)
        chunks.append({
            "chunk_id": cid,
            "pdf_file": pdf_filename,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "region": region
        })
        chunk_index += 1
        start += chunk_size - overlap
    return chunks


def process_pdf_worker(pdf_path_str, file_hash, chunk_size, overlap, force):
    pdf_path = Path(pdf_path_str)
    stem = pdf_path.stem
    pdf_name = pdf_path.name

    chunks_file = CHUNKS_DIR / f"{stem}_chunks.json"
    meta_file = META_DIR / f"{stem}_meta.json"
    raw_text_file = RAW_TEXT_DIR / f"{stem}.txt"
    clean_text_file = CLEAN_TEXT_DIR / f"{stem}.txt"

    try:
        # Extract
        raw_text, page_texts = extract_text_from_pdf(pdf_path)
        raw_text_file.write_text(raw_text, encoding="utf-8")

        # Clean
        cleaned_text, diagnostics = clean_text(raw_text, page_texts=page_texts)
        clean_text_file.write_text(cleaned_text, encoding="utf-8")

        # Region
        region = get_region(pdf_path, cleaned_text)

        # If no selectable text, write meta and return
        if not cleaned_text.strip():
            meta = {
                "pdf_file": pdf_name,
                "file_hash": file_hash,
                "pages": len(page_texts),
                "chunks": 0,
                "region": region,
                "note": "no selectable text (likely scanned)"
            }
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            return {
                "status": "no_text",
                "pdf_file": pdf_name,
                "file_hash": file_hash,
                "pages": len(page_texts),
                "chunks": 0,
                "region": region,
                "meta_path": str(meta_file.resolve()),
                "diagnostics": diagnostics
            }

        # Chunk
        chunks = chunk_text_with_metadata(cleaned_text, pdf_name, file_hash, chunk_size, overlap, region)
        chunks_file.write_text(json.dumps({
            "chunks": chunks,
            "diagnostics": diagnostics,
            "region": region
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Meta
        meta = {
            "pdf_file": pdf_name,
            "file_hash": file_hash,
            "pages": len(page_texts),
            "chunks": len(chunks),
            "region": region
        }
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "status": "processed",
            "pdf_file": pdf_name,
            "file_hash": file_hash,
            "pages": len(page_texts),
            "chunks": len(chunks),
            "region": region,
            "chunks_path": str(chunks_file.resolve()),
            "meta_path": str(meta_file.resolve()),
            "diagnostics": diagnostics
        }
    except Exception as e:
        return {
            "status": "error",
            "pdf_file": pdf_name,
            "error": str(e)
        }


def main():
    pdf_files = sorted([p for p in RAW_PDF_DIR.glob("*.pdf")])
    print(f"Found {len(pdf_files)} PDFs to process in {RAW_PDF_DIR}")

    idx = load_processed_index()

    to_process = []   # workers tasks: (pdf_path_str, file_hash, chunk_size, overlap, force)
    reused = []
    for p in pdf_files:
        file_hash = compute_file_hash(p)
        if file_hash in idx and not FORCE:
            # check if chunks file exists
            existing_chunks_path = idx[file_hash].get("chunks_path")
            if existing_chunks_path:
                existing_chunks_path = Path(existing_chunks_path)
                if existing_chunks_path.exists():
                    # reuse chunks: copy and update pdf_file field
                    with open(existing_chunks_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    source_chunks = data.get("chunks") if isinstance(data, dict) and "chunks" in data else (data if isinstance(data, list) else [])
                    updated_chunks = []
                    for c in source_chunks:
                        nc = c.copy()
                        nc["pdf_file"] = p.name
                        updated_chunks.append(nc)
                    new_chunks_file = CHUNKS_DIR / f"{p.stem}_chunks.json"
                    new_chunks_file.write_text(json.dumps({
                        "chunks": updated_chunks,
                        "diagnostics": idx[file_hash].get("diagnostics", {}),
                        "region": idx[file_hash].get("region", "Unknown")
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    # write metadata for this filename
                    meta_file = META_DIR / f"{p.stem}_meta.json"
                    meta_file.write_text(json.dumps({
                        "pdf_file": p.name,
                        "file_hash": file_hash,
                        "pages": idx[file_hash].get("pages", 0),
                        "chunks": len(updated_chunks),
                        "region": idx[file_hash].get("region", "Unknown"),
                        "reused_from": Path(idx[file_hash].get("chunks_path", "")).name if idx[file_hash].get("chunks_path") else None
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    # update filenames list inside idx (master only)
                    names = set(idx[file_hash].get("filenames", []))
                    names.add(p.name)
                    idx[file_hash]["filenames"] = list(names)
                    reused.append(p.name)
                    continue  # skip dispatching to worker

        # otherwise schedule for processing
        to_process.append((str(p), file_hash, CHUNK_SIZE, OVERLAP, FORCE))


    results = []
    if to_process:
        if WORKERS > 1:
            with Pool(processes=WORKERS) as pool:
                for r in tqdm(pool.starmap(process_pdf_worker, to_process), total=len(to_process)):
                    results.append(r)
        else:
            for task in tqdm(to_process):
                results.append(process_pdf_worker(*task))
    # Update index with new results
    for r in results:
        status = r.get("status")
        if status == "processed":
            fh = r["file_hash"]
            idx[fh] = {
                "chunks_path": r.get("chunks_path"),
                "meta_path": r.get("meta_path"),
                "filenames": [r.get("pdf_file")],
                "pages": r.get("pages"),
                "region": r.get("region"),
                "diagnostics": r.get("diagnostics", {})
            }
        elif status == "no_text":
            fh = r["file_hash"]
            idx[fh] = {
                "chunks_path": None,
                "meta_path": r.get("meta_path"),
                "filenames": [r.get("pdf_file")],
                "pages": r.get("pages"),
                "region": r.get("region"),
                "diagnostics": r.get("diagnostics", {})
            }
        elif status == "error":
            # You may want to log errors; leave them out of index
            print(f"Error processing {r.get('pdf_file')}: {r.get('error')}")

    # Persist index once (master only)
    save_processed_index(idx)

    # Summary
    summary = {
        "processed": [r["pdf_file"] for r in results if r.get("status") == "processed"],
        "no_text": [r["pdf_file"] for r in results if r.get("status") == "no_text"],
        "reused": reused,
        "errors": [r for r in results if r.get("status") == "error"]
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
