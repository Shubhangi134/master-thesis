import pdfplumber
import re
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


RAW_PDF_DIR = Path("../raw_data/pdfs")
PROCESSED_DATA_DIR = Path("../processed_data")
RAW_TEXT_DIR = PROCESSED_DATA_DIR / "raw_text"
CLEAN_TEXT_DIR = PROCESSED_DATA_DIR / "clean_text"
CHUNKS_DIR = PROCESSED_DATA_DIR / "chunks"

RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 300  # words per chunk
OVERLAP = 50      # words overlapping between chunks

PDF_REGION_MAP = {
    # TODO: Add actual mappings
}


def extract_text_from_pdf(pdf_path):
    text_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_pages.append(text)
    return "\n".join(text_pages)

def clean_text(text):
    # TODO: check if additional cleaning is needed
    text = re.sub(r'\n+', '\n', text)      # remove multiple line breaks
    text = re.sub(r' +', ' ', text)        # remove extra spaces
    text = text.strip()
    return text

def chunk_text_with_metadata(text, pdf_file, chunk_size=CHUNK_SIZE, overlap=OVERLAP, region="Unknown"):
    words = text.split()
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "pdf_file": pdf_file.name,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "region": region
        })
        chunk_index += 1
        start += chunk_size - overlap
    return chunks

def process_pdf(pdf_file):
    try:
        region = PDF_REGION_MAP.get(pdf_file.name, "Unknown")

        # Step 1: Extract text
        raw_text = extract_text_from_pdf(pdf_file)
        raw_text_file = RAW_TEXT_DIR / f"{pdf_file.stem}.txt"
        with open(raw_text_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Step 2: Clean text
        cleaned_text = clean_text(raw_text)
        clean_text_file = CLEAN_TEXT_DIR / f"{pdf_file.stem}.txt"
        with open(clean_text_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Step 3: Chunk with metadata
        chunks = chunk_text_with_metadata(cleaned_text, pdf_file, region=region)
        chunks_file = CHUNKS_DIR / f"{pdf_file.stem}_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return pdf_file.name, len(chunks)
    except Exception as e:
        return pdf_file.name, f"Error: {e}"

def main():
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process.")

    # Use multiprocessing
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_pdf, pdf_files), total=len(pdf_files)))

    # Print summary
    print("\nProcessing summary:")
    for pdf_name, status in results:
        print(f"{pdf_name}: {status} chunks")

if __name__ == "__main__":
    main()
