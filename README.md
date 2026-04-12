# Master Thesis: RAG Retrieval Experiments

This repository contains the experimental code and result files for a master
thesis on retrieval-augmented question answering. It compares sparse BM25
retrieval and hybrid sparse+dense retrieval on two document settings:

- Automotive standards PDFs stored in `raw_data/pdfs`
- FRAMES / Wikipedia-style question answering data

The pipeline retrieves context, generates short answers with either Azure
OpenAI or a local Ollama-compatible model, evaluates the answers with Ragas,
and writes per-run CSV/XLSX result files.

## Repository Layout

```text
.
+-- system_bm25_combined.py       # BM25-only retrieval and evaluation pipeline
+-- system_hybrid_combined.py     # Hybrid BM25 + dense FAISS retrieval pipeline
+-- query_expansion_helper.py     # Abbreviation-aware query expansion helpers
+-- hopping.py                    # Multi-hop query rewriting utilities
+-- wilcoxon_rank_test.py         # Wilcoxon signed-rank test helper for result comparison
+-- abbreviations.json            # Domain abbreviation map used by query expansion
+-- requirements.txt              # Python dependencies
+-- raw_data/
|   +-- dataset/
|   |   +-- Questions_Answer.xlsx # PDF QA evaluation questions and ground truth
|   |   +-- frames.tsv
|   +-- pdfs/                     # Automotive standards PDF corpus
+-- results/                      # Intermediate and summary result workbooks
+-- Final Results/                # Final thesis result workbooks
+-- archive/                      # Older experiment scripts kept for reference
```

Generated indexes and caches such as `whoosh_*`, `faiss_*`, `wiki_data`, and
`experiment_cache.json` may be created when running experiments.

## Setup

Create and activate a Python virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The code can run in two model modes:

- **Azure OpenAI mode**: enabled when `ENDPOINT` is set in `.env`
- **Local Ollama-compatible mode**: used when `ENDPOINT` is not set

Create a `.env` file in the repository root. Do not commit real secrets.

```env
# Azure OpenAI mode
API_KEY=your_api_key
ENDPOINT=https://your-resource.openai.azure.com/
API_VERSION=2024-xx-xx
MODEL_NAME=your_generation_and_judge_deployment

# Shared experiment controls
DOC_MODE=pdf
QUESTION_RANGE=1-10
REBUILD_INDEX=1

# Optional retrieval controls
RETRIEVER_TOP_K=5
SPARSE_TOP_K=40
DENSE_TOP_K=40
HYBRID_TOP_K=5
RRF_TOP_K=60

# Optional feature toggles
BM25_ENABLE_QUERY_EXPANSION=0
DENSE_ENABLE_QUERY_EXPANSION=0
ENABLE_HOPPING=0
ENABLE_RERANKER=1

# Local mode defaults can be changed if needed
OLLAMA_EMBED_MODEL=mxbai-embed-large:latest
```

For local mode, make sure Ollama or another OpenAI-compatible local endpoint is
available at `http://localhost:11434/v1`, and that the models referenced in the
scripts are available.

## Running Experiments

Run the BM25 pipeline on the PDF dataset:

```powershell
$env:DOC_MODE = "pdf"
$env:QUESTION_RANGE = "1-10"
python system_bm25_combined.py
```

Run the hybrid pipeline on the PDF dataset:

```powershell
$env:DOC_MODE = "pdf"
$env:QUESTION_RANGE = "1-10"
python system_hybrid_combined.py
```

Run the BM25 pipeline on FRAMES:

```powershell
$env:DOC_MODE = "frames"
$env:QUESTION_RANGE = "1-50"
python system_bm25_combined.py
```

Run the hybrid pipeline on FRAMES:

```powershell
$env:DOC_MODE = "frames"
$env:QUESTION_RANGE = "1-50"
python system_hybrid_combined.py
```

`QUESTION_RANGE` is optional. It uses 1-based inclusive indexing, for example
`1-10` or `25`. If it is omitted, the scripts run all available questions.

## Experiment Options

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `DOC_MODE` | `pdf` for automotive standards PDFs or `frames` for FRAMES/Wikipedia |
| `QUESTION_RANGE` | Optional 1-based question slice such as `1-10` |
| `REBUILD_INDEX` / `REBUILD_SPARSE_INDEX` | Rebuild Whoosh sparse indexes |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Default chunking controls |
| `CHUNK_SIZE_PDF`, `CHUNK_OVERLAP_PDF` | PDF-specific chunking controls |
| `CHUNK_SIZE_WIKI`, `CHUNK_OVERLAP_WIKI` | FRAMES/Wikipedia chunking controls |
| `BM25_ENABLE_QUERY_EXPANSION` | Enable BM25 abbreviation expansion |
| `ENABLE_HOPPING` | Enable multi-hop query rewriting |

Ragas timeout, worker, retry, and batch-size settings are configured inside the
`CONFIG` dictionary in each experiment script.

Hybrid-specific variables:

| Variable | Purpose |
| --- | --- |
| `DENSE_TOP_K` | Number of dense FAISS results before fusion |
| `HYBRID_TOP_K` | Final number of hybrid results used as context |
| `RRF_TOP_K` | Reciprocal-rank-fusion candidate depth |
| `REBUILD_DENSE_INDEX` | Rebuild the dense PDF FAISS index |
| `REBUILD_DENSE_FRAMES_INDEX` | Rebuild the dense FRAMES FAISS index |
| `DENSE_ENABLE_QUERY_EXPANSION` | Enable dense-query abbreviation expansion |
| `ENABLE_RERANKER` | Enable cross-encoder reranking in the hybrid pipeline |
| `CROSS_ENCODER_MODEL_NAME` | Optional reranker model override |

## Outputs

Each run writes a CSV and an XLSX file in the repository root using this naming
pattern:

```text
Results_BM25_<doc_mode>_<azure|local>_<qrange>.csv
Results_BM25_<doc_mode>_<azure|local>_<qrange>.xlsx
Results_Hybrid_<doc_mode>_<azure|local>_<qrange>.csv
Results_Hybrid_<doc_mode>_<azure|local>_<qrange>.xlsx
```

The output includes Ragas metrics, retriever precision/recall/F1, target
documents, retrieved files, debug information, and, for the hybrid pipeline,
retrieval and generation timing columns.

Existing experiment workbooks are stored under `results/` and `Final Results/`.

## Statistical Comparison

`wilcoxon_rank_test.py` compares one metric column across two result workbooks
with a Wilcoxon signed-rank test. Edit the constants at the top of the file:

```python
FILEPATH_1 = r"path\to\first.xlsx"
FILEPATH_2 = r"path\to\second.xlsx"
COLUMN_NAME = "retriever_f1"
ALPHA = 0.05
ALTERNATIVE = "less"  # "two-sided", "greater", or "less"
```

Then run:

```powershell
python wilcoxon_rank_test.py
```

## Notes

- The PDF workflow expects `raw_data/dataset/Questions_Answer.xlsx` and the
  referenced PDFs in `raw_data/pdfs`.
- The FRAMES workflow uses the configured dataset name, defaulting to
  `google/frames-benchmark`, and may use cached Wikipedia article text from
  `wiki_data`.
- Some dependencies download models or datasets on first use. Run small
  `QUESTION_RANGE` values first to validate configuration before starting long
  experiments.
- `.env` is ignored by Git and should remain local because it can contain API
  keys.
