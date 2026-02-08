from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple


_WORD_RE = re.compile(r"\b[\w/]+\b", re.UNICODE)
_NUM_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s+", re.UNICODE)

_UNIT_HINT_TOKENS = {
    "v", "vac", "vdc", "a", "ma", "w", "kw",
    "lm", "lx", "mcd", "deg", "°",
}

def _safe_ascii(s: str) -> str:
    # Make query safe for Whoosh parsing
    return re.sub(r"[^\w\s/]", " ", s, flags=re.UNICODE)

def _normalize_spaces(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()

def _tokenize_lower(q: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(q)]

def query_has_number(q: str) -> bool:
    return bool(q and _NUM_RE.search(q))

def query_has_unit_hint(q: str) -> bool:
    if not q:
        return False
    ql = q.lower()
    toks = set(_tokenize_lower(_safe_ascii(q)))
    if toks.intersection(_UNIT_HINT_TOKENS):
        return True
    return bool(re.search(r"\b\d+\s*(v|vac|vdc|a|ma|w|kw|lm|lx|mcd)\b", ql))

def is_unit_or_symbol_key(key: str) -> bool:
    if not key:
        return False
    k = key.lower()
    return any(ch in k for ch in ["/", "²", "³", "ω", "φ", "°"])

def _contains_phrase(haystack_lower: str, phrase_lower: str) -> bool:
    return bool(re.search(r"\b" + re.escape(phrase_lower) + r"\b", haystack_lower))

def _phrase_or_term(x: str) -> str:
    x = x.strip()
    if " " in x:
        return f"\"{x}\""
    return x

def load_abbrev_map(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k.lower(): v.lower() for k, v in data.items()}

def _build_reverse_map(abbrev_map: Dict[str, str]) -> Dict[str, str]:
    rev = {}
    for k, v in abbrev_map.items():
        rev.setdefault(v, k)
    return rev


def expand_query_for_bm25(
    user_query: str,
    abbrev_map_path: str,
    max_expansions: int = 10,
) -> Tuple[str, dict]:
    q_raw = user_query or ""
    q_clean = _normalize_spaces(_safe_ascii(q_raw))
    q_lower = q_clean.lower()
    tokens = set(_tokenize_lower(q_clean))
    abbrev_map = load_abbrev_map(abbrev_map_path)

    has_num = query_has_number(q_raw)
    has_unit_hint = query_has_unit_hint(q_raw)
    allow_unit_expansion = has_num and has_unit_hint

    rev_map = _build_reverse_map(abbrev_map)
    expansions = []

    # abbr -> full
    for abbr, full in abbrev_map.items():
        if is_unit_or_symbol_key(abbr) and not allow_unit_expansion:
            continue
        if (" " in abbr and _contains_phrase(q_lower, abbr)) or (abbr in tokens):
            expansions.append((abbr, full))

    # full -> abbr
    for full, abbr in rev_map.items():
        if _contains_phrase(q_lower, full):
            if is_unit_or_symbol_key(abbr) and not allow_unit_expansion:
                continue
            expansions.append((full, abbr))

    # dedup + cap
    seen = set()
    uniq = []
    for t, e in expansions:
        if (t, e) not in seen:
            uniq.append((t, e))
            seen.add((t, e))
        if len(uniq) >= max_expansions:
            break

    base_clause = f"({q_clean})" if q_clean else ""
    or_clauses = [
        f"({_phrase_or_term(t)} OR {_phrase_or_term(e)})"
        for t, e in uniq
    ]

    if base_clause and or_clauses:
        bm25_q = base_clause + " AND " + " AND ".join(or_clauses)
    else:
        bm25_q = base_clause or q_clean

    debug = {
        "original": q_raw,
        "bm25_query": bm25_q,
        "expansions": uniq,
        "allow_unit_expansion": allow_unit_expansion,
    }
    return bm25_q, debug



# =========================
# Dense QE helper
# =========================

_STOPWORDS = {
    # light stoplist for "question scaffolding" (keep conservative)
    "which", "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "be", "being", "been",
    "the", "a", "an", "and", "or", "as", "well", "between", "within", "into",
    "in", "on", "at", "for", "to", "of", "from", "by", "with", "without",
    "this", "that", "these", "those",
    "ensures", "ensure", "ensuring",
}

def _strip_question_scaffolding(q: str, max_tokens: int = 48) -> str:
    """
    Keep this conservative: remove common function words and question boilerplate,
    but preserve content terms. Limit length to avoid overly long embedding inputs.
    """
    toks = _tokenize_lower(q)
    kept = [t for t in toks if t not in _STOPWORDS]
    kept = kept[:max_tokens]
    return _normalize_spaces(" ".join(kept))

def _replace_phrases_case_insensitive(text: str, phrase: str, repl: str) -> str:
    """
    Replace whole-phrase occurrences with repl, case-insensitive.
    Uses word boundaries to avoid partial hits.
    """
    if not phrase:
        return text
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)

def _should_expand_key_for_dense(key: str, allow_unit_expansion: bool) -> bool:
    # Dense: be stricter about unit/symbol-like keys
    if is_unit_or_symbol_key(key) and not allow_unit_expansion:
        return False
    return True

def expand_query_for_dense(
    user_query: str,
    abbrev_map_path: str,
    max_tokens: int = 48,
    append_abbrev_parenthetical: bool = True,
) -> Tuple[str, dict]:
    """
    Dense-friendly QE:
      - safe-clean + normalize whitespace
      - expand abbreviations to full forms (preferred)
      - optionally append '(ABBR)' once for full-forms found in the query
      - apply unit/symbol expansions only if numeric+unit context
      - optionally strip question boilerplate to keep the embedding query compact
    Returns: (dense_query, debug)

    NOTE: This function intentionally avoids AND/OR/quotes lists.
    """
    q_raw = user_query or ""
    q_clean = _normalize_spaces(_safe_ascii(q_raw))
    q_lower = q_clean.lower()

    abbrev_map = load_abbrev_map(abbrev_map_path)
    rev_map = _build_reverse_map(abbrev_map)

    has_num = query_has_number(q_raw)
    has_unit_hint = query_has_unit_hint(q_raw)
    allow_unit_expansion = has_num and has_unit_hint

    applied = []
    appended = []

    # 1) Replace abbreviations (and phrase keys) with full forms.
    #    Do longer keys first to avoid partial overlaps ("v dc" before "v").
    items = sorted(abbrev_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    dense = q_clean
    dense_lower = dense.lower()

    for abbr, full in items:
        abbr = abbr.strip().lower()
        full = full.strip().lower()
        if not abbr or not full:
            continue
        if not _should_expand_key_for_dense(abbr, allow_unit_expansion):
            continue

        # phrase key
        if " " in abbr:
            if _contains_phrase(dense_lower, abbr):
                dense = _replace_phrases_case_insensitive(dense, abbr, full)
                dense_lower = dense.lower()
                applied.append((abbr, full))
        else:
            # token key
            if re.search(r"\b" + re.escape(abbr) + r"\b", dense_lower):
                dense = _replace_phrases_case_insensitive(dense, abbr, full)
                dense_lower = dense.lower()
                applied.append((abbr, full))

    # 2) Optionally append one abbreviation in parentheses when user wrote the full form.
    #    This helps if documents predominantly use abbreviations.
    #    Do NOT turn into OR lists; keep it as a compact parenthetical.
    if append_abbrev_parenthetical:
        # Find full-forms present in the (now expanded) dense query
        # Append up to a small number to avoid bloating embeddings.
        max_parentheticals = 3
        count = 0
        for full, abbr in rev_map.items():
            full = full.strip().lower()
            abbr = abbr.strip().lower()
            if not full or not abbr:
                continue
            if not _should_expand_key_for_dense(abbr, allow_unit_expansion):
                continue

            if _contains_phrase(dense_lower, full):
                # Avoid duplicates if abbreviation already appears
                if re.search(r"\b" + re.escape(abbr) + r"\b", dense_lower):
                    continue
                # Append at end; simplest and usually best for embeddings
                dense = dense + f" ({abbr})"
                dense_lower = dense.lower()
                appended.append((full, abbr))
                count += 1
                if count >= max_parentheticals:
                    break

    dense = _normalize_spaces(dense)

    # 3) Compress query to content terms (optional but recommended for long questions)
    dense_compact = _strip_question_scaffolding(dense, max_tokens=max_tokens)
    dense_out = dense_compact if dense_compact else dense

    debug = {
        "original": q_raw,
        "clean": q_clean,
        "allow_unit_expansion": allow_unit_expansion,
        "replaced_abbrev_to_full": applied,
        "appended_full_to_abbrev": appended,
        "dense_query": dense_out,
    }
    return dense_out, debug
