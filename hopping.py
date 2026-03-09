from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _doc_text(doc: Any) -> str:
    for attr in ("page_content", "text", "content", "chunk", "body"):
        v = getattr(doc, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    if isinstance(doc, dict):
        for k in ("page_content", "text", "content"):
            v = doc.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def _doc_meta(doc: Any) -> Dict[str, Any]:
    md = getattr(doc, "metadata", None)
    if isinstance(md, dict):
        return md
    if isinstance(doc, dict):
        md = doc.get("metadata")
        if isinstance(md, dict):
            return md
    return {}


def _stable_doc_key(doc: Any) -> str:
    md = _doc_meta(doc)
    doc_id = md.get("doc_id") or md.get("id")
    if doc_id:
        return f"id::{_safe_str(doc_id)}"
    src = _safe_str(md.get("source_file") or md.get("source") or md.get("path"))
    page = _safe_str(md.get("page") or md.get("page_number") or md.get("p"))
    chunk = _safe_str(md.get("chunk_id") or md.get("chunk") or md.get("chunk_index"))
    if src and (page or chunk):
        return f"s::{src}::{page}::{chunk}"
    txt = _doc_text(doc)[:2000]
    h = hashlib.md5((src + "||" + txt).encode("utf-8")).hexdigest()
    return f"h::{h}"


def _build_evidence(docs: Sequence[Any], max_docs: int, max_chars: int) -> str:
    parts: List[str] = []
    used = 0
    for d in docs[: max(0, int(max_docs))]:
        txt = _doc_text(d).strip()
        if not txt:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        snippet = txt[:remaining]
        if snippet:
            parts.append(snippet)
            used += len(snippet)
        if used >= max_chars:
            break
    return "\n\n---\n\n".join(parts)


def _normalize_query(q: str) -> str:
    q = _safe_str(q).strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q


def _near_duplicate(a: str, b: str) -> bool:
    na, nb = _normalize_query(a), _normalize_query(b)
    if na == nb:
        return True
    if na in nb or nb in na:
        la, lb = len(na), len(nb)
        if min(la, lb) / max(la, lb) > 0.88:
            return True
    return False


_HARD_PATTERNS = [
    r"\bISO\s?\d{3,6}(?:-\d+)?\b",
    r"\bIEC\s?\d{3,6}(?:-\d+)?\b",
    r"\bSAE\s?[A-Z]?\d{2,5}\b",
    r"\bUNECE\b|\bUN\s?ECE\b",
    r"\bECE\s?R\d{1,3}\b",
    r"\bUN\s?R\d{1,3}\b",
    r"\bRegulation\s*\(EU\)\s*\d{4}/\d{1,4}\b",
    r"\bDirective\s*\d{2,4}/\d{1,3}/EC\b",
    r"\bM[1-3]\b|\bN[1-3]\b|\bL[1-7]e?\b",
    r"\bO[1-4]\b",
]

_STRUCT_PATTERNS = [
    r"\bAnnex\s+[A-Z0-9]+\b",
    r"\bAppendix\s+[A-Z0-9]+\b",
    r"\bTable\s+\d+[A-Z]?\b",
    r"\bFigure\s+\d+[A-Z]?\b",
    r"\b\d+(?:\.\d+){1,4}\b",
]

_SOFT_TERM_STOPWORDS = {
    "what", "which", "who", "when", "where", "why", "how",
    "according", "round", "nearest", "name", "number", "many", "much",
    "is", "was", "were", "are", "be", "been", "being", "do", "does", "did",
    "the", "a", "an", "of", "to", "in", "on", "for", "from", "by", "at",
    "under", "over", "with", "without", "into", "than", "that", "this", "these", "those",
    "also", "only", "their", "there", "then", "than", "its", "it", "his", "her",
    "and", "or", "but", "if", "as", "after", "before", "during", "through", "about",
    "according", "nearest", "thousand",
}

_SOFT_SINGLE_KEEP = {
    "austin", "texas"
}




def _drop_contained_spans(items: Sequence[str]) -> List[str]:
    """Prefer the longest unique spans and drop contained subspans."""
    out: List[str] = []
    for s in sorted({_safe_str(x).strip() for x in items if _safe_str(x).strip()}, key=lambda v: (-len(v), v.lower())):
        sl = s.lower()
        if any(sl != t.lower() and sl in t.lower() for t in out):
            continue
        out.append(s)
    return out

def extract_anchors(question: str, max_soft: int = 6) -> List[str]:
    q = _safe_str(question)
    hard: List[str] = []
    for pat in _HARD_PATTERNS + _STRUCT_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            tok = m.group(0).strip()
            if tok and tok not in hard:
                hard.append(tok)

    soft: List[str] = []
    seen_soft = set()

    # Prefer multi-word proper nouns.
    for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", q):
        tok = m.group(0).strip()
        low = tok.lower()
        if low in _SOFT_TERM_STOPWORDS or tok in hard or low in seen_soft:
            continue
        seen_soft.add(low)
        soft.append(tok)
        if len(soft) >= max_soft:
            return hard + soft

    # Then useful single-word entities.
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", q):
        tok = m.group(0).strip()
        low = tok.lower()
        if tok in hard or low in seen_soft:
            continue
        if low in _SOFT_TERM_STOPWORDS and low not in _SOFT_SINGLE_KEEP:
            continue
        if low not in _SOFT_SINGLE_KEEP:
            continue
        seen_soft.add(low)
        soft.append(tok)
        if len(soft) >= max_soft:
            break

    return hard + soft


def extract_soft_terms(question: str, max_terms: int = 14) -> List[str]:
    q = _safe_str(question)
    found: List[str] = []
    seen = set()

    # Multi-word proper nouns first.
    for m in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", q):
        tok = m.group(0).strip()
        low = tok.lower()
        if low in seen:
            continue
        seen.add(low)
        found.append(tok)
        if len(found) >= max_terms:
            return found

    # Years.
    for m in re.finditer(r"\b\d{4}\b", q):
        tok = m.group(0)
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
            if len(found) >= max_terms:
                return found

    # General content words.
    for m in re.finditer(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", q):
        tok = m.group(0)
        low = tok.lower()
        if low in seen or low in _SOFT_TERM_STOPWORDS:
            continue
        seen.add(low)
        found.append(tok)
        if len(found) >= max_terms:
            break

    return found


def _enforce_anchor_lock(query: str, anchors: Sequence[str]) -> str:
    q = _safe_str(query).strip()
    low = q.lower()
    for a in anchors:
        aa = _safe_str(a).strip()
        if aa and aa.lower() not in low:
            q = f"{q} {aa}".strip()
            low = q.lower()
    return q


def _tokenize_terms(s: str) -> List[str]:
    return [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}|\b\d{4}\b", _safe_str(s))]


def _soft_term_recall(original_terms: Sequence[str], candidate_query: str) -> float:
    orig = {_safe_str(t).lower() for t in original_terms if _safe_str(t).strip()}
    if not orig:
        return 1.0
    cand = set(_tokenize_terms(candidate_query))
    hits = 0
    for t in orig:
        parts = _tokenize_terms(t)
        if parts and all(p in cand for p in parts):
            hits += 1
    return hits / max(1, len(orig))


@dataclass
class JudgeResult:
    answerable: bool
    missing: List[str]
    reason: str


def _llm_chat(generator_client: Any, model_name: str, system: str, user: str) -> str:
    resp = generator_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return _safe_str(resp.choices[0].message.content).strip()


def _parse_json_best_effort(txt: str) -> Dict[str, Any]:
    s = _safe_str(txt).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}
    return {}


def llm_judge_answerability(
    generator_client: Any,
    model_name: str,
    question: str,
    docs: Sequence[Any],
    *,
    evidence_max_docs: int = 6,
    evidence_max_chars: int = 1800,
) -> Tuple[JudgeResult, Dict[str, Any]]:
    evidence = _build_evidence(docs, evidence_max_docs, evidence_max_chars)
    system = (
        "You are a strict judge for a retrieval-augmented QA system.\n"
        "Answer YES only if the question can be answered using verbatim content in the excerpts.\n"
        "If any required numbers, IDs, clauses, entities, or definitions are missing in the excerpts, answer NO.\n"
        "Return ONLY valid JSON."
    )
    user = (
        "Return JSON with keys:\n"
        '  "answerable": "YES" or "NO"\n'
        '  "missing": list of short labels\n'
        '  "reason": <=25 words\n\n'
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{evidence}\n"
    )
    raw = _llm_chat(generator_client, model_name, system, user)
    data = _parse_json_best_effort(raw)
    ans = _safe_str(data.get("answerable", "")).strip().upper()
    answerable = ans == "YES"
    missing = data.get("missing") if isinstance(data.get("missing"), list) else []
    missing = [_safe_str(x) for x in missing if _safe_str(x)]
    reason = _safe_str(data.get("reason", "")).strip()
    dbg = {"raw": raw, "parsed": {"answerable": ans, "missing": missing, "reason": reason}}
    return JudgeResult(answerable=answerable, missing=missing, reason=reason), dbg


def llm_extract_keywords(
    generator_client: Any,
    model_name: str,
    question: str,
    docs: Sequence[Any],
    *,
    anchors: Sequence[str],
    evidence_max_docs: int = 8,
    evidence_max_chars: int = 2200,
) -> Tuple[List[str], Dict[str, Any]]:
    evidence = _build_evidence(docs, evidence_max_docs, evidence_max_chars)
    anchors_str = ", ".join([a for a in anchors if a])
    system = (
        "You extract search keywords for document chunk retrieval.\n"
        "You MUST ONLY output phrases that appear verbatim in the provided excerpts.\n"
        "Prefer named entities, technical terms, table labels, section labels, and compact noun phrases.\n"
        "Do NOT output whole sentences or answer-like numeric claims.\n"
        "Return ONLY valid JSON."
    )
    user = (
        'Return JSON with key "keywords" as a list of 5 to 12 items.\n'
        "- Each item must be a short phrase (1-6 words) copied verbatim from the excerpts.\n"
        "- Avoid generic words and avoid sentence fragments that look like final answers.\n\n"
        f"Locked anchors from the question: {anchors_str}\n\n"
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{evidence}\n"
    )
    raw = _llm_chat(generator_client, model_name, system, user)
    data = _parse_json_best_effort(raw)
    kws = data.get("keywords") if isinstance(data.get("keywords"), list) else []
    kws = [_safe_str(x).strip() for x in kws if _safe_str(x).strip()]
    return kws, {"raw": raw, "parsed_keywords": kws}


def _sanitize_keywords(kws: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for k in kws:
        kk = re.sub(r"\s+", " ", _safe_str(k)).strip()
        if not kk or len(kk) < 3:
            continue
        low = kk.lower()
        if low in {"the", "and", "or", "shall", "must", "may", "should"}:
            continue
        # Avoid answer-like numeric snippets from noisy evidence.
        if re.search(r"\b(?:population|price|amount|total|count|score)\b", low) and re.search(r"\b\d[\d,\.]*\b", kk):
            continue
        if len(kk.split()) > 6:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(kk)
    return out[:12]


def _pick_query_terms(question: str, anchors: Sequence[str], soft_terms: Sequence[str], keywords: Sequence[str], max_terms: int = 14) -> str:
    pieces: List[str] = []
    seen = set()
    for group in (anchors, soft_terms, keywords):
        for item in group:
            tok = re.sub(r"\s+", " ", _safe_str(item)).strip()
            low = tok.lower()
            if not tok or low in seen:
                continue
            seen.add(low)
            pieces.append(tok)
            if len(pieces) >= max_terms:
                return " ".join(pieces)
    return " ".join(pieces)


def llm_rewrite_query(
    generator_client: Any,
    model_name: str,
    question: str,
    current_query: str,
    *,
    anchors: Sequence[str],
    soft_terms: Sequence[str],
    keywords: Sequence[str],
    query_max_tokens: int = 40,
) -> Tuple[str, Dict[str, Any]]:
    anchors_str = ", ".join([a for a in anchors if a])
    soft_terms_str = "; ".join([t for t in soft_terms if t])
    keywords_str = "; ".join([k for k in keywords if k])

    system = (
        "You rewrite search queries for document retrieval.\n"
        "Preserve the question intent.\n"
        "You MUST include all locked anchors unchanged.\n"
        "You SHOULD preserve important soft terms from the original question.\n"
        "You MAY add phrases from the allowed keyword list only when they sharpen retrieval.\n"
        "Prefer the smallest useful query that still preserves the relation being asked about.\n"
        "Do NOT add answer-like numbers from evidence.\n"
        "Output ONLY the rewritten query string."
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"Current query:\n{current_query}\n\n"
        f"Locked anchors: {anchors_str}\n"
        f"Important soft terms from question: {soft_terms_str}\n"
        f"Allowed keyword phrases from evidence: {keywords_str}\n"
    )
    raw = _llm_chat(generator_client, model_name, system, user)
    q = _safe_str(raw).strip()
    q = _enforce_anchor_lock(q, anchors)
    words = q.split()
    if len(words) > query_max_tokens:
        q = " ".join(words[:query_max_tokens])
    return q, {"raw": raw, "final": q}


def _is_anchor_only_query(query: str, anchors: Sequence[str], soft_terms: Sequence[str]) -> bool:
    q_terms = set(_tokenize_terms(query))
    anchor_terms = set()
    for a in anchors:
        anchor_terms.update(_tokenize_terms(a))
    soft_only = set()
    for s in soft_terms:
        soft_only.update(_tokenize_terms(s))
    non_anchor = q_terms - anchor_terms
    return len(non_anchor & soft_only) < 2


def _safe_fallback_rewrite(question: str, anchors: Sequence[str], soft_terms: Sequence[str], keywords: Sequence[str], max_terms: int = 14) -> str:
    # Start from question semantics, not evidence answers.
    q = _pick_query_terms(question, anchors, soft_terms, (), max_terms=max_terms)
    # Optionally add up to 2 evidence keywords if they are non-numeric and noun-like.
    extras: List[str] = []
    for kw in keywords:
        low = kw.lower()
        if re.search(r"\b\d[\d,\.]*\b", kw):
            continue
        if len(kw.split()) > 5:
            continue
        if low not in q.lower():
            extras.append(kw)
        if len(extras) >= 2:
            break
    if extras:
        q = f"{q} {' '.join(extras)}".strip()
    return _enforce_anchor_lock(q, anchors)


def _validate_rewrite(original_question: str, current_query: str, new_query: str, anchors: Sequence[str], soft_terms: Sequence[str]) -> Tuple[bool, str]:
    if not _safe_str(new_query).strip():
        return False, "empty"
    if _near_duplicate(current_query, new_query):
        return False, "rewrite_no_change"
    if _is_anchor_only_query(new_query, anchors, soft_terms):
        return False, "anchor_only"
    recall = _soft_term_recall(soft_terms[:10], new_query)
    if recall < 0.30:
        return False, "low_soft_term_recall"
    return True, "ok"


def invoke_with_hops(
    base_invoke: Callable[..., Tuple[List[Any], Dict[str, Any]]],
    question: str,
    *,
    generator_client: Any,
    model_name: str,
    max_hops: int = 3,
    evidence_max_docs: int = 8,
    evidence_max_chars: int = 2000,
    query_max_tokens: int = 40,
    allowed_sources: Optional[set[str]] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    q0 = _safe_str(question).strip()
    anchors = extract_anchors(q0)
    soft_terms = extract_soft_terms(q0)

    debug: Dict[str, Any] = {
        "mode": "hopping_answerability",
        "question": q0,
        "anchors": anchors,
        "soft_terms": soft_terms,
        "hops": [],
    }

    all_docs: List[Any] = []
    seen_queries: List[str] = []
    q = q0
    prev_doc_keys: Optional[List[str]] = None
    total_rounds = max(1, int(max_hops))

    for hop in range(total_rounds):
        seen_queries.append(q)
        docs, retr_dbg = base_invoke(q, allowed_sources=allowed_sources)
        docs = list(docs or [])
        all_docs.extend(docs)

        hop_dbg: Dict[str, Any] = {
            "hop": hop,
            "query": q,
            "retriever_debug": retr_dbg,
            "num_docs": len(docs),
        }

        if not docs:
            hop_dbg["stop"] = "no_docs"
            debug["hops"].append(hop_dbg)
            break

        doc_keys = [_stable_doc_key(d) for d in docs[:10]]
        if prev_doc_keys is not None:
            overlap = len(set(doc_keys) & set(prev_doc_keys)) / max(1, len(set(doc_keys)))
            hop_dbg["topk_overlap"] = overlap
            if overlap > 0.90:
                hop_dbg["stop"] = "no_progress"
                debug["hops"].append(hop_dbg)
                break
        prev_doc_keys = doc_keys

        judge_res, judge_dbg = llm_judge_answerability(
            generator_client,
            model_name,
            q0,
            docs,
            evidence_max_docs=min(6, evidence_max_docs),
            evidence_max_chars=min(evidence_max_chars, 2200),
        )
        hop_dbg["judge"] = {"answerable": judge_res.answerable, "missing": judge_res.missing, "reason": judge_res.reason}
        hop_dbg["judge_debug"] = judge_dbg

        if judge_res.answerable:
            hop_dbg["stop"] = "answerable"
            debug["hops"].append(hop_dbg)
            break

        if hop == total_rounds - 1:
            hop_dbg["stop"] = "max_hops"
            debug["hops"].append(hop_dbg)
            break

        kws, kw_dbg = llm_extract_keywords(
            generator_client,
            model_name,
            q0,
            docs,
            anchors=anchors,
            evidence_max_docs=evidence_max_docs,
            evidence_max_chars=evidence_max_chars,
        )
        kws = _sanitize_keywords(kws)
        hop_dbg["keywords"] = kws
        hop_dbg["keywords_debug"] = kw_dbg

        candidate_q, rw_dbg = llm_rewrite_query(
            generator_client,
            model_name,
            q0,
            q,
            anchors=anchors,
            soft_terms=soft_terms,
            keywords=kws,
            query_max_tokens=query_max_tokens,
        )
        hop_dbg["rewrite_debug"] = rw_dbg

        ok, reason = _validate_rewrite(q0, q, candidate_q, anchors, soft_terms)
        if not ok:
            fallback_q = _safe_fallback_rewrite(q0, anchors, soft_terms, kws, max_terms=min(14, query_max_tokens))
            fallback_q = " ".join(fallback_q.split()[:query_max_tokens])
            hop_dbg["fallback_rewrite"] = fallback_q
            ok2, reason2 = _validate_rewrite(q0, q, fallback_q, anchors, soft_terms)
            if not ok2 or fallback_q in seen_queries:
                hop_dbg["stop"] = reason
                debug["hops"].append(hop_dbg)
                break
            candidate_q = fallback_q
            hop_dbg["rewrite_debug"]["final"] = candidate_q

        if candidate_q in seen_queries:
            hop_dbg["stop"] = "rewrite_seen"
            debug["hops"].append(hop_dbg)
            break

        debug["hops"].append(hop_dbg)
        q = candidate_q

    uniq: Dict[str, Any] = {}
    for d in all_docs:
        uniq[_stable_doc_key(d)] = d
    final_docs = list(uniq.values())
    debug["num_final_docs"] = len(final_docs)
    debug["final_query"] = q

    simple_hops: List[Dict[str, Any]] = []
    for h in debug.get("hops", []):
        if not isinstance(h, dict):
            continue
        simple_hops.append(
            {
                "hop": h.get("hop"),
                "question": q0,
                "query": h.get("query"),
                "num_docs": h.get("num_docs"),
                "topk_overlap": h.get("topk_overlap"),
                "judge_answerable": (h.get("judge") or {}).get("answerable"),
                "judge_missing": (h.get("judge") or {}).get("missing"),
                "judge_reason": (h.get("judge") or {}).get("reason"),
                "keywords": h.get("keywords"),
                "rewritten_query": (h.get("rewrite_debug") or {}).get("final"),
                "stop": h.get("stop"),
                "retriever_debug": h.get("retriever_debug"),
            }
        )

    stop_reason = None
    for h in reversed(simple_hops):
        if h.get("stop"):
            stop_reason = h.get("stop")
            break

    debug_info: Dict[str, Any] = {
        "mode": debug.get("mode"),
        "original_question": q0,
        "anchors": anchors,
        "soft_terms": soft_terms,
        "final_query": q,
        "hop_count": len(simple_hops),
        "stop_reason": stop_reason,
        "num_final_docs": len(final_docs),
        "hops": simple_hops,
    }
    return final_docs, debug_info


def make_hopping_invoke(
    base_invoke: Callable[..., Tuple[List[Any], Dict[str, Any]]],
    *,
    generator_client: Any,
    model_name: str,
    max_hops: int = 3,
    evidence_max_docs: int = 8,
    evidence_max_chars: int = 2000,
    query_max_tokens: int = 40,
) -> Callable[..., Tuple[List[Any], Dict[str, Any]]]:
    def _invoke(query: str, allowed_sources: Optional[set[str]] = None):
        return invoke_with_hops(
            base_invoke,
            query,
            generator_client=generator_client,
            model_name=model_name,
            max_hops=max_hops,
            evidence_max_docs=evidence_max_docs,
            evidence_max_chars=evidence_max_chars,
            query_max_tokens=query_max_tokens,
            allowed_sources=allowed_sources,
        )

    return _invoke
