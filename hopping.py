from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ----------------------------
# Utilities
# ----------------------------

def _safe_str(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""


def _doc_text(doc: Any) -> str:
    """Best-effort extraction of chunk text from various doc types."""
    for attr in ("page_content", "text", "content", "chunk", "body"):
        v = getattr(doc, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    # Sometimes doc is dict-like
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
    """Stable-ish dedupe key across hops."""
    md = _doc_meta(doc)
    # Prefer explicit doc_id if present
    doc_id = md.get("doc_id") or md.get("id")
    if doc_id:
        return f"id::{_safe_str(doc_id)}"

    # Try source file + page + chunk index
    src = _safe_str(md.get("source_file") or md.get("source") or md.get("path"))
    page = _safe_str(md.get("page") or md.get("page_number") or md.get("p"))
    chunk = _safe_str(md.get("chunk_id") or md.get("chunk") or md.get("chunk_index"))
    if src and (page or chunk):
        return f"s::{src}::{page}::{chunk}"

    txt = _doc_text(doc)[:2000]
    h = hashlib.md5((src + "||" + txt).encode("utf-8")).hexdigest()
    return f"h::{h}"


def _build_evidence(docs: Sequence[Any], max_docs: int, max_chars: int) -> str:
    """Concatenate doc texts into a bounded evidence string."""
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
    """Cheap near-duplicate check."""
    na, nb = _normalize_query(a), _normalize_query(b)
    if na == nb:
        return True
    # If one is contained in the other and length ratio is close, treat as same
    if na in nb or nb in na:
        la, lb = len(na), len(nb)
        if min(la, lb) / max(la, lb) > 0.85:
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
    r"\b\d+(?:\.\d+){1,4}\b",  # 5.3.2 style
]

_SOFT_PATTERNS = [
    r"\b[A-Z]{2,6}\b",
]

def extract_anchors(question: str, max_soft: int = 6) -> List[str]:
    """Extract anchors from the original question."""
    q = _safe_str(question)
    hard: List[str] = []
    for pat in _HARD_PATTERNS + _STRUCT_PATTERNS:
        for m in re.finditer(pat, q, flags=re.IGNORECASE):
            tok = m.group(0).strip()
            if tok and tok not in hard:
                hard.append(tok)

    soft: List[str] = []
    for pat in _SOFT_PATTERNS:
        for m in re.finditer(pat, q):
            tok = m.group(0).strip()
            # Avoid adding very generic acronyms
            if tok in {"ISO", "IEC", "UN", "ECE", "EU"}:
                continue
            if tok and tok not in hard and tok not in soft:
                soft.append(tok)
            if len(soft) >= max_soft:
                break
        if len(soft) >= max_soft:
            break

    return hard + soft


def _enforce_anchor_lock(query: str, anchors: Sequence[str]) -> str:
    """Ensure locked anchors appear in the rewritten query."""
    q = _safe_str(query).strip()
    low = q.lower()
    for a in anchors:
        aa = _safe_str(a).strip()
        if not aa:
            continue
        if aa.lower() not in low:
            q = f"{q} {aa}".strip()
            low = q.lower()
    return q


# ----------------------------
# LLM helpers
# ----------------------------

@dataclass
class JudgeResult:
    answerable: bool
    missing: List[str]
    reason: str


def _llm_chat(generator_client: Any, model_name: str, system: str, user: str) -> str:
    """Call chat.completions.create and return assistant content."""
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
    """Parse JSON even if the model wraps it in code fences."""
    s = _safe_str(txt).strip()
    # strip fenced code blocks
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # try direct
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        pass
    # try to locate first {...}
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
    """Binary answerability judge: can the question be answered using ONLY the excerpts?"""
    evidence = _build_evidence(docs, evidence_max_docs, evidence_max_chars)
    system = (
        "You are a strict judge for a retrieval-augmented QA system.\n"
        "Answer YES only if the question can be answered using verbatim content in the excerpts.\n"
        "If any required numbers, IDs, clauses, or definitions are missing in the excerpts, answer NO.\n"
        "Return ONLY valid JSON."
    )
    user = (
        "Return JSON with keys:\n"
        '  "answerable": "YES" or "NO"\n'
        '  "missing": list of short labels (e.g., "definition", "numeric limit", "procedure", "clause", "scope", "exception", "other")\n'
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
    """Extract 5-12 high-signal keywords/phrases that appear verbatim in the excerpts."""
    evidence = _build_evidence(docs, evidence_max_docs, evidence_max_chars)
    anchors_str = ", ".join([a for a in anchors if a])

    system = (
        "You extract search keywords for document chunk retrieval (PDFs or wiki text).\n"
        "You MUST ONLY output phrases that appear verbatim in the provided excerpts.\n"
        "Do NOT invent clause numbers, IDs, or terms not present in the excerpts.\n"
        "Return ONLY valid JSON."
    )
    user = (
        "Return JSON with key \"keywords\" as a list of 5 to 12 items.\n"
        "- Each item must be a short phrase (1-6 words) copied verbatim from the excerpts.\n"
        "- Prefer technical terms, parameter names, component names, test procedure names, and table/annex labels if present.\n"
        "- Avoid generic words.\n\n"
        f"Locked anchors from the question (must be preserved later): {anchors_str}\n\n"
        f"Question:\n{question}\n\n"
        f"Excerpts:\n{evidence}\n"
    )
    raw = _llm_chat(generator_client, model_name, system, user)
    data = _parse_json_best_effort(raw)
    kws = data.get("keywords") if isinstance(data.get("keywords"), list) else []
    kws = [_safe_str(x).strip() for x in kws if _safe_str(x).strip()]
    dbg = {"raw": raw, "parsed_keywords": kws}
    return kws, dbg


def _sanitize_keywords(kws: Sequence[str]) -> List[str]:
    """Drop duplicates and low-signal items."""
    out: List[str] = []
    seen = set()
    for k in kws:
        kk = re.sub(r"\s+", " ", _safe_str(k)).strip()
        if not kk:
            continue
        # Drop very short / generic
        if len(kk) < 3:
            continue
        if kk.lower() in {"the", "and", "or", "shall", "must", "may", "should"}:
            continue
        low = kk.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(kk)
    # bound
    return out[:12]


def llm_rewrite_query(
    generator_client: Any,
    model_name: str,
    question: str,
    current_query: str,
    *,
    anchors: Sequence[str],
    keywords: Sequence[str],
    query_max_tokens: int = 40,
) -> Tuple[str, Dict[str, Any]]:
    """Rewrite query using only provided keywords + locked anchors, preserving intent."""
    anchors_str = ", ".join([a for a in anchors if a])
    keywords_str = "; ".join([k for k in keywords if k])

    system = (
        "You rewrite search queries for a document retrieval system (PDFs or wiki text).\n"
        "You MUST preserve the question intent.\n"
        "You MUST include all locked anchors unchanged.\n"
        "You MAY ONLY add phrases from the provided keyword list.\n"
        "Do NOT add clause/annex/table numbers unless they are in the keyword list.\n"
        "Output ONLY the rewritten query string."
    )
    user = (
        f"Original question:\n{question}\n\n"
        f"Current query:\n{current_query}\n\n"
        f"Locked anchors (must include unchanged): {anchors_str}\n"
        f"Allowed keyword phrases (may add ONLY these): {keywords_str}\n"
    )
    raw = _llm_chat(generator_client, model_name, system, user)
    q = _safe_str(raw).strip()
    q = _enforce_anchor_lock(q, anchors)
    # Bound length by words
    words = q.split()
    if len(words) > query_max_tokens:
        q = " ".join(words[:query_max_tokens])
    dbg = {"raw": raw, "final": q}
    return q, dbg


# ----------------------------
# Multi-hop retrieval driver
# ----------------------------

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
    """Perform retrieval with answerability-driven multi-hop query rewriting.

    Parameters kept compatible with the previous hopping.py so existing code won't break.
    """
    q0 = _safe_str(question).strip()
    anchors = extract_anchors(q0)

    debug: Dict[str, Any] = {
        "mode": "hopping_answerability",
        "question": q0,
        "anchors": anchors,
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

        # Stop: no docs -> do not rewrite from nothing
        if not docs:
            hop_dbg["stop"] = "no_docs"
            debug["hops"].append(hop_dbg)
            break

        # Progress stop: if retrieval is not changing materially
        doc_keys = [_stable_doc_key(d) for d in docs[:10]]
        if prev_doc_keys is not None:
            overlap = len(set(doc_keys) & set(prev_doc_keys)) / max(1, len(set(doc_keys)))
            hop_dbg["topk_overlap"] = overlap
            if overlap > 0.85:
                hop_dbg["stop"] = "no_progress"
                debug["hops"].append(hop_dbg)
                break
        prev_doc_keys = doc_keys

        # Judge answerability from current evidence
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

        anchor_hits = 0
        for a in anchors:
            aa = a.lower()
            if aa and any(aa in _doc_text(d).lower() for d in docs[:8]):
                anchor_hits += 1
        anchor_cov = anchor_hits / max(1, len(anchors))
        hop_dbg["anchor_cov"] = round(anchor_cov, 3)
        answerable = judge_res.answerable and anchor_cov >= 0.35

        if answerable:
            hop_dbg["stop"] = "answerable"
            debug["hops"].append(hop_dbg)
            break

        if judge_res.answerable:
            hop_dbg["stop"] = "answerable"
            debug["hops"].append(hop_dbg)
            break

        if hop == total_rounds - 1:
            hop_dbg["stop"] = "max_hops"
            debug["hops"].append(hop_dbg)
            break

        # Extract keywords from evidence
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

        if not kws:
            hop_dbg["stop"] = "no_keywords"
            debug["hops"].append(hop_dbg)
            break

        # Rewrite query using only extracted keywords + locked anchors
        new_q, rw_dbg = llm_rewrite_query(
            generator_client,
            model_name,
            q0,
            q,
            anchors=anchors,
            keywords=kws,
            query_max_tokens=query_max_tokens,
        )
        hop_dbg["rewrite_debug"] = rw_dbg

        # Stop if rewrite didn't change meaningfully
        if _near_duplicate(q, new_q) or new_q in seen_queries:
            hop_dbg["stop"] = "rewrite_no_change"
            debug["hops"].append(hop_dbg)
            break

        debug["hops"].append(hop_dbg)
        q = new_q

    # Dedupe final docs
    uniq: Dict[str, Any] = {}
    for d in all_docs:
        uniq[_stable_doc_key(d)] = d
    final_docs = list(uniq.values())
    debug["num_final_docs"] = len(final_docs)
    debug["final_query"] = q


# ----------------------------
    # Single debug_info dict (no bm25/dense shim)
    # ----------------------------
    hops_list = debug.get("hops") if isinstance(debug.get("hops"), list) else []
    simple_hops: List[Dict[str, Any]] = []
    for h in hops_list:
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
    
    # Derive stop_reason from the last hop that has it (or None).
    stop_reason = None
    for h in reversed(simple_hops):
        if h.get("stop"):
            stop_reason = h.get("stop")
            break
    
    debug_info: Dict[str, Any] = {
        "mode": debug.get("mode"),
        "original_question": q0,
        "anchors": anchors,
        "final_query": q,
        "hop_count": len(simple_hops),
        "stop_reason": stop_reason,
        "num_final_docs": debug.get("num_final_docs"),
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
    """Return an invoke() wrapper compatible with existing retriever code.

    The returned function has signature:
        invoke(query: str, allowed_sources: set[str] | None = None) -> (docs, debug_info)

    It returns a *single* debug_info dict (no bm25/dense shims).
    """

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
