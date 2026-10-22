"""Calcul des features a la volee.

Inclut medline_status : string du XML PubMed (MEDLINE, Publisher,
In-Process, In-Data-Review, PubMed-not-MEDLINE, OldMEDLINE, ou None).
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from precite.utils.logging import get_logger

log = get_logger(__name__)


PUB_TYPE_COLUMNS = {
    "Journal Article": "is_journal_article",
    "Review": "is_review",
    "Clinical Trial": "is_clinical_trial",
    "Randomized Controlled Trial": "is_rct",
    "Meta-Analysis": "is_meta_analysis",
    "Editorial": "is_editorial",
    "Letter": "is_letter",
    "Comment": "is_comment",
    "Case Reports": "is_case_report",
}


@lru_cache(maxsize=1)
def _load_mesh_map() -> dict[str, set[str]]:
    path = Path("data/mesh/mesh_map.pkl")
    if not path.exists():
        log.warning("mesh_map.missing", path=str(path))
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    log.info("mesh_map.loaded", n=len(data))
    return data


def _extract_year(record: dict[str, Any]) -> int | None:
    y = record.get("year")
    if y and str(y).isdigit():
        return int(y)
    md = record.get("medline_date")
    if md:
        first = str(md).split()[0]
        if first.isdigit():
            return int(first)
    return None


def _normalize_to_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _language_rank(langs: list[str]) -> int:
    if not langs:
        return 3
    lang_set = set(langs)
    if lang_set == {"eng"}:
        return 1
    if "eng" in lang_set:
        return 2
    return 3


def _compute_mesh_scores(mesh_terms: list[str]) -> tuple[float, float, float]:
    if not mesh_terms:
        return 0.0, 0.0, 0.0

    mesh_map = _load_mesh_map()
    if not mesh_map:
        return 0.0, 0.0, 0.0

    n_a = n_c = n_h = 0
    for term in mesh_terms:
        cats = mesh_map.get(term.lower(), set())
        if "A" in cats:
            n_a += 1
        if "C" in cats:
            n_c += 1
        if "H" in cats:
            n_h += 1

    total = len(mesh_terms)
    return n_a / total, n_c / total, n_h / total


def compute_features(
    record: dict[str, Any],
    *,
    current_year: int | None = None,
) -> dict[str, Any] | None:
    pmid = record.get("pmid")
    if not pmid:
        return None

    year = _extract_year(record)
    if year is None or year < 1800 or year > 2100:
        return None

    title = record.get("title") or ""
    abstract = record.get("abstract")
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    abstract = abstract or ""

    title_len = len(title.split())
    abstract_len = len(abstract.split())

    ref_years: list[int] = record.get("ref_years") or []
    n_refs = record.get("n_references", 0) or 0

    valid_ref_years = [y for y in ref_years if 1800 <= y <= year]
    mean_ref_age = year - sum(valid_ref_years) / len(valid_ref_years) if valid_ref_years else 0.0

    n_pages = record.get("n_pages")
    if n_pages is not None:
        try:
            n_pages = int(n_pages)
        except (ValueError, TypeError):
            n_pages = None
    langs = _normalize_to_list(record.get("language"))
    language_rank = _language_rank(langs)

    pub_types = _normalize_to_list(record.get("pub_types"))
    pub_type_flags = {col: 0 for col in PUB_TYPE_COLUMNS.values()}
    for pt in pub_types:
        if pt in PUB_TYPE_COLUMNS:
            pub_type_flags[PUB_TYPE_COLUMNS[pt]] = 1

    mesh_terms = _normalize_to_list(record.get("mesh_terms"))
    chemicals = _normalize_to_list(record.get("chemical_list"))
    grants = _normalize_to_list(record.get("grant_list"))

    score_a, score_c, score_h = _compute_mesh_scores(mesh_terms)

    # Status MEDLINE reel extrait du XML par parser.py
    # Valeurs possibles : MEDLINE, Publisher, In-Process, In-Data-Review,
    # PubMed-not-MEDLINE, OldMEDLINE, ou None
    medline_status = record.get("medline_status")

    return {
        "pmid": str(pmid),
        "year": year,
        "n_references": n_refs,
        "n_pages": n_pages,
        "mean_ref_age": round(mean_ref_age, 3),
        "title_len": title_len,
        "abstract_len": abstract_len,
        "n_mesh": len(mesh_terms),
        "n_chemicals": len(chemicals),
        "n_grants": len(grants),
        "language_rank": language_rank,
        "journal_issn": record.get("journal_issn"),
        "score_a": round(score_a, 4),
        "score_c": round(score_c, 4),
        "score_h": round(score_h, 4),
        "medline_status": medline_status,
        **pub_type_flags,
    }


def utc_now_year() -> int:
    return datetime.now(UTC).year