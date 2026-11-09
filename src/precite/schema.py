"""Schema canonique des features Precite.

Source unique de verite pour :
- les colonnes utilisees a l'entrainement (FEATURE_COLUMNS)
- les transformations a appliquer (FEATURE_TRANSFORMS)
- les schemas d'ecriture Parquet (FEATURES_SCHEMA, PAPERS_SCHEMA)
- les colonnes exposees par l'API (API_COLUMNS)

Types Arrow FIDELES aux schemas originaux pour ne pas casser la
compatibilite des fichiers existants sur HuggingFace.
"""

from __future__ import annotations

import pyarrow as pa

# ─── Features utilisees a l'entrainement ────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    "n_references",
    "mean_ref_age",
    "title_len",
    "abstract_len",
    "n_mesh",
    "n_pages",
    "n_chemicals",
    "n_grants",
    "language_rank",
    "score_a",
    "score_c",
    "score_h",
    "is_journal_article",
    "is_review",
    "is_clinical_trial",
    "is_rct",
    "is_meta_analysis",
    "is_editorial",
    "is_letter",
    "is_comment",
    "is_case_report",
]


# ─── Transformations ────────────────────────────────────────────────────────

FEATURE_TRANSFORMS: dict[str, str] = {
    "n_references": "arsinh",
    "mean_ref_age": "arsinh",
    "abstract_len": "arsinh",
}


# ─── Colonnes par categorie ─────────────────────────────────────────────────

META_COLUMNS: list[str] = [
    "pmid",
    "year",
    "batch_id",
    "ingested_at",
    "medline_status",
]

TARGET_COLUMNS: list[str] = [
    "citations_3y",
    "weighted_citations_3y",
]

FILTER_COLUMNS: list[str] = [
    "is_research_article",
    "has_enough_history",
]

HAS_INDICATOR_COLUMNS: list[str] = [
    "has_reference_list",
    "has_ref_years",
    "has_pages",
    "has_abstract",
]


# ─── Schema d'ecriture ParquetBatchWriter (ingestion) ──────────────────────

FEATURES_SCHEMA = pa.schema([
    ("pmid", pa.string()),
    ("year", pa.int32()),
    ("n_references", pa.int32()),
    ("mean_ref_age", pa.float32()),
    ("title_len", pa.int32()),
    ("abstract_len", pa.int32()),
    ("n_pages", pa.int32()),
    ("n_mesh", pa.int32()),
    ("n_chemicals", pa.int32()),
    ("n_grants", pa.int32()),
    ("language_rank", pa.int8()),
    ("journal_issn", pa.string()),
    ("score_a", pa.float32()),
    ("score_c", pa.float32()),
    ("score_h", pa.float32()),
    ("medline_status", pa.string()),
    ("is_journal_article", pa.int8()),
    ("is_review", pa.int8()),
    ("is_clinical_trial", pa.int8()),
    ("is_rct", pa.int8()),
    ("is_meta_analysis", pa.int8()),
    ("is_editorial", pa.int8()),
    ("is_letter", pa.int8()),
    ("is_comment", pa.int8()),
    ("is_case_report", pa.int8()),
])


# ─── Schema d'ecriture ParquetStore.write_papers (API) ─────────────────────

PAPERS_SCHEMA = pa.schema([
    ("pmid", pa.string()),
    ("year", pa.int32()),
    ("batch_id", pa.string()),
    ("ingested_at", pa.string()),
    ("medline_status", pa.string()),
    ("n_pages", pa.int32()),
    *[(c, pa.float64()) for c in [
        "n_references", "mean_ref_age", "title_len", "abstract_len",
        "n_mesh", "n_chemicals", "n_grants", "language_rank",
        "score_a", "score_c", "score_h",
    ]],
    *[(c, pa.int8()) for c in [
        "is_journal_article", "is_review", "is_clinical_trial",
        "is_rct", "is_meta_analysis", "is_editorial",
        "is_letter", "is_comment", "is_case_report",
        "is_research_article", "has_enough_history",
    ]],
    ("citations_3y", pa.float64()),
    ("weighted_citations_3y", pa.float64()),
])


# ─── Colonnes exposees par rebuild_papers.py (API /papers) ─────────────────

API_COLUMNS: list[str] = [
    "pmid", "year", "medline_status", "batch_id", "ingested_at",
    "n_references", "mean_ref_age", "title_len", "abstract_len",
    "n_mesh", "n_chemicals", "n_grants", "language_rank",
    "score_a", "score_c", "score_h", "n_pages",
    "is_journal_article", "is_review", "is_clinical_trial",
    "is_rct", "is_meta_analysis", "is_editorial",
    "is_letter", "is_comment", "is_case_report",
    "citations_3y", "weighted_citations_3y",
]