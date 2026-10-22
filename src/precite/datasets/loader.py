"""Chargement des features Parquet depuis S3 ou local, split temporel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from precite.utils.logging import get_logger

log = get_logger(__name__)

# ─── Transformations des features a queue lourde ───────────────────────────
# Ces features sont transformees par arsinh au moment de la construction
# de la matrice de design (et NON stockees transformees en base).
#
# Justification : n_references, mean_ref_age et abstract_len ont une queue
# tres lourde. Les coefficients appris sur la plage normale (n_refs ~30)
# produisent des extrapolations aberrantes sur les articles extremes
# (n_refs > 1000), amplifiees par l'inverse sinh a la sortie.
#
# arsinh(x) ≈ log(2x) pour x grand, et arsinh(0) = 0. Cela borne la
# contribution de ces features dans le predicteur lineaire.

FEATURE_TRANSFORMS = {
    "n_references": "arsinh",
    "mean_ref_age": "arsinh",
    "abstract_len": "arsinh",
}


def _apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les transformations arsinh sur les features concernees."""
    for col, kind in FEATURE_TRANSFORMS.items():
        if col not in df.columns:
            continue
        if kind == "arsinh":
            df[col] = np.arcsinh(df[col].astype("float64"))
    return df

FEATURE_COLUMNS = [
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

TARGET_CITATIONS = "citations"
TARGET_WEIGHTED = "weighted_citations"


def load_features(
    source: str | Path,
    *,
    columns: list[str] | None = None,
    filters: list[tuple] | None = None,
) -> pd.DataFrame:
    """Charge un dataset Parquet depuis S3 ou local."""
    src = str(source)

    if src.startswith("s3://"):
        import os

        from precite.config import get_settings

        s = get_settings()
        os.environ.setdefault("AWS_ENDPOINT_URL", s.s3_endpoint)
        os.environ.setdefault("AWS_ACCESS_KEY_ID", s.s3_access_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", s.s3_secret_key)
        os.environ.setdefault("AWS_REGION", s.s3_region)
        dataset = ds.dataset(src, format="parquet")
        table = dataset.to_table(columns=columns, filter=filters)
        df = table.to_pandas()
    else:
        path = Path(src)
        if path.is_file():
            table = pq.read_table(path, columns=columns)
            df = table.to_pandas()
        else:
            files = sorted(path.rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"aucun .parquet dans {path}")
            tables = [pq.read_table(f, columns=columns) for f in files]
            table = pa.concat_tables(tables, promote_options="default")
            df = table.to_pandas()

    log.info("loader.loaded", source=src, rows=len(df), columns=len(df.columns))
    return df


def temporal_split(
    df: pd.DataFrame,
    *,
    cutoff_year: int,
    year_col: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporel strict."""
    train = df[df[year_col] < cutoff_year].copy()
    test = df[df[year_col] >= cutoff_year].copy()
    log.info(
        "loader.temporal_split",
        cutoff=cutoff_year,
        train_rows=len(train),
        test_rows=len(test),
    )
    return train, test


def build_design_matrix(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    target: str = TARGET_CITATIONS,
    drop_constant: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Retourne (X, y) avec seulement les colonnes presentes dans df."""
    feats = features or FEATURE_COLUMNS
    available = [c for c in feats if c in df.columns]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        log.warning("loader.missing_features", missing=missing)

    if target not in df.columns:
        raise ValueError(f"target '{target}' absent des donnees")

    X = df[available].fillna(0).astype("float64")
    X = _apply_transforms(X)
    y = df[target].astype("float64")

    if drop_constant:
        before = X.shape[1]
        nunique = X.nunique(dropna=False)
        zero_var = nunique[nunique <= 1].index.tolist()
        if zero_var:
            log.info("loader.drop_constant", columns=zero_var)
            X = X.drop(columns=zero_var)

        if X.shape[1] > 1:
            corr = X.corr().abs()
            upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
            redundant = [c for c in upper.columns if (upper[c] >= 0.9999).any()]
            if redundant:
                log.info("loader.drop_redundant", columns=redundant)
                X = X.drop(columns=redundant)

        after = X.shape[1]
        log.info("loader.design_matrix", before=before, after=after)

    return X, y