"""Pipeline d'ingestion : URL(s) .xml.gz -> features Parquet (S3 ou local).

Streaming pur, rien sur disque. Reprise HTTP Range geree par ftp.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from precite.config import load_field_config
from precite.ingest.features import compute_features, utc_now_year
from precite.ingest.ftp import stream_file
from precite.ingest.parser import iter_articles
from precite.storage.parquet import ParquetBatchWriter
from precite.storage.s3 import ensure_bucket
from precite.utils.logging import get_logger
from precite.utils.timing import Timer

log = get_logger(__name__)


def _prefix_from_url(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    stem = name.replace(".xml.gz", "").replace(".xml", "")
    return f"features/{stem}"


def ingest_url(
    url: str,
    *,
    local_dir: str | Path | None = None,
    skip_pmids: set[str] | None = None,
) -> dict[str, int]:
    """Ingere un fichier PubMed en streaming pur.

    skip_pmids : PMIDs a ignorer (deja presents dans le dataset).
    """
    fields_cfg = load_field_config()
    current_year = utc_now_year()
    timer = Timer()

    writer = ParquetBatchWriter(
        prefix=_prefix_from_url(url),
        local_dir=local_dir,
    )
    n_read = 0
    n_kept = 0
    n_skipped_existing = 0
    n_skipped_dupe = 0
    seen_local: set[str] = set()

    try:
        chunks = stream_file(url)
        for raw in iter_articles(chunks, fields_cfg):
            n_read += 1
            pmid = str(raw.get("pmid", ""))

            # Dedup contre dataset existant
            if skip_pmids and pmid in skip_pmids:
                n_skipped_existing += 1
                continue

            # Dedup intra-fichier
            if pmid in seen_local:
                n_skipped_dupe += 1
                continue
            seen_local.add(pmid)

            feats = compute_features(raw, current_year=current_year)
            if feats is None:
                continue
            writer.add(feats)
            n_kept += 1

            if n_read % 50_000 == 0:
                log.info(
                    "pipeline.progress",
                    url=url,
                    read=n_read,
                    kept=n_kept,
                    elapsed_s=round(timer.elapsed(), 1),
                )
    finally:
        writer.close()

    summary = {
        "articles_read": n_read,
        "articles_kept": n_kept,
        "skipped_existing": n_skipped_existing,
        "skipped_dupe": n_skipped_dupe,
        "parquet_files": writer.file_index,
        "elapsed_seconds": int(timer.elapsed()),
    }
    log.info("pipeline.done", url=url, **summary)
    return summary


def ingest_many(
    urls: Iterable[str],
    *,
    auto_predict: bool = False,
    local_dir: str | Path | None = None,
    db_path: str | None = None,
    skip_pmids: set[str] | None = None,
) -> list[dict[str, int]]:
    """Ingere plusieurs fichiers sequentiellement."""
    if local_dir is None:
        ensure_bucket()

    results = []
    for url in urls:
        summary = ingest_url(url, local_dir=local_dir, skip_pmids=skip_pmids)
        results.append(summary)

        if auto_predict and local_dir is None:
            from precite.config import get_settings
            from precite.predict.batch import predict_batch

            s = get_settings()
            batch_id = url.rsplit("/", 1)[-1].replace(".xml.gz", "").replace(".xml", "")
            source = f"s3://{s.s3_bucket}/{_prefix_from_url(url)}"
            try:
                psummary = predict_batch(source=source, batch_id=batch_id, db_path=db_path)
                summary["predictions"] = psummary["predictions"]
                summary["models"] = psummary["models"]
            except Exception as exc:
                log.error("ingest.auto_predict_failed", url=url, error=str(exc))
                summary["predictions"] = 0
                summary["models"] = 0

    return results