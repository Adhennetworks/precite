"""Ecriture Parquet en streaming, vers S3-compatible OU local."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from precite.config import get_settings
from precite.storage.s3 import get_s3_client
from precite.utils.logging import get_logger

log = get_logger(__name__)


FEATURES_SCHEMA = pa.schema(
    [
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
    ]
)


class ParquetBatchWriter:
    def __init__(
        self,
        prefix: str,
        *,
        bucket: str | None = None,
        batch_size: int | None = None,
        row_group_size: int | None = None,
        local_dir: str | Path | None = None,
    ):
        s = get_settings()
        self.bucket = bucket or s.s3_bucket
        self.prefix = prefix.rstrip("/")
        self.batch_size = batch_size or s.write_batch_size
        self.row_group_size = row_group_size or s.row_group_size
        self.buffer: list[dict[str, Any]] = []
        self.file_index = 0
        self.total_rows = 0

        self.local_dir: Path | None = Path(local_dir) if local_dir else None
        if self.local_dir is not None:
            (self.local_dir / self.prefix).mkdir(parents=True, exist_ok=True)
            self._client = None
        else:
            self._client = get_s3_client()

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=FEATURES_SCHEMA)
        buf = io.BytesIO()
        pq.write_table(
            table,
            buf,
            compression="zstd",
            row_group_size=self.row_group_size,
        )
        buf.seek(0)
        size = buf.getbuffer().nbytes

        filename = f"part-{self.file_index:05d}-{uuid.uuid4().hex[:8]}.parquet"

        if self.local_dir is not None:
            out_path = self.local_dir / self.prefix / filename
            out_path.write_bytes(buf.getvalue())
            log.info(
                "parquet.flush.local",
                path=str(out_path),
                rows=len(self.buffer),
                bytes=size,
            )
        else:
            key = f"{self.prefix}/{filename}"
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=buf.getvalue(),
                ContentType="application/octet-stream",
            )
            log.info(
                "parquet.flush.s3",
                bucket=self.bucket,
                key=key,
                rows=len(self.buffer),
                bytes=size,
            )

        self.total_rows += len(self.buffer)
        self.buffer.clear()
        self.file_index += 1

    def close(self) -> None:
        self.flush()
        log.info(
            "parquet.writer.close",
            destination=("local:" + str(self.local_dir)) if self.local_dir else f"s3://{self.bucket}",
            prefix=self.prefix,
            files=self.file_index,
            total_rows=self.total_rows,
        )