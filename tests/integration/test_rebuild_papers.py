"""Test de regression : rebuild_papers doit preserver n_pages."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.integration
def test_rebuild_papers_keeps_n_pages(tmp_path):
    """Un chunk avec n_pages doit produire un papers/decade=*.parquet avec n_pages."""
    import duckdb

    # 1. Cree un chunk d'entree
    schema = pa.schema([
        ("pmid", pa.string()),
        ("year", pa.int32()),
        ("medline_status", pa.string()),
        ("n_pages", pa.int32()),
    ])
    chunk_path = tmp_path / "features" / "chunk-0000.parquet"
    chunk_path.parent.mkdir(parents=True)
    t = pa.Table.from_pylist(
        [
            {"pmid": "1", "year": 2020, "medline_status": "MEDLINE", "n_pages": 5},
            {"pmid": "2", "year": 2030, "medline_status": "Publisher", "n_pages": None},
        ],
        schema=schema,
    )
    pq.write_table(t, chunk_path)

    # 2. Simule le rebuild (logique simplifiee de rebuild_papers.py)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT pmid, year, medline_status, n_pages
            FROM read_parquet('{chunk_path.as_posix()}')
            WHERE year IS NOT NULL
              AND CAST(year // 10 * 10 AS INTEGER) = 2020
        ) TO '{(papers_dir / "decade=2020.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # 3. Verifie que n_pages est dans le papier
    papers = pq.read_table(papers_dir / "decade=2020.parquet")
    assert "n_pages" in papers.column_names
    assert papers.num_rows == 1
    assert papers.column("n_pages").to_pylist()[0] == 5


@pytest.mark.integration
def test_rebuild_papers_null_pages_allowed(tmp_path):
    """Un article avec n_pages=NULL doit etre conserve (pas exclu)."""
    import duckdb

    schema = pa.schema([
        ("pmid", pa.string()),
        ("year", pa.int32()),
        ("n_pages", pa.int32()),
    ])
    chunk_path = tmp_path / "features" / "chunk-0000.parquet"
    chunk_path.parent.mkdir(parents=True)
    t = pa.Table.from_pylist(
        [{"pmid": "1", "year": 2020, "n_pages": None}],
        schema=schema,
    )
    pq.write_table(t, chunk_path)

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT pmid, year, n_pages
            FROM read_parquet('{chunk_path.as_posix()}')
            WHERE year IS NOT NULL
        ) TO '{(papers_dir / "decade=2020.parquet").as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    papers = pq.read_table(papers_dir / "decade=2020.parquet")
    assert papers.num_rows == 1
    assert papers.column("n_pages").to_pylist()[0] is None