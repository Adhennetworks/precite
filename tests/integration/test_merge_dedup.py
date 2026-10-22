
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.integration
def test_merge_dedup_within_file(tmp_path):
    # Crée 2 fichiers avec le même PMID
    schema = pa.schema([
        ("pmid", pa.string()),
        ("year", pa.int32()),
        ("n_pages", pa.int32()),
    ])
    t1 = pa.Table.from_pylist(
        [{"pmid": "1", "year": 2020, "n_pages": 5},
         {"pmid": "1", "year": 2020, "n_pages": 5}],
        schema=schema,
    )
    f1 = tmp_path / "chunk-0000.parquet"
    pq.write_table(t1, f1)

    # Simule le merge
    con = duckdb.connect()
    con.execute(f"""
        CREATE TABLE merged AS
        SELECT DISTINCT pmid, year, n_pages
        FROM read_parquet('{f1.as_posix()}')
    """)
    n = con.execute("SELECT COUNT(*) FROM merged").fetchone()[0]
    assert n == 1