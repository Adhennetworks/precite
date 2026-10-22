"""Extrait (pmid, medline_status, n_pages) de TOUS les fichiers PubMed XML.

Usage :
    python scripts/backfill_medline_status.py
    python scripts/backfill_medline_status.py --start-from 500
    python scripts/backfill_medline_status.py --skip-cache
"""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import time
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "metadata_parts"
OUT_FINAL = ROOT / "data" / "article_metadata.parquet"
CACHE_DIR = ROOT / "data" / ".cache_pubmed"

BASELINE_DIR = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
UPDATEFILES_DIR = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"

SCHEMA = pa.schema([
    ("pmid", pa.int64()),
    ("status", pa.string()),
    ("n_pages", pa.int32()),
])

FLUSH_EVERY = 500_000  # reduit de 5M a 500k (sauvegarde plus frequente)


def _count_pages(pages_raw: str | None) -> int | None:
    if not pages_raw:
        return None
    total = 0
    for match in re.finditer(r"([A-Za-z]*)(\d+)(?:\s*-\s*([A-Za-z]*)(\d+))?", pages_raw):
        prefix, start, _, end = match.groups()
        start_n = int(start)
        if end:
            end_n = int(end)
            n = end_n - start_n + 1
            if 0 < n < 500:
                total += n
            else:
                total += 1
        else:
            total += 1
    return total if total > 0 else None


def list_files(base_url: str, pattern: str) -> list[str]:
    print(f"Listing {base_url}...")
    r = httpx.get(base_url, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    names = []
    rx = re.compile(rf'href="({pattern})"')
    for line in r.text.splitlines():
        m = rx.search(line)
        if m:
            names.append(m.group(1))
    return sorted(names)


def parse_xml_gz(content: bytes) -> list[tuple[int, str | None, int | None]]:
    results: list[tuple[int, str | None, int | None]] = []
    try:
        gz = gzip.GzipFile(fileobj=io.BytesIO(content))
        context = etree.iterparse(
            gz,
            events=("end",),
            tag="PubmedArticle",
            recover=True,
            huge_tree=True,
        )
        for _event, elem in context:
            pmid_node = elem.find(".//PMID")
            if pmid_node is None or not pmid_node.text:
                continue
            try:
                pmid = int(pmid_node.text)
            except ValueError:
                continue
            mc = elem.find("MedlineCitation")
            status = mc.get("Status") if mc is not None else None
            pgn = elem.find(".//MedlinePgn")
            pages_raw = pgn.text.strip() if (pgn is not None and pgn.text) else None
            n_pages = _count_pages(pages_raw)
            results.append((pmid, status, n_pages))
            elem.clear()
            parent = elem.getparent()
            if parent is not None:
                while elem.getprevious() is not None:
                    del parent[0]
    except Exception as exc:
        print(f"    [WARN] parse error: {exc}", flush=True)
    return results


def download_and_parse(
    url: str,
    client: httpx.Client,
    *,
    use_cache: bool = True,
) -> list[tuple[int, str | None, int | None]]:
    import time as _time

    cache_file = CACHE_DIR / url.rsplit("/", 1)[-1]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_file.exists():
        content = cache_file.read_bytes()
    else:
        content = None
        last_exc = None
        for attempt in range(1, 4):
            try:
                r = client.get(url, timeout=600.0)
                r.raise_for_status()
                content = r.content
                break
            except Exception as exc:
                last_exc = exc
                wait = 5 * attempt
                print(f"    [retry {attempt}/3] {type(exc).__name__} : attente {wait}s", flush=True)
                _time.sleep(wait)
        if content is None:
            raise last_exc  # type: ignore[misc]
        try:
            cache_file.write_bytes(content)
        except Exception:
            pass

    return parse_xml_gz(content)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from", type=int, default=0, help="Index de depart (0 = debut)")
    parser.add_argument("--skip-cache", action="store_true", help="Re-telecharger (ignore cache)")
    args = parser.parse_args()

    t_start = time.time()

    baseline = list_files(BASELINE_DIR, r"pubmed26n\d+\.xml\.gz")
    updates = list_files(UPDATEFILES_DIR, r"pubmed26n\d+\.xml\.gz")

    all_files: list[tuple[str, str]] = []
    for f in baseline:
        all_files.append(("baseline", f))
    for f in updates:
        all_files.append(("updatefiles", f))

    print(f"Baseline    : {len(baseline)} fichiers")
    print(f"Updatefiles : {len(updates)} fichiers")
    print(f"Total       : {len(all_files)} fichiers")
    if args.start_from:
        print(f"Reprise a partir de l'index {args.start_from}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine le prochain part_idx disponible
    existing_parts = sorted(OUT_DIR.glob("part-*.parquet"))
    part_idx = len(existing_parts)
    if part_idx:
        print(f"{part_idx} parts existants, on continue a part-{part_idx:04d}.parquet")
        print()

    buf_pmids: list[int] = []
    buf_status: list[str | None] = []
    buf_pages: list[int | None] = []
    total_rows = 0
    n_failed = 0
    n_parsed = 0

    def flush():
        nonlocal buf_pmids, buf_status, buf_pages, part_idx, total_rows
        if not buf_pmids:
            return
        out = OUT_DIR / f"part-{part_idx:04d}.parquet"
        t = pa.Table.from_arrays(
            [
                pa.array(buf_pmids, type=pa.int64()),
                pa.array(buf_status, type=pa.string()),
                pa.array(buf_pages, type=pa.int32()),
            ],
            schema=SCHEMA,
        )
        pq.write_table(t, out, compression="zstd")
        print(f"    >> FLUSH {out.name} : {len(buf_pmids):,} lignes  (total {total_rows + len(buf_pmids):,})", flush=True)
        total_rows += len(buf_pmids)
        part_idx += 1
        buf_pmids.clear()
        buf_status.clear()
        buf_pages.clear()

    with httpx.Client(follow_redirects=True) as client:
        for i, (kind, fname) in enumerate(all_files, 1):
            if i <= args.start_from:
                continue

            base = BASELINE_DIR if kind == "baseline" else UPDATEFILES_DIR
            url = base + fname
            print(f"[{i}/{len(all_files)}] {kind}/{fname}", end=" ", flush=True)
            try:
                rows = download_and_parse(url, client, use_cache=not args.skip_cache)
                n_parsed += 1
                for pmid, status, n_pages in rows:
                    buf_pmids.append(pmid)
                    buf_status.append(status)
                    buf_pages.append(n_pages)
                print(f"-> {len(rows):,} PMIDs  (buffer {len(buf_pmids):,})", flush=True)
                if len(buf_pmids) >= FLUSH_EVERY:
                    flush()
            except Exception as exc:
                n_failed += 1
                print(f"-> [ERROR] {exc}", flush=True)

    flush()

    print()
    print(f"Fichiers traites : {n_parsed}/{len(all_files) - args.start_from}")
    print(f"Fichiers echoues : {n_failed}")
    print(f"Total lignes     : {total_rows:,}")
    print(f"Parts            : {part_idx}")
    print()

    # Merge
    print("Merge des parts...")
    parts = sorted(OUT_DIR.glob("part-*.parquet"))
    if not parts:
        print("Aucun part a merger")
        return 1

    writer = None
    n_final = 0
    for i, p in enumerate(parts, 1):
        t = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(OUT_FINAL, t.schema, compression="zstd")
        writer.write_table(t)
        n_final += t.num_rows
        if i % 5 == 0 or i == len(parts):
            print(f"  [{i}/{len(parts)}] {n_final:,} lignes")
    if writer:
        writer.close()

    print()
    print(f"=== Termine en {(time.time()-t_start)/3600:.1f}h ===")
    print(f"Total lignes : {n_final:,}")
    print(f"Fichier      : {OUT_FINAL}")
    print(f"Taille       : {OUT_FINAL.stat().st_size / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())