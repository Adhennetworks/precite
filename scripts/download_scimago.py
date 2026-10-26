"""Telecharge SCImago Journal Rank depuis le miroir GitHub.

Source : https://github.com/Michael-E-Rose/SCImagoJournalRankIndicators

Format ISSN :
- SCImago : sans tiret, parfois plusieurs separes par virgule ("00284866, 19372213")
- PubMed  : avec tiret ("0028-4793")
-> On normalise tout en retirant les caracteres non-alphanumeriques.

Sortie : data/scimago/issn_sjr_by_year.pkl
Format : { issn_normalise: { year: sjr, ... } }
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "scimago" / "raw"
OUTPUT = ROOT / "data" / "scimago" / "issn_sjr_by_year.pkl"

ALL_CSV_URL = (
    "https://raw.githubusercontent.com/Michael-E-Rose/"
    "SCImagoJournalRankIndicators/master/all.csv"
)

ISSN_NORMALIZE = re.compile(r"[^0-9X]")


def normalize_issn(value: str) -> str:
    """Retire tirets, espaces, et met en majuscule. Ex: '0028-4793' -> '00284793'."""
    return ISSN_NORMALIZE.sub("", value.upper())


def download_csv(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DIR / "scimago_all.csv"

    if cache_file.exists() and cache_file.stat().st_size > 1_000_000 and not force:
        size_mb = cache_file.stat().st_size // 1024 // 1024
        print(f"CSV deja present : {cache_file} ({size_mb} MB)")
        return cache_file

    print(f"Downloading {ALL_CSV_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Precite/1.0)"}
    with httpx.stream(
        "GET", ALL_CSV_URL, timeout=300.0, follow_redirects=True, headers=headers
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        print(f"  Taille attendue : {total // 1024 // 1024} MB")

        received = 0
        with open(cache_file, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    received += len(chunk)
                    if received % (50 * 1024 * 1024) < (1 << 20):
                        print(f"  {received // 1024 // 1024} MB")

    size_mb = cache_file.stat().st_size // 1024 // 1024
    print(f"OK : {cache_file} ({size_mb} MB)")
    return cache_file


def build_map(csv_path: Path) -> dict[str, dict[int, float]]:
    """Construit { issn_normalise: { year: sjr } }."""
    print("Lecture du CSV...")

    # Detection de colonnes (insensible a la casse)
    header = pd.read_csv(csv_path, nrows=0)
    cols_lower = {c.lower(): c for c in header.columns}

    issn_col = cols_lower.get("issn")
    year_col = cols_lower.get("year")
    sjr_col = cols_lower.get("sjr")
    src_col = cols_lower.get("sourceid")

    if not all([issn_col, year_col, sjr_col]):
        raise RuntimeError(
            f"Colonnes manquantes. Dispo : {list(header.columns)}"
        )

    print(f"  Colonnes : ISSN={issn_col}, Year={year_col}, SJR={sjr_col}, Sourceid={src_col}")

    # Lecture par chunks
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=200_000, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"  {len(df):,} lignes chargees")

    # Dedup sur (Sourceid, year) si dispo, sinon (ISSN, year)
    dedup_keys = [src_col, year_col] if src_col else [issn_col, year_col]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys).copy()
    print(f"  Dedup : {before:,} -> {len(df):,}")

    # Construire le mapping
    mapping: dict[str, dict[int, float]] = {}
    n_issn_total = 0
    n_skipped_nan = 0

    for _, row in df.iterrows():
        issn_raw = row[issn_col]
        year_val = row[year_col]
        sjr_val = row[sjr_col]

        if pd.isna(issn_raw) or pd.isna(year_val) or pd.isna(sjr_val):
            n_skipped_nan += 1
            continue

        try:
            year = int(year_val)
            sjr = float(sjr_val)
        except (ValueError, TypeError):
            continue

        if year < 1999 or year > 2030 or sjr < 0:
            continue

        # Multi-ISSN separes par virgule, avec ou sans tiret
        for raw in str(issn_raw).split(","):
            issn = normalize_issn(raw)
            if len(issn) != 8:
                continue
            mapping.setdefault(issn, {})[year] = sjr
            n_issn_total += 1

    print(f"  {n_issn_total:,} entrees ISSN x annee")
    print(f"  {n_skipped_nan:,} lignes ignorees (NaN)")

    return mapping


def main() -> int:
    print("Telechargement SCImago (miroir GitHub)")
    print()

    csv_path = download_csv()
    mapping = build_map(csv_path)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(mapping, f)

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print()
    print("=== Termine ===")
    print(f"ISSN distincts : {len(mapping):,}")
    print(f"Taille         : {size_mb:.1f} MB")
    print(f"Fichier        : {OUTPUT}")

    # Verification (sans tirets, comme SCImago)
    samples = [
        ("00284793", "NEJM"),
        ("14764687", "Nature"),
        ("00368075", "Science"),
        ("00928674", "Cell"),
        ("01406736", "Lancet"),
        ("10959203", "Science (alt)"),
    ]
    print()
    print("Verification sur des journaux connus :")
    for issn, name in samples:
        sjrs = mapping.get(issn, {})
        if sjrs:
            recent = sorted(sjrs.items())[-5:]
            vals = ", ".join(f"{y}:{v:.2f}" for y, v in recent)
            print(f"  {name:15s} ({issn}) : {vals}")
        else:
            print(f"  {name:15s} ({issn}) : INTROUVABLE")

    # Compter combien de valeurs par annee
    print()
    print("Nombre d'ISSN par annee (echantillon) :")
    from collections import Counter
    year_counts = Counter()
    for sjrs in mapping.values():
        for y in sjrs:
            year_counts[y] += 1
    for y in sorted(year_counts)[-10:]:
        print(f"  {y} : {year_counts[y]:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())