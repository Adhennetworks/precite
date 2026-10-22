"""Construit le cache des citations ponderees par SJR.

Pour chaque citation (citing -> cited) de l'OCC :
- si cited est dans nos PMIDs cibles (year <= cutoff)
- si citing_year in [pub_year(cited), pub_year(cited) + horizon]
- on ajoute SJR(ISSN du journal citant, citing_year) a weighted[cited]

Sources necessaires :
- data/icite/open_citation_collection.zip
- data/icite/year_map.pkl              (PMID -> annee)
- data/scimago/issn_sjr_by_year.pkl    (ISSN -> {annee: SJR})
- data/features/**/*.parquet           (PMID -> ISSN)

Sortie : data/icite/weighted_citations_3y_cache.pkl
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq

ISSN_NORMALIZE = re.compile(r"[^0-9X]")


def normalize_issn(value: str) -> str:
    return ISSN_NORMALIZE.sub("", str(value).upper())


def load_issn_map() -> dict[int, str]:
    """Charge { pmid: issn_normalise } depuis les features."""
    print("Construction de PMID -> ISSN depuis data/features...")
    mapping: dict[int, str] = {}

    files = sorted((ROOT / "data" / "features").rglob("*.parquet"))
    print(f"  {len(files)} fichiers Parquet")

    for i, f in enumerate(files, 1):
        try:
            df = pq.read_table(f, columns=["pmid", "journal_issn"]).to_pandas()
            for _, row in df.iterrows():
                issn_raw = row["journal_issn"]
                if issn_raw and isinstance(issn_raw, str):
                    issn = normalize_issn(issn_raw)
                    if len(issn) == 8:
                        mapping[int(row["pmid"])] = issn
        except Exception as e:
            print(f"  [!] {f.name} : {e}")

        if i % 200 == 0:
            print(f"  [{i}/{len(files)}] {len(mapping):,} PMIDs avec ISSN")

    print(f"  Total : {len(mapping):,} PMIDs avec ISSN")
    return mapping


def load_scimago() -> dict[str, dict[int, float]]:
    print("Chargement SCImago...")
    path = ROOT / "data" / "scimago" / "issn_sjr_by_year.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data):,} ISSN distincts")

    # Verifications de type
    if not data:
        raise RuntimeError("SCImago vide")
    sample_key = next(iter(data))
    sample_val = data[sample_key]
    if not isinstance(sample_val, dict):
        raise RuntimeError(
            f"SCImago : valeur attendue dict, recu {type(sample_val).__name__}"
        )
    sample_year = next(iter(sample_val))
    if not isinstance(sample_year, int):
        raise RuntimeError(
            f"SCImago : annee attendue int, recu {type(sample_year).__name__} = {sample_year!r}"
        )
    print(f"  Structure OK (ex: {sample_key} -> {dict(list(sample_val.items())[:2])})")
    return data


def load_year_map() -> dict[int, int]:
    print("Chargement year_map...")
    path = ROOT / "data" / "icite" / "year_map.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data):,} PMIDs")

    if data:
        k = next(iter(data))
        v = data[k]
        print(f"  Structure OK (ex: {k} -> {v} type={type(v).__name__})")
    return data


def build_weighted_cache(
    *,
    occ_zip: Path,
    year_map: dict[int, int],
    issn_map: dict[int, str],
    scimago: dict[str, dict[int, float]],
    horizon_years: int = 3,
    snapshot_year: int = 2026,
    debug_first_n: int = 20,
) -> dict[int, float]:
    csv.field_size_limit(2**31 - 1)

    cutoff_year = snapshot_year - horizon_years - 1
    target_pmids = {pmid for pmid, y in year_map.items() if y <= cutoff_year}
    print(f"  {len(target_pmids):,} PMIDs cibles (year <= {cutoff_year})")

    # Utiliser un dict normal (pas defaultdict)
    weighted: dict[int, float] = {}

    n_read = 0
    n_kept = 0
    n_no_issn = 0
    n_no_sjr = 0
    n_exact_year = 0
    n_fallback_year = 0
    n_weighted_added = 0

    # Compteurs de debug
    missing_issns = Counter()
    debug_shown = 0
    t0 = time.time()

    with zipfile.ZipFile(occ_zip) as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv")]
        with z.open(csvs[0]) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            header = reader.fieldnames or []
            citing_col = next(
                (k for k in header if k.lower() in ("citing", "citing_pmid", "source_pmid")),
                None,
            )
            cited_col = next(
                (k for k in header if k.lower() in ("referenced", "cited", "cited_pmid", "target_pmid")),
                None,
            )
            if not citing_col or not cited_col:
                raise RuntimeError(f"Colonnes introuvables : {header}")
            print(f"  Colonnes : citing={citing_col}, cited={cited_col}")

            for row in reader:
                n_read += 1
                try:
                    citing = int(row[citing_col])
                    cited = int(row[cited_col])
                except (ValueError, KeyError, TypeError):
                    continue

                if cited not in target_pmids:
                    continue

                citing_year = year_map.get(citing)
                if citing_year is None:
                    continue
                pub_year = year_map.get(cited)
                if pub_year is None:
                    continue
                if not (pub_year <= citing_year <= pub_year + horizon_years):
                    continue

                n_kept += 1

                citing_issn = issn_map.get(citing)
                if citing_issn is None:
                    n_no_issn += 1
                    continue

                sjrs = scimago.get(citing_issn)
                if not sjrs:
                    n_no_sjr += 1
                    missing_issns[citing_issn] += 1
                    continue

                # Chercher SJR de l'annee exacte, sinon fallback
                sjr = sjrs.get(citing_year)
                if sjr is not None:
                    n_exact_year += 1
                else:
                    closest = min(sjrs.keys(), key=lambda y: abs(y - citing_year))
                    sjr = sjrs[closest]
                    n_fallback_year += 1

                if sjr is None or sjr <= 0:
                    continue

                # Ajout explicite (pas de defaultdict)
                if cited not in weighted:
                    weighted[cited] = 0.0
                weighted[cited] += float(sjr)
                n_weighted_added += 1

                # Debug : afficher les premiers cas reussis
                if debug_shown < debug_first_n:
                    print(
                        f"    OK citing={citing} cited={cited} "
                        f"cy={citing_year} py={pub_year} "
                        f"issn={citing_issn} sjr={sjr:.3f}"
                    )
                    debug_shown += 1

                if n_read % 50_000_000 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"  {n_read:,} lus | {n_kept:,} kept | "
                        f"{len(weighted):,} distincts | "
                        f"no_issn={n_no_issn:,} no_sjr={n_no_sjr:,} "
                        f"exact={n_exact_year:,} fallback={n_fallback_year:,} | "
                        f"{elapsed:.0f}s"
                    )

    print()
    print(f"  Fini en {time.time() - t0:.0f}s")
    print(f"  Citations brutes retenues   : {n_kept:,}")
    print(f"  Sans ISSN (ignorees)        : {n_no_issn:,}")
    print(f"  ISSN absent de SCImago      : {n_no_sjr:,}")
    print(f"  SJR trouve annee exacte     : {n_exact_year:,}")
    print(f"  SJR trouve fallback annee   : {n_fallback_year:,}")
    print(f"  Contributions ajoutees      : {n_weighted_added:,}")
    print(f"  Articles distincts          : {len(weighted):,}")

    if missing_issns:
        print()
        print("  Top 15 ISSN absents de SCImago (par volume) :")
        for issn, count in missing_issns.most_common(15):
            print(f"    {issn} : {count:,} citations")

    return weighted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug-first-n", type=int, default=20,
                        help="Afficher les N premieres citations reussies")
    args = parser.parse_args()

    output = ROOT / "data" / "icite" / "weighted_citations_3y_cache.pkl"

    if output.exists() and not args.force:
        size_mb = output.stat().st_size / 1024 / 1024
        print(f"Cache deja present : {output} ({size_mb:.1f} MB)")
        print("Utiliser --force pour reconstruire.")
        return 0

    issn_map = load_issn_map()
    scimago = load_scimago()
    year_map = load_year_map()

    print()
    print("Scan de l'OCC...")
    weighted = build_weighted_cache(
        occ_zip=ROOT / "data" / "icite" / "open_citation_collection.zip",
        year_map=year_map,
        issn_map=issn_map,
        scimago=scimago,
        debug_first_n=args.debug_first_n,
    )

    if not weighted:
        print()
        print("!!! ATTENTION : weighted est VIDE !!!")
        print("Verifier les logs de debug ci-dessus.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(weighted, f)

    size_mb = output.stat().st_size / 1024 / 1024
    print()
    print(f"=== Ecrit : {output} ({size_mb:.1f} MB) ===")

    vals = list(weighted.values())
    if vals:
        import statistics
        print()
        print("Stats weighted citations :")
        print(f"  count  : {len(vals):,}")
        print(f"  mean   : {statistics.mean(vals):.4f}")
        print(f"  median : {statistics.median(vals):.4f}")
        print(f"  max    : {max(vals):.4f}")
        # Distribution
        n_zero = sum(1 for v in vals if v == 0)
        print(f"  zeros  : {n_zero:,}")
        nonzero = [v for v in vals if v > 0]
        if nonzero:
            print(f"  parmi non-zero ({len(nonzero):,}):")
            print(f"    mean   : {statistics.mean(nonzero):.4f}")
            print(f"    median : {statistics.median(nonzero):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())