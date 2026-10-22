"""Enrichissement des features avec une cible de citations a horizon fixe.

Probleme : les citations accumulees ne sont pas comparables entre un article
publie en 2015 (11 ans de citations) et un article publie en 2023 (3 ans).
Solution : on tronque la fenetre de citation a N ans post-publication.

Sources NIH (snapshot mensuel, article Figshare 33437476 - 2026-08) :
- icite_metadata.zip            : metadonnees par article (pmid, year, ...)
- open_citation_collection.zip  : liens de citation (citing_pmid, cited_pmid)
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import httpx

from precite.utils.logging import get_logger

log = get_logger(__name__)


# Le CSV iCite contient des champs de plusieurs MB (abstracts, titres).
# Par defaut, csv.field_size_limit = 128 KB. On augmente.
csv.field_size_limit(2**31 - 1)


# --- Snapshot 2026-08 (coherent entre iCite et OCC) -------------------------
SNAPSHOT_LABEL = "2026-08"
ICITE_METADATA_URL = "https://ndownloader.figshare.com/files/68211877"
OCC_URL = "https://ndownloader.figshare.com/files/68211523"

# Annee de reference du snapshot : on exclut les articles trop recents
SNAPSHOT_YEAR = 2026


# --- Telechargement ---------------------------------------------------------

def _download(url: str, dest_path: Path, *, force: bool = False) -> Path:
    if dest_path.exists() and not force:
        log.info(
            "download.cached",
            path=str(dest_path),
            size_mb=dest_path.stat().st_size / 1024 / 1024,
        )
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("download.start", url=url, dest=str(dest_path))

    with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as r:
        r.raise_for_status()
        total = 0
        step = 200 * 1024 * 1024
        next_log = step
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                total += len(chunk)
                if total >= next_log:
                    log.info("download.progress", mb=total // (1024 * 1024))
                    next_log += step

    log.info(
        "download.done",
        path=str(dest_path),
        size_mb=dest_path.stat().st_size / 1024 / 1024,
    )
    return dest_path


def download_icite_metadata(dest_dir: str | Path, *, force: bool = False) -> Path:
    return _download(ICITE_METADATA_URL, Path(dest_dir) / "icite_metadata.zip", force=force)


def download_occ(dest_dir: str | Path, *, force: bool = False) -> Path:
    return _download(OCC_URL, Path(dest_dir) / "open_citation_collection.zip", force=force)


# --- iCite metadata -> year map ---------------------------------------------

def _iter_csv_in_zip(zip_path: Path, suffix: str):
    """Yield (filename, iterator of rows as dict) depuis un zip contenant un CSV."""
    with zipfile.ZipFile(zip_path) as z:
        candidates = [n for n in z.namelist() if n.endswith(suffix)]
        if not candidates:
            raise RuntimeError(
                f"{suffix} introuvable dans {zip_path}. Contenu : {z.namelist()[:10]}"
            )
        name = candidates[0]
        with z.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", newline="")
            yield name, csv.DictReader(text)


def load_year_map(
    icite_zip: str | Path,
    *,
    cache_path: str | Path | None = None,
    force_rebuild: bool = False,
) -> dict[int, int]:
    """Charge un dict {pmid_int: year_int} depuis icite_metadata.zip.

    Optimisations memoire :
    - cles en int (pas str) : ~2x moins de RAM
    - on ne garde que pmid + year

    Cache optionnel : ecrit un dict pickle (ou parquet) pour eviter de
    reparcourir les 13 GB a chaque run.
    """
    import pickle

    icite_zip = Path(icite_zip)

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force_rebuild:
            log.info("year_map.cache.hit", path=str(cache_path))
            with open(cache_path, "rb") as f:
                year_map = pickle.load(f)
            log.info("year_map.cache.loaded", n=len(year_map))
            return year_map

    log.info("year_map.build.start", zip=str(icite_zip))
    year_map: dict[int, int] = {}

    for name, reader in _iter_csv_in_zip(icite_zip, "icite_metadata.csv"):
        log.info("year_map.file", name=name)
        n = 0
        for row in reader:
            pmid = row.get("pmid")
            year = row.get("year")
            if not pmid or not year:
                continue
            try:
                year_map[int(pmid)] = int(year)
            except (ValueError, TypeError):
                continue
            n += 1
            if n % 5_000_000 == 0:
                log.info("year_map.progress", n=n)
        break

    log.info("year_map.build.done", n=len(year_map))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(year_map, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(
            "year_map.cache.written",
            path=str(cache_path),
            size_mb=cache_path.stat().st_size / 1024 / 1024,
        )

    return year_map


# --- OCC -> citations tronquees ---------------------------------------------

def _detect_columns(first_row: dict) -> tuple[str, str]:
    """Detecte les noms de colonnes 'citing' et 'cited' dans une ligne OCC."""
    keys = {k.lower(): k for k in first_row}
    citing = (
        keys.get("citing_pmid")
        or keys.get("citing")
        or keys.get("source_pmid")
    )
    cited = (
        keys.get("cited_pmid")
        or keys.get("cited")
        or keys.get("referenced")      # <-- ajout
        or keys.get("referenced_pmid")
        or keys.get("target_pmid")
    )
    if not citing or not cited:
        raise RuntimeError(
            f"colonnes citing/cited introuvables dans l'OCC. Cles : {list(first_row.keys())}"
        )
    return citing, cited

def build_truncated_citations(
    *,
    occ_zip: str | Path,
    year_map: dict[int, int],
    target_pmids: set[int],
    horizon_years: int = 3,
    snapshot_year: int = SNAPSHOT_YEAR,
) -> dict[int, int]:
    """Compte pour chaque pmid cible les citations recues dans les N ans.

    Regle : pub_year(cited) <= year(citing) <= pub_year(cited) + horizon.

    On ne garde que les articles cible dont on connait l'annee de publication.
    """
    occ_zip = Path(occ_zip)
    log.info(
        "truncated_citations.start",
        occ=str(occ_zip),
        n_targets=len(target_pmids),
        horizon=horizon_years,
    )

    citations: dict[int, int] = defaultdict(int)
    n_read = 0
    n_kept = 0
    n_cited_seen = 0

    with zipfile.ZipFile(occ_zip) as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv")]
        if not csvs:
            raise RuntimeError(f"aucun CSV dans {occ_zip}")
        csv_name = csvs[0]
        log.info("truncated_citations.file", name=csv_name)

        with z.open(csv_name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                raise RuntimeError("OCC vide")
            citing_col, cited_col = _detect_columns(
                {k: None for k in reader.fieldnames}
            )
            log.info(
                "truncated_citations.columns",
                citing=citing_col,
                cited=cited_col,
            )

            for row in reader:
                n_read += 1
                cited_raw = row.get(cited_col)
                citing_raw = row.get(citing_col)
                if not cited_raw or not citing_raw:
                    continue
                try:
                    cited = int(cited_raw)
                    citing = int(citing_raw)
                except (ValueError, TypeError):
                    continue

                if cited not in target_pmids:
                    continue
                n_cited_seen += 1

                citing_year = year_map.get(citing)
                if citing_year is None:
                    continue
                pub_year = year_map.get(cited)
                if pub_year is None:
                    continue

                if pub_year <= citing_year <= pub_year + horizon_years:
                    citations[cited] += 1
                    n_kept += 1

                if n_read % 10_000_000 == 0:
                    log.info(
                        "truncated_citations.progress",
                        read=n_read,
                        hits=n_cited_seen,
                        kept=n_kept,
                        distinct_cited=len(citations),
                    )

    log.info(
        "truncated_citations.done",
        n_read=n_read,
        n_hits=n_cited_seen,
        n_kept=n_kept,
        distinct_cited=len(citations),
    )
    return dict(citations)


# --- Pipeline complet -------------------------------------------------------

def enrich_parquet(
    *,
    input_path: str,
    output_path: str,
    icite_zip: str | Path,
    occ_zip: str | Path,
    horizon_years: int = 3,
    snapshot_year: int = SNAPSHOT_YEAR,
    cache_year_map: str | Path | None = "data/icite/year_map.pkl",
    target_year_max: int | None = None,
) -> dict[str, int | float]:
    """Pipeline complet : features + citations tronquees a horizon N.

    - target_year_max : si fourni, filtre les articles publies APRES cette annee.
      Recommande : snapshot_year - horizon_years (ex: 2026 - 3 = 2023).
      Cela garantit que tous les articles ont au moins N ans d'historique.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Features
    log.info("enrich.read_features", input=str(in_path))
    df = pq.read_table(in_path).to_pandas()
    df["pmid"] = df["pmid"].astype("int64")
    log.info("enrich.features_loaded", rows=len(df))

    # 2. Year map (avec cache)
    year_map = load_year_map(icite_zip, cache_path=cache_year_map)

    # 3. Filtrer les cibles : uniquement articles publies avant cutoff
    if target_year_max is None:
        target_year_max = snapshot_year - horizon_years

    df["icite_year"] = df["pmid"].map(year_map)
    n_before = len(df)
    df = df[df["icite_year"].notna()].copy()
    df["icite_year"] = df["icite_year"].astype("int32")
    df = df[df["icite_year"] <= target_year_max].copy()
    log.info(
        "enrich.filtered_by_year",
        before=n_before,
        after=len(df),
        cutoff=target_year_max,
    )

    if df.empty:
        raise RuntimeError(
            f"aucun article avec year <= {target_year_max}. "
            "Verifier que le Parquet contient bien des articles historiques."
        )

    # 4. Citations tronquees
    target_pmids = set(df["pmid"].astype(int).tolist())
    citations = build_truncated_citations(
        occ_zip=occ_zip,
        year_map=year_map,
        target_pmids=target_pmids,
        horizon_years=horizon_years,
        snapshot_year=snapshot_year,
    )

    # 5. Jointure
    df[f"citations_{horizon_years}y"] = (
        df["pmid"].map(citations).fillna(0).astype("int32")
    )
    col = f"citations_{horizon_years}y"

    # 6. Ecrire
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path, compression="zstd")

    summary = {
        "rows": len(df),
        "with_citations": int((df[col] > 0).sum()),
        "total_citations": int(df[col].sum()),
        "mean_citations": float(df[col].mean()),
        "median_citations": float(df[col].median()),
        "max_citations": int(df[col].max()),
        "horizon_years": horizon_years,
        "target_year_max": target_year_max,
    }
    log.info("enrich.done", output=str(out_path), **summary)
    return summary