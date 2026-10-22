"""Cache global des citations tronquees a horizon N ans.

Strategie :
1. Scanner l'OCC UNE FOIS et sauvegarder {pmid: count_3y} en pickle.
2. A chaque nouveau batch de features, on fait une jointure O(1) sur ce cache.

Principe de la cible tronquee :
- Un article publie en 2015 a eu 11 ans pour etre cite (trop).
- Un article publie en 2023 n'a eu que 3 ans (trop court).
- Pour comparer, on tronque a N ans post-publication.

Cible valide :
- Si pub_year <= snapshot_year - horizon - 1 : la cible est fiable.
- Sinon : la cible est NaN, l'article est exclu de l'entrainement
  (mais reste predictible via l'API).
"""

from __future__ import annotations

import csv
import io
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

from precite.utils.logging import get_logger

log = get_logger(__name__)

# Le CSV OCC contient des champs potentiellement longs (rare).
# Valeur max int32 pour Windows.
csv.field_size_limit(2**31 - 1)


# ─── Construction du cache ──────────────────────────────────────────────────

def build_global_citations_cache(
    *,
    occ_zip: str | Path,
    year_map: dict[int, int],
    horizon_years: int = 3,
    cache_path: str | Path = "data/icite/citations_3y_cache.pkl",
    force_rebuild: bool = False,
) -> Path:
    """Scanne l'OCC une seule fois et sauvegarde {pmid_cite: count_3y}.

    IMPORTANT : on ne filtre PAS les articles recents ici. Le cache contient
    les counts pour TOUS les PMIDs cites. C'est la couche d'attache qui
    decidera si un article est eligible a l'entrainement (via has_enough_history).
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        log.info("cache.hit", path=str(cache_path))
        return cache_path

    occ_zip = Path(occ_zip)

    log.info(
        "cache.build.start",
        occ=str(occ_zip),
        horizon=horizon_years,
    )

    citations: dict[int, int] = defaultdict(int)
    n_read = 0
    n_kept = 0

    with zipfile.ZipFile(occ_zip) as z:
        csvs = [n for n in z.namelist() if n.endswith(".csv")]
        if not csvs:
            raise RuntimeError(f"aucun CSV dans {occ_zip}")
        csv_name = csvs[0]
        log.info("cache.build.file", name=csv_name)

        with z.open(csv_name) as f:
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
                raise RuntimeError(f"colonnes citing/cited introuvables : {header}")

            log.info(
                "cache.build.columns",
                citing=citing_col,
                cited=cited_col,
            )

            for row in reader:
                n_read += 1
                try:
                    citing = int(row[citing_col])
                    cited = int(row[cited_col])
                except (ValueError, KeyError, TypeError):
                    continue

                citing_year = year_map.get(citing)
                if citing_year is None:
                    continue
                pub_year = year_map.get(cited)
                if pub_year is None:
                    continue

                # Fenetre : pub_year <= citing_year <= pub_year + horizon
                if pub_year <= citing_year <= pub_year + horizon_years:
                    citations[cited] += 1
                    n_kept += 1

                if n_read % 50_000_000 == 0:
                    log.info(
                        "cache.build.progress",
                        read=n_read,
                        kept=n_kept,
                        distinct=len(citations),
                    )

    log.info(
        "cache.build.scan_done",
        n_read=n_read,
        n_kept=n_kept,
        distinct=len(citations),
    )

    # Ecriture du cache complet (sans filtre)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dict(citations), f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(
        "cache.build.written",
        path=str(cache_path),
        n=len(citations),
        size_mb=cache_path.stat().st_size / 1024 / 1024,
    )
    return cache_path


# ─── Chargement du cache ────────────────────────────────────────────────────

def load_citations_cache(
    cache_path: str | Path = "data/icite/citations_3y_cache.pkl",
) -> dict[int, int]:
    """Charge le dict {pmid: count_3y} depuis le pickle."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"cache introuvable : {cache_path}")
    log.info("cache.load", path=str(cache_path))
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    log.info("cache.loaded", n=len(data))
    return data


# ─── Attache des citations a un Parquet ─────────────────────────────────────

def attach_citations(
    features_parquet: str | Path,
    output_parquet: str | Path,
    *,
    cache_path: str | Path = "data/icite/citations_3y_cache.pkl",
    weighted_cache_path: str | Path | None = "data/icite/weighted_citations_3y_cache.pkl",
    cache: dict[int, int] | None = None,
    weighted_cache: dict[int, float] | None = None,   # <-- cle
    horizon_years: int = 3,
    snapshot_year: int = 2026,
    col_name: str = "citations_3y",
    weighted_col_name: str = "weighted_citations_3y",
    flag_col: str = "has_enough_history",
) -> dict:
    """Joint les caches raw ET weighted avec un Parquet de features.

    Ajoute 3 colonnes :
    - citations_3y           : nb de citations dans la fenetre 3 ans
    - weighted_citations_3y  : somme des SJR des articles citants
    - has_enough_history     : True si l'article a eu 3 ans pleins
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    features_parquet = Path(features_parquet)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    df = pq.read_table(features_parquet).to_pandas()
    df["pmid"] = df["pmid"].astype("int64")

    # Charger les caches si pas fournis
    if cache is None:
        cache = load_citations_cache(cache_path)

    if weighted_col_name and weighted_cache is None and weighted_cache_path:
        wp = Path(weighted_cache_path)
        if wp.exists():
            log.info("cache.load_weighted", path=str(wp))
            with open(wp, "rb") as f:
                weighted_cache = pickle.load(f)
            log.info("cache.loaded_weighted", n=len(weighted_cache))
        else:
            log.warning("cache.weighted_missing", path=str(wp))

    cutoff_year = snapshot_year - horizon_years - 1

    # Citations brutes
    count_raw = df["pmid"].map(cache).fillna(0).astype("int32")
    pub_year = df["year"].astype("int32")
    has_history = (pub_year <= cutoff_year).to_numpy()

    citations = count_raw.where(has_history, other=np.nan).astype("Float64")
    df[col_name] = citations
    df[flag_col] = has_history

    # Citations ponderees
    if weighted_cache is not None:
        weighted_raw = df["pmid"].map(weighted_cache).fillna(0.0).astype("float32")
        weighted = weighted_raw.where(has_history, other=np.nan).astype("Float64")
        df[weighted_col_name] = weighted

    # Ecriture
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_parquet, compression="zstd")

    n_valid = int(has_history.sum())
    n_with_cit = int(((count_raw > 0) & has_history).sum())

    summary = {
        "rows": len(df),
        "with_history": n_valid,
        "without_history": len(df) - n_valid,
        "with_citations": n_with_cit,
        "total_citations": int(count_raw[has_history].sum()),
        "mean": float(count_raw[has_history].mean()) if n_valid > 0 else 0.0,
        "median": float(count_raw[has_history].median()) if n_valid > 0 else 0.0,
        "max": int(count_raw[has_history].max()) if n_valid > 0 else 0,
        "cutoff_year": cutoff_year,
    }

    if weighted_cache is not None:
        w = weighted_raw[has_history]
        summary["weighted_mean"] = float(w.mean()) if n_valid > 0 else 0.0
        summary["weighted_median"] = float(w.median()) if n_valid > 0 else 0.0
        summary["weighted_max"] = float(w.max()) if n_valid > 0 else 0.0

    log.info("attach.done", output=str(output_parquet), **summary)
    return summary