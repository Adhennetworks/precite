"""Calcul des citations a plusieurs horizons depuis l'OCC.

Pour chaque article cite, on compte les citations recues dans la fenetre
[pub_year, pub_year + H] pour H = 1, 2, 3.

Utilise le year_map + le cache des citations.
"""

from __future__ import annotations

import csv
import io
import pickle
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from precite.utils.logging import get_logger

log = get_logger(__name__)

csv.field_size_limit(2**31 - 1)


def build_multi_horizon_cache(
    *,
    occ_zip: str | Path,
    year_map: dict[int, int],
    horizons: tuple[int, ...] = (1, 2, 3),
    snapshot_year: int = 2026,
    cache_path: str | Path = "data/icite/citations_multi_horizon.pkl",
    force_rebuild: bool = False,
) -> Path:
    """Scanne l'OCC une fois et compte les citations a plusieurs horizons.

    Sortie : dict { pmid: {1: n1, 2: n2, 3: n3} }
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        log.info("multi_horizon.cache.hit", path=str(cache_path))
        return cache_path

    occ_zip = Path(occ_zip)
    log.info(
        "multi_horizon.start",
        occ=str(occ_zip),
        horizons=list(horizons),
    )

    # Structure : { cited_pmid: {horizon: count} }
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    n_read = 0
    n_kept = 0
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

            for row in reader:
                n_read += 1
                try:
                    citing = int(row[citing_col])
                    cited = int(row[cited_col])
                except (ValueError, KeyError, TypeError):
                    continue

                citing_year = year_map.get(citing)
                pub_year = year_map.get(cited)
                if citing_year is None or pub_year is None:
                    continue

                delta = citing_year - pub_year
                if delta < 0:
                    continue

                # Incrementer chaque horizon couvert
                matched = False
                for h in horizons:
                    if delta <= h:
                        counts[cited][h] += 1
                        matched = True
                if matched:
                    n_kept += 1

                if n_read % 50_000_000 == 0:
                    elapsed = time.time() - t0
                    log.info(
                        "multi_horizon.progress",
                        read=n_read,
                        kept=n_kept,
                        distinct=len(counts),
                        elapsed_s=int(elapsed),
                    )

    log.info(
        "multi_horizon.scan_done",
        n_read=n_read,
        n_kept=n_kept,
        distinct=len(counts),
        elapsed_s=int(time.time() - t0),
    )

    # Convertir en dict normal
    result = {pmid: dict(h_dict) for pmid, h_dict in counts.items()}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = cache_path.stat().st_size / 1024 / 1024
    log.info("multi_horizon.written", path=str(cache_path), size_mb=size_mb)
    return cache_path


def load_multi_horizon_cache(
    cache_path: str | Path = "data/icite/citations_multi_horizon.pkl",
) -> dict[int, dict[int, int]]:
    with open(cache_path, "rb") as f:
        return pickle.load(f)