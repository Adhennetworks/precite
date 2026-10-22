"""Dependances FastAPI : singleton ParquetStore."""

from __future__ import annotations

from functools import lru_cache

from precite.storage.parquet_store import ParquetStore


@lru_cache(maxsize=1)
def get_store() -> ParquetStore:
    """Singleton ParquetStore, partage entre toutes les requetes du process."""
    return ParquetStore("data")