"""Configuration centralisee via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PRECITE_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Stockage S3-compatible (Cloudflare R2, MinIO, etc.)
    s3_endpoint: str = Field("http://localhost:9000")
    s3_access_key: str = Field("minioadmin")
    s3_secret_key: str = Field("minioadmin")
    s3_bucket: str = Field("precite")
    s3_region: str = Field("auto")

    # Source NCBI
    ftp_base_url: str = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
    ftp_update_url: str = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles"

    # Pipeline
    write_batch_size: int = 50_000
    row_group_size: int = 100_000
    log_level: str = "INFO"

    # MLflow
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "precite"
    mlflow_artifact_root: str = "mlartifacts"  # <-- artefacts separes du tracking

    # Chemins locaux
    config_dir: Path = Path("configs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def load_field_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else get_settings().config_dir / "fields.yml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)