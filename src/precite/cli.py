"""CLI Precite.

Sous-commandes :
    ingest   : PubMed .xml.gz -> features Parquet
    storage  : verification S3
    train    : entrainement (simple, rolling, rolling-cv)
    predict  : prediction batch sur les features ingerees
    enrich   : enrichissement des features (citations iCite)
    serve    : API FastAPI + app Streamlit
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from precite import __version__
from precite.config import get_settings
from precite.ingest.pipeline import ingest_many
from precite.storage.s3 import ensure_bucket, get_s3_client
from precite.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="precite",
    help="Precite - citation prediction from pre-submission features.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
log = get_logger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"precite [bold cyan]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True)
    ] = False,
    log_level: Annotated[str, typer.Option("--log-level", "-l")] = "INFO",
    json_logs: Annotated[bool, typer.Option("--json-logs/--no-json-logs")] = True,
) -> None:
    configure_logging(level=log_level, json_output=json_logs)


# ─── ingest ────────────────────────────────────────────────────────────────

ingest_app = typer.Typer(help="Ingestion PubMed -> Parquet sur S3.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("urls")
def ingest_urls_cmd(
    urls: Annotated[list[str], typer.Option("--url", "-u", help="URL(s) .xml.gz")],
    local_dir: Annotated[str | None, typer.Option("--local-dir", help="Ecriture locale")] = None,
) -> None:
    if not urls:
        console.print("[red]Aucune URL fournie.[/]")
        raise typer.Exit(code=1)
    if local_dir:
        console.print(f"[yellow]Mode local[/] : ecriture dans {local_dir}")
    else:
        console.print("[green]Streaming pur[/] : rien sur disque")
    results = ingest_many(urls, local_dir=local_dir)
    _print_ingest_summary(results)


@ingest_app.command("latest")
def ingest_latest_cmd(
    n: Annotated[int, typer.Option("--n", "-n")] = 1,
    update: Annotated[bool, typer.Option("--update/--baseline")] = True,
) -> None:
    s = get_settings()
    base = s.ftp_update_url if update else s.ftp_base_url
    names = _list_recent_pubmed_files(base, n)
    if not names:
        console.print(f"[red]Aucun fichier trouve sur {base}[/]")
        raise typer.Exit(code=1)
    urls = [f"{base}/{name}" for name in names]
    console.print(f"Ingestion de {len(urls)} fichier(s)...")
    results = ingest_many(urls)
    _print_ingest_summary(results)


# ─── storage ───────────────────────────────────────────────────────────────

storage_app = typer.Typer(help="Verifications de stockage S3.")
app.add_typer(storage_app, name="storage")


@storage_app.command("check")
def storage_check_cmd() -> None:
    s = get_settings()
    ensure_bucket()
    client = get_s3_client()
    resp = client.list_objects_v2(Bucket=s.s3_bucket, MaxKeys=1)
    console.print(
        f"[green]OK[/] bucket=[cyan]{s.s3_bucket}[/] "
        f"endpoint={s.s3_endpoint} objects>0={resp.get('KeyCount', 0) > 0}"
    )


# ─── train ─────────────────────────────────────────────────────────────────

train_app = typer.Typer(help="Entrainement + tracking MLflow.")
app.add_typer(train_app, name="train")


@train_app.command("run")
def train_run_cmd(
    data: Annotated[str, typer.Option("--data", "-d", help="Source Parquet")],
    config: Annotated[str, typer.Option("--config", "-c", help="Chemin du YAML de modele")],
    cutoff_year: Annotated[int, typer.Option("--cutoff-year", help="Annee de coupure train/test")],
    target: Annotated[str, typer.Option("--target", "-t")] = "citations_3y",
    register_as: Annotated[str | None, typer.Option("--register-as")] = None,
) -> None:
    """Entraine un modele Precite (split temporel fixe)."""
    from precite.training.trainer import train_from_config

    console.print(f"[cyan]Modele:[/] {config}")
    console.print(f"[cyan]Donnees:[/] {data}")
    console.print(f"[cyan]Cutoff:[/] {cutoff_year}")

    result = train_from_config(
        data_source=data,
        model_config_path=config,
        cutoff_year=cutoff_year,
        target=target,
        register_as=register_as,
    )

    table = Table(title="Resultats")
    table.add_column("Metrique", style="cyan")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Test", justify="right", style="yellow")
    for k in ("mae", "mae_median", "msep", "r2", "spearman"):
        tv = result["train_metrics"].get(k)
        ev = result["test_metrics"].get(k)
        table.add_row(
            k,
            f"{tv:.4f}" if tv is not None else "-",
            f"{ev:.4f}" if ev is not None else "-",
        )
    for k in ("aic", "bic", "nll"):
        tv = result["train_metrics"].get(k)
        if tv is not None:
            table.add_row(k, f"{tv:.2f}", "-")
    console.print(table)
    console.print(f"[bold green]Run MLflow:[/] {result['run_id']}")


@train_app.command("rolling")
def train_rolling_cmd(
    data: Annotated[str, typer.Option("--data", "-d")],
    config: Annotated[str, typer.Option("--config", "-c")],
    window: Annotated[int, typer.Option("--window", "-w", help="Fenetre glissante (annees)")] = 10,
    target: Annotated[str, typer.Option("--target", "-t")] = "citations_3y",
    year_min: Annotated[int | None, typer.Option("--year-min")] = None,
    year_max: Annotated[int | None, typer.Option("--year-max")] = None,
    register_as: Annotated[str | None, typer.Option("--register-as")] = None,
) -> None:
    """Entrainement rolling simple (fenetre glissante, test sur l'annee suivante)."""
    from precite.training.trainer import train_rolling_from_config

    console.print(f"[cyan]Modele :[/]   {config}")
    console.print(f"[cyan]Donnees :[/]  {data}")
    console.print(f"[cyan]Fenetre :[/]  {window} ans")
    console.print()

    result = train_rolling_from_config(
        data_source=data,
        model_config_path=config,
        window_years=window,
        target=target,
        year_min=year_min,
        year_max=year_max,
        register_as=register_as,
    )

    console.print()
    console.print(f"[bold green]{result['n_folds']} folds[/]")
    console.print()

    agg = result["aggregate_metrics"]
    table = Table(title=f"Metriques agregees (fenetre {window} ans)")
    table.add_column("Metrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    for k in ("mae", "mae_median", "mae_p99", "msep", "r2", "spearman"):
        v = agg.get(k)
        if v is not None:
            table.add_row(k, f"{v:.4f}")
    console.print(table)

    table2 = Table(title="Detail par fold")
    table2.add_column("Annee", style="cyan")
    table2.add_column("Train", justify="right")
    table2.add_column("Test", justify="right")
    table2.add_column("MAE", justify="right", style="green")
    table2.add_column("MAE med", justify="right", style="green")
    table2.add_column("Spearman", justify="right", style="yellow")
    for f in result["fold_metrics"]:
        table2.add_row(
            str(f["test_year"]),
            f"{f['n_train']:,}",
            f"{f['n_test']:,}",
            f"{f.get('mae', 0):.2f}",
            f"{f.get('mae_median', 0):.2f}",
            f"{f.get('spearman', 0):.4f}",
        )
    console.print(table2)

    console.print(f"[bold green]Run MLflow :[/] {result['run_id']}")


@train_app.command("rolling-cv")
def train_rolling_cv_cmd(
    data: Annotated[str, typer.Option("--data", "-d")],
    config: Annotated[str, typer.Option("--config", "-c")],
    window: Annotated[int, typer.Option("--window", "-w")] = 10,
    inner_folds: Annotated[int, typer.Option("--inner-folds")] = 5,
    target: Annotated[str, typer.Option("--target", "-t")] = "citations_3y",
    year_min: Annotated[int | None, typer.Option("--year-min")] = None,
    year_max: Annotated[int | None, typer.Option("--year-max")] = None,
    register_as: Annotated[str | None, typer.Option("--register-as")] = None,
) -> None:
    """Rolling + CV interne avec expanding window temporel."""
    from precite.training.trainer import train_rolling_cv_from_config

    console.print(f"[cyan]Modele :[/]         {config}")
    console.print(f"[cyan]Donnees :[/]        {data}")
    console.print(f"[cyan]Fenetre :[/]        {window} ans")
    console.print(f"[cyan]Folds internes :[/] {inner_folds}")
    console.print("[cyan]Filtre :[/]         research articles uniquement")
    console.print()

    result = train_rolling_cv_from_config(
        data_source=data,
        model_config_path=config,
        window_years=window,
        n_inner_folds=inner_folds,
        target=target,
        year_min=year_min,
        year_max=year_max,
        register_as=register_as,
    )

    console.print()
    console.print(f"[bold green]{result['n_outer_folds']} outer x {result['n_inner_folds']} inner[/]")
    console.print()

    # Outer agrege
    agg = result["outer_aggregate"]
    table = Table(title=f"Outer rolling ({window} ans)")
    table.add_column("Metrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    for k in ("mae", "mae_median", "mae_p99", "msep", "r2", "spearman"):
        v = agg.get(k)
        if v is not None:
            table.add_row(k, f"{v:.4f}")
    console.print(table)

    # Inner CV agrege
    inner = result["inner_cv_aggregate"]
    table2 = Table(title=f"CV interne {inner_folds}-fold (expanding window)")
    table2.add_column("Metrique", style="cyan")
    table2.add_column("Mean", justify="right", style="green")
    table2.add_column("Std", justify="right", style="yellow")
    for k in ("mae", "mae_median", "msep", "r2", "spearman"):
        m = inner.get(f"{k}_mean")
        s = inner.get(f"{k}_std_between_folds")
        if m is not None:
            table2.add_row(k, f"{m:.4f}", f"{s:.4f}" if s is not None else "-")
    console.print(table2)

    # Detail outer par annee
    table3 = Table(title="Detail outer par annee")
    table3.add_column("Annee", style="cyan")
    table3.add_column("Train", justify="right")
    table3.add_column("Test", justify="right")
    table3.add_column("MAE", justify="right", style="green")
    table3.add_column("MAE med", justify="right", style="green")
    table3.add_column("Spearman", justify="right", style="yellow")
    for f in result["outer_folds"]:
        om = f["outer_metrics"]
        table3.add_row(
            str(f["test_year"]),
            f"{f['n_train']:,}",
            f"{f['n_test']:,}",
            f"{om.get('mae', 0):.2f}",
            f"{om.get('mae_median', 0):.2f}",
            f"{om.get('spearman', 0):.4f}",
        )
    console.print(table3)
    console.print(f"[bold green]Run MLflow :[/] {result['run_id']}")


# ─── predict ───────────────────────────────────────────────────────────────

predict_app = typer.Typer(help="Prediction batch sur les features ingerees.")
app.add_typer(predict_app, name="predict")


@predict_app.command("batch")
def predict_batch_cmd(
    source: Annotated[str, typer.Option("--source", "-s", help="Dossier Parquet")],
    batch_id: Annotated[str, typer.Option("--batch-id", help="Identifiant du batch")],
    db: Annotated[str | None, typer.Option("--db", help="Chemin SQLite")] = None,
) -> None:
    """Applique tous les modeles en Production a un batch de features."""
    from precite.predict.batch import predict_batch

    console.print(f"[cyan]Source:[/] {source}")
    console.print(f"[cyan]Batch:[/] {batch_id}")

    summary = predict_batch(source=source, batch_id=batch_id, db_path=db)

    table = Table(title="Prediction batch")
    table.add_column("Metrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    table.add_row("Papers", f"{summary['papers']:,}")
    table.add_row("Predictions ecrites", f"{summary['predictions']:,}")
    table.add_row("Modeles utilises", str(summary["models"]))
    console.print(table)


# ─── enrich ────────────────────────────────────────────────────────────────

enrich_app = typer.Typer(help="Enrichissement des features (citations, etc.).")
app.add_typer(enrich_app, name="enrich")


@enrich_app.command("build-cache")
def enrich_build_cache_cmd(
    icite_zip: Annotated[str, typer.Option("--icite")] = "data/icite/icite_metadata.zip",
    occ_zip: Annotated[str, typer.Option("--occ")] = "data/icite/open_citation_collection.zip",
    cache_path: Annotated[str, typer.Option("--cache")] = "data/icite/citations_3y_cache.pkl",
    year_map_cache: Annotated[str, typer.Option("--year-map-cache")] = "data/icite/year_map.pkl",
    horizon: Annotated[int, typer.Option("--horizon")] = 3,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
) -> None:
    """Scanne l'OCC une fois et met en cache les citations tronquees globales."""
    from precite.enrich.cache import build_global_citations_cache
    from precite.enrich.citations import load_year_map

    console.print("[cyan]Construction du cache OCC global[/]")
    console.print(f"  iCite     : {icite_zip}")
    console.print(f"  OCC       : {occ_zip}")
    console.print(f"  Cache     : {cache_path}")
    console.print(f"  Horizon   : {horizon} ans")
    console.print()

    year_map = load_year_map(icite_zip, cache_path=year_map_cache)

    path = build_global_citations_cache(
        occ_zip=occ_zip,
        year_map=year_map,
        horizon_years=horizon,
        cache_path=cache_path,
        force_rebuild=force,
    )

    size_mb = path.stat().st_size / 1024 / 1024
    console.print()
    console.print(f"[bold green]Cache pret :[/] {path} ({size_mb:.1f} MB)")


@enrich_app.command("attach")
def enrich_attach_cmd(
    input_parquet: Annotated[str, typer.Option("--input", "-i")],
    output_parquet: Annotated[str, typer.Option("--output", "-o")],
    cache_path: Annotated[str, typer.Option("--cache")] = "data/icite/citations_3y_cache.pkl",
    horizon: Annotated[int, typer.Option("--horizon")] = 3,
    snapshot_year: Annotated[int, typer.Option("--snapshot-year")] = 2026,
) -> None:
    """Joint le cache OCC a un Parquet de features (instantane)."""
    from precite.enrich.cache import attach_citations

    console.print(f"[cyan]Input :[/]   {input_parquet}")
    console.print(f"[cyan]Output :[/]  {output_parquet}")
    console.print(f"[cyan]Cache :[/]   {cache_path}")
    console.print()

    summary = attach_citations(
        input_parquet,
        output_parquet,
        cache_path=cache_path,
        horizon_years=horizon,
        snapshot_year=snapshot_year,
    )

    table = Table(title="Attache des citations")
    table.add_column("Metrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    table.add_row("Articles", f"{summary['rows']:,}")
    table.add_row("Avec historique", f"{summary['with_history']:,}")
    table.add_row("Sans historique", f"{summary['without_history']:,}")
    table.add_row("Avec citations", f"{summary['with_citations']:,}")
    table.add_row("Total citations", f"{summary['total_citations']:,}")
    table.add_row("Moyenne", f"{summary['mean']:.2f}")
    table.add_row("Mediane", f"{summary['median']:.0f}")
    table.add_row("Max", f"{summary['max']:,}")
    table.add_row("Cutoff annee", str(summary["cutoff_year"]))
    console.print(table)


@enrich_app.command("attach-batch")
def enrich_attach_batch_cmd(
    input_dir: Annotated[str, typer.Option("--input-dir", "-i")],
    output_dir: Annotated[str, typer.Option("--output-dir", "-o")],
    cache_path: Annotated[str, typer.Option("--cache")] = "data/icite/citations_3y_cache.pkl",
    horizon: Annotated[int, typer.Option("--horizon")] = 3,
    snapshot_year: Annotated[int, typer.Option("--snapshot-year")] = 2026,
) -> None:
    """Attache les citations a tous les Parquet d'un dossier (recursif)."""
    from precite.enrich.cache import attach_citations

    input_root = Path(input_dir)
    output_root = Path(output_dir)

    files = sorted(input_root.rglob("*.parquet"))
    if not files:
        console.print(f"[red]Aucun Parquet dans {input_root}[/]")
        raise typer.Exit(code=1)

    console.print(f"[cyan]{len(files)} fichiers a traiter[/]")
    console.print()

    total_rows = 0
    total_with_cit = 0
    total_with_hist = 0

    for i, f in enumerate(files, 1):
        rel = f.relative_to(input_root)
        out = output_root / rel

        try:
            summary = attach_citations(
                f,
                out,
                cache_path=cache_path,
                horizon_years=horizon,
                snapshot_year=snapshot_year,
            )
            total_rows += summary["rows"]
            total_with_cit += summary["with_citations"]
            total_with_hist += summary["with_history"]
            console.print(
                f"[{i}/{len(files)}] {rel} : {summary['rows']:,} lignes, "
                f"{summary['with_citations']:,} avec citations"
            )
        except Exception as exc:
            console.print(f"[red]ECHEC {rel} : {exc}[/]")

    console.print()
    console.print(f"[bold green]Termine[/] : {total_rows:,} articles traites")
    console.print(f"  Avec historique : {total_with_hist:,}")
    console.print(f"  Avec citations  : {total_with_cit:,}")


# ─── serve ─────────────────────────────────────────────────────────────────

serve_app = typer.Typer(help="Serveur d'inference FastAPI + app Streamlit.")
app.add_typer(serve_app, name="serve")


@serve_app.command("api")
def serve_api_cmd(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    reload: Annotated[bool, typer.Option("--reload/--no-reload")] = False,
    model_uri: Annotated[str | None, typer.Option("--model-uri")] = None,
) -> None:
    """Lance le serveur FastAPI."""
    import uvicorn

    if model_uri:
        import os

        os.environ["PRECITE_MODEL_URI"] = model_uri

    console.print(f"[bold green]Precite API[/] -> http://{host}:{port}")
    console.print(f"[cyan]Docs:[/] http://{host}:{port}/docs")
    uvicorn.run(
        "precite.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@serve_app.command("web")
def serve_web_cmd(
    port: Annotated[int, typer.Option("--port", "-p")] = 8501,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
) -> None:
    """Lance l'application Streamlit."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = os.environ.copy()
    if api_url:
        env["PRECITE_API_URL"] = api_url
    env.setdefault("PRECITE_API_URL", "http://127.0.0.1:8000")

    app_path = Path("apps/web/Home.py").resolve()
    if not app_path.exists():
        console.print(f"[red]Fichier introuvable : {app_path}[/]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Precite Web[/] -> http://127.0.0.1:{port}")
    console.print(f"[cyan]API cible:[/] {env['PRECITE_API_URL']}")

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    subprocess.run(cmd, env=env, check=False)


# ─── helpers ───────────────────────────────────────────────────────────────

def _list_recent_pubmed_files(base_url: str, n: int) -> list[str]:
    import httpx

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        r = client.get(base_url + "/")
        r.raise_for_status()
        names = [
            line.split('href="')[1].split('"')[0]
            for line in r.text.splitlines()
            if 'href="' in line
        ]
    gz = sorted(
        [x for x in names if x.endswith(".xml.gz") and "pubmed" in x.lower()],
        reverse=True,
    )
    return gz[:n]


def _print_ingest_summary(results: list[dict[str, int]]) -> None:
    table = Table(title="Resume de l'ingestion")
    table.add_column("Metrique", style="cyan")
    table.add_column("Valeur", justify="right", style="green")
    total_read = sum(r["articles_read"] for r in results)
    total_kept = sum(r["articles_kept"] for r in results)
    total_files = sum(r["parquet_files"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    table.add_row("Articles lus", f"{total_read:,}")
    table.add_row("Articles conserves", f"{total_kept:,}")
    table.add_row("Fichiers Parquet", str(total_files))
    table.add_row("Duree totale", f"{total_time} s")
    console.print(table)


if __name__ == "__main__":
    app()