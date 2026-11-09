"""App factory FastAPI."""

from __future__ import annotations

import asyncio
import contextlib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from precite import __version__
from precite.api.routes import router
from precite.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


def _cors_origins() -> list[str]:
    raw = os.environ.get("PRECITE_CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Fallback dev : Streamlit local uniquement. En prod, definir PRECITE_CORS_ORIGINS.
    return ["http://localhost:8501", "http://127.0.0.1:8501"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(
        level=os.environ.get("PRECITE_LOG_LEVEL", "INFO"),
        json_output=os.environ.get("PRECITE_LOG_JSON", "true").lower() != "false",
    )
    log.info("api.startup", version=__version__)

    app.state.warmup_done = asyncio.Event()

    async def warmup() -> None:
        try:
            from precite.api.deps import get_store
            store = await asyncio.to_thread(get_store)
            for pmid in ("19585584", "38509752"):
                await asyncio.to_thread(store.get_paper, pmid)
            log.info("api.warmup.done")
        except Exception as exc:
            log.warning("api.warmup.failed", error=str(exc))
        finally:
            app.state.warmup_done.set()

    task = asyncio.create_task(warmup(), name="api-warmup")
    app.state.warmup_task = task
    try:
        yield
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def create_app() -> FastAPI:
    app = FastAPI(
        title="Precite API",
        description="Consultation des predictions de citations Precite.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()