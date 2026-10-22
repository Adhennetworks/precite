"""App factory FastAPI."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from precite import __version__
from precite.api.routes import router
from precite.utils.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup", version=__version__)

    async def warmup():
        try:
            from precite.api.deps import get_store
            store = await asyncio.to_thread(get_store)
            # Warmup multi-annees : charge le cache HTTP de DuckDB
            # pour les fichiers annuels ET production
            warmup_pmids = [
                "19585584",   # 2017 -> annuel
                "38509752",   # 2024 -> production
                "41599690",   # 2026 -> production
            ]
            for pmid in warmup_pmids:
                await asyncio.to_thread(store.get_paper, pmid)
            log.info("api.warmup.done", n_warmed=len(warmup_pmids))
        except Exception as exc:
            log.warning("api.warmup.failed", error=str(exc))

    # Garde une reference forte pour eviter le GC
    task = asyncio.create_task(warmup())
    app.state.warmup_task = task

    yield

    if not task.done():
        task.cancel()
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Precite API",
        description="Consultation des predictions de citations Precite.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()