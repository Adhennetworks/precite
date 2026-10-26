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


async def _warmup_task(app: FastAPI):
    """Warmup non bloquant : charge le store en arriere-plan."""
    try:
        from precite.api.deps import get_store
        store = await asyncio.to_thread(get_store)
        # 3 PMIDs : 1 annuel + 2 production
        for pmid in ("19585584", "38509752", "41599690"):
            await asyncio.to_thread(store.get_paper, pmid)
        log.info("api.warmup.done")
    except Exception as exc:
        log.warning("api.warmup.failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup", version=__version__)
    # Lancer le warmup en arriere-plan SANS bloquer le demarrage
    task = asyncio.create_task(_warmup_task(app))
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