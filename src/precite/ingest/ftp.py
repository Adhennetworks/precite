"""Telechargement PubMed en streaming avec reprise HTTP Range.

Zero disque. Memoire constante. Reprise automatique en cas de coupure reseau.
Rate limit global pour eviter le 503 de NCBI.
"""

from __future__ import annotations

import threading
import time as _time
from collections.abc import Iterator

import httpx

from precite.utils.logging import get_logger

log = get_logger(__name__)


MAX_RESUMES = 10
_rate_lock = threading.Lock()
_last_time = 0.0


def _rate_limit(min_interval: float = 0.25) -> None:
    """Limite globale : max ~4 nouvelles connexions/seconde.

    NCBI throttle a ~3 req/s par IP. Ce rate limiter evite le 503.
    """
    global _last_time
    with _rate_lock:
        now = _time.monotonic()
        wait = _last_time + min_interval - now
        if wait > 0:
            _time.sleep(wait)
        _last_time = _time.monotonic()


class DownloadError(RuntimeError):
    pass


def _stream_once(
    url: str,
    *,
    offset: int,
    chunk_size: int,
    client: httpx.Client,
) -> Iterator[bytes]:
    """Un seul passage de streaming a partir de `offset`."""
    _rate_limit(0.25)
    headers = {}
    if offset > 0:
        headers["Range"] = f"bytes={offset}-"

    with client.stream("GET", url, headers=headers) as r:
        if r.status_code == 503:
            log.warning("ftp.503", url=url, offset=offset)
            raise httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=r.request,
                response=r,
            )
        if r.status_code not in (200, 206):
            r.raise_for_status()

        if offset > 0 and r.status_code == 200:
            log.warning("ftp.range.unsupported", url=url, offset=offset)
            skipped = 0
            for chunk in r.iter_bytes(chunk_size=chunk_size):
                if skipped + len(chunk) <= offset:
                    skipped += len(chunk)
                    continue
                yield chunk[offset - skipped :]
                skipped = offset
                break

        for chunk in r.iter_bytes(chunk_size=chunk_size):
            if chunk:
                yield chunk


def stream_file(
    url: str,
    *,
    chunk_size: int = 1 << 20,
    timeout_read: float | None = None,
    max_resumes: int = MAX_RESUMES,
) -> Iterator[bytes]:
    """Yield des chunks bruts depuis une URL HTTP(S), avec reprise automatique."""
    log.info("ftp.stream.start", url=url)

    total = 0
    n_resumes = 0
    timeout = httpx.Timeout(connect=30.0, read=timeout_read, write=30.0, pool=30.0)

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        while True:
            try:
                for chunk in _stream_once(
                    url,
                    offset=total,
                    chunk_size=chunk_size,
                    client=client,
                ):
                    total += len(chunk)
                    yield chunk

                log.info("ftp.stream.done", url=url, bytes=total, resumes=n_resumes)
                return

            except (httpx.HTTPError, httpx.StreamError) as exc:
                n_resumes += 1
                if n_resumes > max_resumes:
                    log.error(
                        "ftp.stream.max_resumes",
                        url=url,
                        bytes=total,
                        resumes=n_resumes,
                    )
                    raise DownloadError(
                        f"trop de reprises ({n_resumes}) pour {url}, "
                        f"derniere erreur : {exc}"
                    ) from exc

                log.warning(
                    "ftp.stream.resume",
                    url=url,
                    bytes_consumed=total,
                    attempt=n_resumes,
                    error=str(exc)[:200],
                )

                err_lower = str(exc).lower()
                if "peer closed connection" in err_lower:
                    _time.sleep(0.5)
                elif "503" in err_lower or "service unavailable" in err_lower:
                    log.warning("ftp.503_backoff", url=url, wait=15)
                    _time.sleep(15)
                else:
                    _time.sleep(min(2 ** n_resumes, 30))