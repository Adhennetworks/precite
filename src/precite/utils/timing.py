"""Chronometrage simple reutilisable."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Timer:
    marks: dict[str, float] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)

    def mark(self, name: str) -> None:
        self.marks[name] = time.perf_counter() - self._start

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def summary(self) -> dict[str, float]:
        out = dict(self.marks)
        out["total"] = self.elapsed()
        return out


@contextmanager
def timed(label: str, logger=None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if logger:
        logger.info("timing", label=label, seconds=round(dt, 3))
