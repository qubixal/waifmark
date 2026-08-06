"""Shared control primitives for cancellable benchmark execution."""

from __future__ import annotations

import threading


class BenchmarkCancelledError(RuntimeError):
    """Raised when a benchmark run is cancelled by the user."""


class RunControl:
    """Pause/resume/cancel control shared between UI and worker threads."""

    def __init__(self) -> None:
        self._paused = False
        self._cancelled = False
        self._condition = threading.Condition()

    def pause(self) -> None:
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._paused = False
            self._condition.notify_all()

    def wait_if_paused(self) -> None:
        with self._condition:
            while self._paused and not self._cancelled:
                self._condition.wait(timeout=0.5)
        self.check_cancelled()

    def check_cancelled(self) -> None:
        with self._condition:
            if self._cancelled:
                raise BenchmarkCancelledError("Benchmark run cancelled.")

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled
