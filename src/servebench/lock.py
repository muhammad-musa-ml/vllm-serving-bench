"""Exclusive lock file for runs that share a machine.

`acquire()` creates the file with mode "x", so two processes cannot both succeed, and polls until
the file is free. The file holds the owner's name and the UTC time it was taken. `release()` only
deletes a lock this owner took.
"""
from __future__ import annotations

import datetime
import os
import time
from pathlib import Path


class LockTimeout(Exception):
    pass


class MeasurementLock:
    def __init__(self, path: str | os.PathLike, owner: str, *, poll_s: float = 2.0,
                 timeout_s: float | None = None, sleep=time.sleep, clock=time.monotonic):
        self.path = Path(path)
        self.owner = owner
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self._sleep = sleep
        self._clock = clock
        self.waited_s = 0.0
        self.held = False

    def _stamp(self) -> str:
        now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return f"{self.owner} {now}\n"

    def acquire(self) -> None:
        start = self._clock()
        while True:
            try:
                with open(self.path, "x", encoding="utf-8") as f:
                    f.write(self._stamp())
                self.held = True
                self.waited_s = self._clock() - start
                return
            except FileExistsError:
                if self.timeout_s is not None and self._clock() - start >= self.timeout_s:
                    raise LockTimeout(f"{self.path} still held after {self.timeout_s} s") from None
                self._sleep(self.poll_s)

    def holder(self) -> str | None:
        try:
            return self.path.read_text(encoding="utf-8").split()[0]
        except (OSError, IndexError):
            return None

    def release(self) -> None:
        if not self.held:
            return
        if self.holder() == self.owner:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
