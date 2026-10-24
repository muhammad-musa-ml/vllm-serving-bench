"""GPU readings from nvidia-smi, sampled while a load level runs.

One nvidia-smi process runs in loop mode (`-lms`) for the whole pass; each level takes the samples
whose receive time falls inside its window. The memory figure is total memory in use on the card,
so it includes whatever the desktop is using; the idle reading before the server starts is recorded
separately by the caller.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

QUERY = "memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw"
FIELDS = ("mem_mib", "util_pct", "temp_c", "sm_mhz", "power_w")


def parse_gpu_line(line: str) -> dict | None:
    """Parse one `--format=csv,noheader,nounits` line; None if it is not a full reading."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) != len(FIELDS):
        return None
    out = {}
    for name, part in zip(FIELDS, parts, strict=True):
        try:
            out[name] = float(part)
        except ValueError:
            out[name] = None
    if out["mem_mib"] is None:
        return None
    return out


def summarize_samples(samples: list[dict]) -> dict:
    """Peak and mean memory, and mean/max of the other readings, over a list of samples."""
    if not samples:
        return {"n": 0}

    def col(name):
        return [s[name] for s in samples if s.get(name) is not None]

    def stats(name):
        v = col(name)
        if not v:
            return None
        return {"mean": sum(v) / len(v), "min": min(v), "max": max(v)}

    return {
        "n": len(samples),
        "mem_mib": stats("mem_mib"),
        "util_pct": stats("util_pct"),
        "temp_c": stats("temp_c"),
        "sm_mhz": stats("sm_mhz"),
        "power_w": stats("power_w"),
    }


def read_once(smi: str = "nvidia-smi") -> dict | None:
    """One reading, for idle baselines. None if nvidia-smi is missing or fails."""
    exe = shutil.which(smi)
    if not exe:
        return None
    try:
        result = subprocess.run([exe, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    lines = result.stdout.strip().splitlines()
    return parse_gpu_line(lines[0]) if lines else None


class GpuSampler:
    """Background sampler; `window(a, b)` gives the samples received in a perf_counter interval."""

    def __init__(self, interval_ms: int = 250, smi: str = "nvidia-smi"):
        self.interval_ms = interval_ms
        self._smi = shutil.which(smi)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, dict]] = []
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._smi is not None

    def start(self) -> None:
        if not self._smi:
            return
        self._proc = subprocess.Popen(
            [self._smi, f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
             "-lms", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            sample = parse_gpu_line(line)
            if sample is not None:
                with self._lock:
                    self._samples.append((time.perf_counter(), sample))

    def window(self, t_a: float, t_b: float) -> list[dict]:
        with self._lock:
            return [s for t, s in self._samples if t_a <= t <= t_b]

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
