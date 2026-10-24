"""Streaming chat-completion client that timestamps every chunk.

Times come from time.perf_counter(). `t_sched` is when the request was supposed to start. In a
closed loop that is the moment the worker sends it; in an open loop it is the scheduled arrival
time, so time spent waiting for the client itself to get going is charged to the latency.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import aiohttp

DONE = "[DONE]"


@dataclass
class RequestRecord:
    idx: int
    prompt_id: int
    ok: bool
    error: str | None
    status: int | None
    t_sched: float
    t_send: float
    t_end: float
    ttft_s: float | None
    e2e_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    n_chunks: int
    tpot_s: float | None
    itl_max_s: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_sse_line(line: bytes | str):
    """Parse one line of a server-sent-event stream.

    Returns the decoded JSON object for a `data:` line, DONE for the end marker, and None for
    blank lines, comments and anything else.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == DONE:
        return DONE
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def chunk_text(event: dict) -> str:
    """The text delta carried by a chat-completion chunk ('' when it has none)."""
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


async def stream_request(session: aiohttp.ClientSession, url: str, body: dict, *, idx: int,
                         prompt_id: int, t0: float, t_sched: float | None = None,
                         timeout_s: float = 300.0) -> RequestRecord:
    """Send one streamed request and return its timing record. Never raises on request errors."""
    t_send_abs = time.perf_counter()
    t_send = t_send_abs - t0
    sched_abs = t0 + t_sched if t_sched is not None else t_send_abs
    sched_rel = sched_abs - t0
    status = None
    error = None
    first_abs = None
    last_abs = None
    prev_abs = None
    itl_max = 0.0
    n_chunks = 0
    usage = None
    saw_done = False
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=body, timeout=timeout) as resp:
            status = resp.status
            if status != 200:
                text = (await resp.text())[:200]
                error = f"HTTP {status}: {text}"
            else:
                async for raw in resp.content:
                    event = parse_sse_line(raw)
                    if event is None:
                        continue
                    if event == DONE:
                        saw_done = True
                        break
                    if event.get("usage"):
                        usage = event["usage"]
                    if chunk_text(event):
                        now = time.perf_counter()
                        n_chunks += 1
                        if first_abs is None:
                            first_abs = now
                        else:
                            itl_max = max(itl_max, now - prev_abs)
                        prev_abs = now
                        last_abs = now
    except (TimeoutError, aiohttp.ClientError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    end_abs = time.perf_counter()
    ok = error is None and saw_done and usage is not None and first_abs is not None
    if error is None and not ok:
        error = "stream ended without a [DONE] marker, usage chunk or any content"
    completion = usage.get("completion_tokens") if usage else None
    prompt = usage.get("prompt_tokens") if usage else None
    tpot = None
    if ok and completion and completion > 1 and last_abs is not None and first_abs is not None:
        tpot = (last_abs - first_abs) / (completion - 1)
    return RequestRecord(
        idx=idx, prompt_id=prompt_id, ok=ok, error=error, status=status,
        t_sched=sched_rel, t_send=t_send, t_end=end_abs - t0,
        ttft_s=(first_abs - sched_abs) if first_abs is not None else None,
        e2e_s=(last_abs - sched_abs) if (ok and last_abs is not None) else None,
        prompt_tokens=prompt, completion_tokens=completion, n_chunks=n_chunks,
        tpot_s=tpot, itl_max_s=itl_max if n_chunks > 1 else None,
    )
