"""Closed-loop and open-loop load generation, and the per-level summaries.

Closed loop: `concurrency` workers, each sending its next request when the last one finishes.
Throughput is completed work divided by the time from the first send to the last completion.

Open loop: requests arrive as a Poisson process at a fixed rate whether or not the server keeps
up. Latency runs from the scheduled arrival, so a client that falls behind does not make the server
look faster than it is. After the last arrival the queue gets a bounded time to drain; whatever is
still running then is cancelled and counted as unfinished. Throughput uses the closed loop's basis
(completed work over first arrival to last completion), which reads a little low under light load
because the last requests finish after the window.
"""
from __future__ import annotations

import asyncio
import itertools
import random
import re
import time

import aiohttp

from .client import RequestRecord, stream_request
from .gpu import GpuSampler, summarize_samples
from .stats import summarize
from .workload import Prompt, chat_body

CHAT_PATH = "/v1/chat/completions"
IDLE_METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def requests_for_level(concurrency: int) -> int:
    """Requests sent at a closed-loop level: enough for several waves, never fewer than 32."""
    return max(32, 16 * concurrency)


def poisson_arrivals(rate: float, duration_s: float, seed: int) -> list[float]:
    """Arrival offsets in [0, duration_s) of a Poisson process; one seed gives one list."""
    if rate <= 0 or duration_s <= 0:
        raise ValueError("rate and duration must be positive")
    rng = random.Random(seed)
    t = 0.0
    out: list[float] = []
    while True:
        t += rng.expovariate(rate)
        if t >= duration_s:
            return out
        out.append(t)


def _clean(values):
    return [v for v in values if v is not None]


def summarize_records(records: list[RequestRecord], *, max_tokens: int) -> dict:
    ok = [r for r in records if r.ok]
    failed = [r for r in records if not r.ok]
    errors: dict[str, int] = {}
    for r in failed:
        key = (r.error or "unknown")[:120]
        errors[key] = errors.get(key, 0) + 1
    completion = _clean(r.completion_tokens for r in ok)
    prompt = _clean(r.prompt_tokens for r in ok)
    return {
        "n_requests": len(records),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "errors": errors,
        "ttft_s": summarize(_clean(r.ttft_s for r in ok)),
        "e2e_s": summarize(_clean(r.e2e_s for r in ok)),
        "tpot_s": summarize(_clean(r.tpot_s for r in ok)),
        "itl_max_s": summarize(_clean(r.itl_max_s for r in ok)),
        "prompt_tokens_mean": (sum(prompt) / len(prompt)) if prompt else None,
        "completion_tokens_mean": (sum(completion) / len(completion)) if completion else None,
        "short_outputs": sum(1 for c in completion if c != max_tokens),
    }


def _gpu_summary(sampler: GpuSampler | None, t_a: float, t_b: float) -> dict | None:
    if sampler is None or not sampler.available:
        return None
    return summarize_samples(sampler.window(t_a, t_b))


async def measure_closed(base_url: str, prompts: list[Prompt], *, concurrency: int,
                         n_requests: int, max_tokens: int, model: str,
                         sampler: GpuSampler | None = None,
                         timeout_s: float = 300.0) -> tuple[dict, list[RequestRecord]]:
    url = base_url.rstrip("/") + CHAT_PATH
    counter = itertools.count()
    records: list[RequestRecord] = []
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def worker():
            while True:
                i = next(counter)
                if i >= n_requests:
                    return
                p = prompts[i % len(prompts)]
                body = chat_body(p.text, model=model, max_tokens=max_tokens, fixed_length=True)
                records.append(await stream_request(session, url, body, idx=i, prompt_id=p.id,
                                                    t0=t0, timeout_s=timeout_s))

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    t1 = time.perf_counter()
    records.sort(key=lambda r: r.idx)
    ok = [r for r in records if r.ok]
    wall = (max(r.t_end for r in records) - min(r.t_send for r in records)) if records else 0.0
    tokens = sum(r.completion_tokens or 0 for r in ok)
    level = {
        "mode": "closed",
        "concurrency": concurrency,
        "wall_s": wall,
        "req_s": (len(ok) / wall) if wall > 0 else None,
        "out_tok_s": (tokens / wall) if wall > 0 else None,
        **summarize_records(records, max_tokens=max_tokens),
        "gpu": _gpu_summary(sampler, t0, t1),
    }
    return level, records


async def measure_open(base_url: str, prompts: list[Prompt], *, rate: float, duration_s: float,
                       seed: int, max_tokens: int, model: str, max_in_flight: int = 512,
                       drain_s: float = 60.0, sampler: GpuSampler | None = None,
                       timeout_s: float = 300.0) -> tuple[dict, list[RequestRecord]]:
    url = base_url.rstrip("/") + CHAT_PATH
    arrivals = poisson_arrivals(rate, duration_s, seed)
    records: list[RequestRecord] = []
    tasks: set[asyncio.Task] = set()
    dropped = 0
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def fire(i: int, at: float):
            p = prompts[i % len(prompts)]
            body = chat_body(p.text, model=model, max_tokens=max_tokens, fixed_length=True)
            rec = await stream_request(session, url, body, idx=i, prompt_id=p.id, t0=t0,
                                       t_sched=at, timeout_s=timeout_s)
            records.append(rec)

        for i, at in enumerate(arrivals):
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if len(tasks) >= max_in_flight:
                dropped += 1
                continue
            task = asyncio.create_task(fire(i, at))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        # let the window close, then give the queue a bounded time to drain
        remaining = t0 + duration_s - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        pending = set(tasks)
        if pending:
            _, still = await asyncio.wait(pending, timeout=drain_s)
            for task in still:
                task.cancel()
            if still:
                await asyncio.gather(*still, return_exceptions=True)
        unfinished = len(pending) - len([t for t in pending if t.done() and not t.cancelled()])
    t1 = time.perf_counter()
    records.sort(key=lambda r: r.idx)
    ok = [r for r in records if r.ok]
    realized = len(arrivals) / duration_s
    # throughput on the closed loop's basis: completed work over first arrival to last completion
    span = (max(r.t_end for r in ok) - min(r.t_sched for r in ok)) if ok else 0.0
    tokens = sum(r.completion_tokens or 0 for r in ok)
    level = {
        "mode": "open",
        "offered_rate": rate,
        "realized_rate": realized,
        "duration_s": duration_s,
        "n_arrivals": len(arrivals),
        "n_dropped": dropped,
        "n_unfinished": unfinished,
        "span_s": span,
        "req_s_overall": (len(ok) / span) if span > 0 else None,
        "out_tok_s_overall": (tokens / span) if span > 0 else None,
        "completion_ratio": (len(ok) / len(arrivals)) if arrivals else None,
        "send_lag_s": summarize(_clean(r.t_send - r.t_sched for r in records)),
        **summarize_records(records, max_tokens=max_tokens),
        "gpu": _gpu_summary(sampler, t0, t1),
    }
    return level, records


def parse_gauge(metrics_text: str, name: str) -> float | None:
    """Sum of all samples of a Prometheus gauge in exposition text (None if absent)."""
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", re.M)
    values = [float(m.group(1)) for m in pattern.finditer(metrics_text)]
    return sum(values) if values else None


async def wait_idle(base_url: str, *, timeout_s: float = 60.0, poll_s: float = 0.5) -> bool:
    """Wait until the server reports no running or waiting requests. True if it got there."""
    deadline = time.monotonic() + timeout_s
    url = base_url.rstrip("/") + "/metrics"
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    text = await resp.text()
            except (TimeoutError, aiohttp.ClientError, OSError):
                await asyncio.sleep(poll_s)
                continue
            vals = [parse_gauge(text, n) for n in IDLE_METRICS]
            if all(v is not None for v in vals) and sum(vals) == 0:
                return True
            await asyncio.sleep(poll_s)
    return False


async def run_pass(base_url: str, prompts: list[Prompt], *, model: str, levels: list[int],
                   open_rates: list[float], max_tokens: int, open_duration_s: float,
                   drain_s: float, seed: int, sampler: GpuSampler | None,
                   max_in_flight: int = 512, cooldown_s: float = 3.0, n_override: int | None = None,
                   log=print) -> tuple[list[dict], list[tuple[dict, list[RequestRecord]]]]:
    """One full pass: every closed-loop level, then every open-loop rate.

    `n_override` replaces the per-level request count (used for the short warm-up pass).
    """
    summaries: list[dict] = []
    raw: list[tuple[dict, list[RequestRecord]]] = []
    for c in levels:
        n = n_override if n_override is not None else requests_for_level(c)
        level, recs = await measure_closed(base_url, prompts, concurrency=c, n_requests=n,
                                           max_tokens=max_tokens, model=model, sampler=sampler)
        level["idle_before_next"] = await wait_idle(base_url)
        summaries.append(level)
        raw.append(({"mode": "closed", "concurrency": c}, recs))
        log(f"closed c={c}: {(level['out_tok_s'] or 0):.1f} tok/s, p50 e2e "
            f"{(level['e2e_s']['p50'] or float('nan')):.2f} s, {level['n_failed']} failed")
        await asyncio.sleep(cooldown_s)
    for rate in open_rates:
        level, recs = await measure_open(
            base_url, prompts, rate=rate, duration_s=open_duration_s,
            seed=seed + int(rate * 1000), max_tokens=max_tokens, model=model,
            max_in_flight=max_in_flight, drain_s=drain_s, sampler=sampler)
        level["idle_before_next"] = await wait_idle(base_url)
        summaries.append(level)
        raw.append(({"mode": "open", "offered_rate": rate}, recs))
        log(f"open rate={rate}: {(level['req_s_overall'] or 0):.2f} req/s completed against "
            f"{level['realized_rate']:.2f} offered, {level['n_unfinished']} unfinished")
        await asyncio.sleep(cooldown_s)
    return summaries, raw
