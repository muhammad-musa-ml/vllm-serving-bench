"""Command line: `servebench bench`, `servebench quality`, `servebench summarize`."""
from __future__ import annotations

import argparse
import asyncio
import datetime
import gzip
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import aiohttp

from . import __version__
from .gpu import GpuSampler, read_once
from .lock import MeasurementLock
from .quality import load_tasks, run_quality, summarize_quality
from .runner import run_pass
from .workload import DEFAULT_COUNT, DEFAULT_SEED, build_prompts

DEFAULT_LEVELS = "1,2,4,8,16"
DEFAULT_RATES = "2,4,6,8"
REPO = Path(__file__).resolve().parents[2]


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_list(text: str, cast) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")


def write_jsonl_gz(path: Path, rows) -> None:
    """Gzip with a zero timestamp, so identical content gives identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              timeout=15)
    try:
        head = run("rev-parse", "HEAD")
        dirty = run("status", "--porcelain", "--", "src", "scripts", "arms.json", "quality")
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    if head.returncode != 0:
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


def capture_docker_ps(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("docker")
    if not exe:
        path.write_text("docker not found on PATH\n", encoding="utf-8", newline="\n")
        return
    try:
        out = subprocess.run([exe, "ps"], capture_output=True, text=True, timeout=30)
        text = out.stdout if out.returncode == 0 else f"docker ps failed: {out.stderr}"
    except (OSError, subprocess.SubprocessError) as exc:
        text = f"docker ps failed: {exc}"
    path.write_text(f"# captured {utc_now()}\n{text}", encoding="utf-8", newline="\n")


async def server_info(url: str) -> dict:
    info: dict = {}
    async with aiohttp.ClientSession() as s:
        for key, path in (("version", "/version"), ("models", "/v1/models")):
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with s.get(url.rstrip("/") + path, timeout=timeout) as r:
                    info[key] = await r.json()
            except (aiohttp.ClientError, TimeoutError, ValueError):
                info[key] = None
    return info


def with_lock(args):
    if not args.lock_file:
        return None
    lock = MeasurementLock(args.lock_file, args.lock_owner, poll_s=args.lock_poll)
    print(f"waiting for the measurement lock {Path(args.lock_file).name} ...", flush=True)
    lock.acquire()
    print(f"lock taken after {lock.waited_s:.1f} s", flush=True)
    return lock


def cmd_bench(args) -> int:
    out = Path(args.out)
    levels = parse_list(args.levels, int)
    rates = parse_list(args.open_rates, float)
    prompts = build_prompts(DEFAULT_COUNT, DEFAULT_SEED)
    name = "warmup" if args.warmup else f"run-{args.run:02d}"
    if args.warmup:
        levels, rates = [4], []
    lock = with_lock(args)
    try:
        started = utc_now()
        capture_docker_ps(out / f"{name}.docker-ps.txt")
        idle = read_once()
        info = asyncio.run(server_info(args.url))
        warmup_pass = None
        if args.warmup_first and not args.warmup:
            # a short pass inside the same lock hold, kept in the file but never counted
            warmup_pass, _ = asyncio.run(run_pass(
                args.url, prompts, model=args.model, levels=[4], open_rates=[],
                max_tokens=args.max_tokens, open_duration_s=args.open_duration,
                drain_s=args.drain, seed=args.seed, sampler=None, cooldown_s=args.cooldown,
                n_override=24, log=lambda *_: None))
        with GpuSampler(interval_ms=args.gpu_interval_ms) as sampler:
            summaries, raw = asyncio.run(run_pass(
                args.url, prompts, model=args.model, levels=levels, open_rates=rates,
                max_tokens=args.max_tokens, open_duration_s=args.open_duration,
                drain_s=args.drain, seed=args.seed, sampler=sampler,
                max_in_flight=args.max_in_flight, cooldown_s=args.cooldown,
                n_override=24 if args.warmup else None))
        finished = utc_now()
    finally:
        if lock:
            lock.release()
    result = {
        "schema": 1,
        "arm": args.arm,
        "name": name,
        "started_utc": started,
        "finished_utc": finished,
        "harness": {"version": __version__, **git_state()},
        "settings": {
            "levels": levels, "open_rates": rates, "open_duration_s": args.open_duration,
            "drain_s": args.drain, "max_tokens": args.max_tokens, "fixed_length": True,
            "seed": args.seed,
            "prompt_set": {"count": len(prompts), "seed": DEFAULT_SEED,
                           "mean_words": sum(p.n_words for p in prompts) / len(prompts)},
            "model": args.model, "gpu_interval_ms": args.gpu_interval_ms,
        },
        "server": info,
        "gpu_before_pass": idle,
        "lock": ({"owner": args.lock_owner, "waited_s": lock.waited_s} if lock else None),
        "machine": {"python": sys.version.split()[0], "platform": platform.platform(),
                    "aiohttp": aiohttp.__version__},
        "warmup_pass": warmup_pass,
        "levels": summaries,
    }
    write_json(out / f"{name}.json", result)
    rows = []
    for meta, recs in raw:
        for r in recs:
            rows.append({**meta, **r.to_dict()})
    write_jsonl_gz(out / f"{name}.requests.jsonl.gz", rows)
    print(f"wrote {out / (name + '.json')}", flush=True)
    return 0


def cmd_quality(args) -> int:
    out = Path(args.out)
    tasks = []
    sets = parse_list(args.sets, str)
    if "mmlu" in sets:
        tasks += load_tasks(REPO / "quality" / "mmlu_subset.jsonl")
    if "synthetic" in sets:
        tasks += load_tasks(REPO / "quality" / "synthetic_tasks.jsonl")
    lock = with_lock(args)
    try:
        started = utc_now()
        results = asyncio.run(run_quality(args.url, tasks, model=args.model,
                                          concurrency=args.concurrency))
        finished = utc_now()
    finally:
        if lock:
            lock.release()
    name = f"run-{args.run:02d}"
    summary = {
        "schema": 1, "arm": args.arm, "name": name, "started_utc": started,
        "finished_utc": finished, "harness": {"version": __version__, **git_state()},
        "settings": {"sets": sets, "concurrency": args.concurrency, "temperature": 0.0,
                     "n_tasks": len(tasks), "model": args.model},
        "summary": summarize_quality(results),
    }
    write_json(out / f"{name}.json", summary)
    write_jsonl_gz(out / f"{name}.items.jsonl.gz", results)
    acc = summary["summary"]["all"]
    print(f"quality {args.arm} run {args.run}: {acc['correct']}/{acc['n']} "
          f"({acc['accuracy']:.3f}), {acc['request_errors']} request errors", flush=True)
    return 0


def cmd_summarize(args) -> int:
    from .report import summarize_results
    summarize_results(Path(args.results), plots=not args.no_plots)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="servebench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--url", default="http://127.0.0.1:18480")
        sp.add_argument("--model", default="bench", help="served model name")
        sp.add_argument("--arm", required=True)
        sp.add_argument("--run", type=int, default=1)
        sp.add_argument("--out", required=True)
        sp.add_argument("--lock-file", default=None)
        sp.add_argument("--lock-owner", default="servebench")
        sp.add_argument("--lock-poll", type=float, default=2.0,
                        help="seconds between attempts to take the lock file")

    b = sub.add_parser("bench", help="one load pass: every closed-loop level, then open-loop rates")
    common(b)
    b.add_argument("--levels", default=DEFAULT_LEVELS)
    b.add_argument("--open-rates", default=DEFAULT_RATES)
    b.add_argument("--open-duration", type=float, default=40.0)
    b.add_argument("--drain", type=float, default=60.0)
    b.add_argument("--max-tokens", type=int, default=128)
    b.add_argument("--max-in-flight", type=int, default=512)
    b.add_argument("--cooldown", type=float, default=3.0)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--gpu-interval-ms", type=int, default=250)
    b.add_argument("--warmup", action="store_true", help="short pass, written as warmup.json")
    b.add_argument("--warmup-first", action="store_true",
                   help="run a short uncounted pass inside the same lock hold, before the levels")
    b.set_defaults(func=cmd_bench)

    q = sub.add_parser("quality", help="answer the fixed question sets at temperature 0")
    common(q)
    q.add_argument("--sets", default="mmlu,synthetic")
    q.add_argument("--concurrency", type=int, default=8)
    q.set_defaults(func=cmd_quality)

    s = sub.add_parser("summarize", help="build summary.json, summary.md, runs.csv and plots")
    s.add_argument("--results", default="results")
    s.add_argument("--no-plots", action="store_true")
    s.set_defaults(func=cmd_summarize)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
