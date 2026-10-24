"""The fixed quality set: prompts, deterministic scorers and a runner.

Two sets, both scored by code with no model in the loop:

* `mmlu-subset`: four-choice questions drawn from MMLU (see `scripts/make_mmlu_subset.py`). A reply
  is right if the letter extracted from it is the answer.
* `synthetic-tasks`: generated items (see `scripts/make_synthetic_tasks.py`): integer arithmetic,
  extraction of one fact from a made-up passage, and format-following checked by a script.

Replies come from greedy decoding (temperature 0) and stop at the end of the answer, unlike the
load runs, which force a fixed length.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path

import aiohttp

from .stats import wilson_interval

MAX_TOKENS = {"mcq": 8, "int": 24, "text": 24, "exact": 24, "json": 64, "lines": 96}


def load_tasks(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mcq_prompt(task: dict) -> str:
    subject = task["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}. "
             "Reply with only the letter (A, B, C or D) of the correct answer.", "",
             task["question"]]
    lines += [f"{letter}. {choice}" for letter, choice in zip("ABCD", task["choices"], strict=True)]
    lines.append("Answer:")
    return "\n".join(lines)


def prompt_for(task: dict) -> str:
    return mcq_prompt(task) if task["scorer"] == "mcq" else task["prompt"]


_MCQ_START = re.compile(r"^\s*\(?([ABCD])\b")
_MCQ_PHRASE = re.compile(r"[Aa]nswer\s*(?:is)?\s*[:\-]?\s*\(?([ABCD])\b")
_MCQ_ANY = re.compile(r"\b([ABCD])\b")


def extract_choice(reply: str) -> str | None:
    """The answer letter in a reply: a leading one, else 'answer is X', else the first lone A-D."""
    for pattern in (_MCQ_START, _MCQ_PHRASE, _MCQ_ANY):
        m = pattern.search(reply)
        if m:
            return m.group(1)
    return None


_INT = re.compile(r"-?\d[\d,]*")


def extract_int(reply: str) -> int | None:
    """The last integer in a reply (commas allowed): 'a + b = c' and 'The answer is c' give c."""
    found = _INT.findall(reply)
    if not found:
        return None
    try:
        return int(found[-1].replace(",", ""))
    except ValueError:
        return None


def normalize_text(s: str) -> str:
    return s.strip().strip(".\"'`").strip().lstrip("#").strip().lower()


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


def score_task(task: dict, reply: str) -> bool:
    scorer = task["scorer"]
    expected = task["answer"] if scorer == "mcq" else task["expected"]
    if scorer == "mcq":
        return extract_choice(reply) == expected
    if scorer == "int":
        return extract_int(reply) == int(expected)
    if scorer == "text":
        return normalize_text(reply) == normalize_text(str(expected))
    if scorer == "exact":
        return reply.strip() == expected
    if scorer == "json":
        try:
            return json.loads(_strip_fences(reply)) == expected
        except (json.JSONDecodeError, ValueError):
            return False
    if scorer == "lines":
        lines = [ln for ln in reply.strip().splitlines() if ln.strip()]
        prefix = expected["prefix"]
        return len(lines) == expected["n"] and all(ln.startswith(prefix) for ln in lines)
    raise ValueError(f"unknown scorer {scorer!r}")


async def run_quality(base_url: str, tasks: list[dict], *, model: str, concurrency: int = 8,
                      timeout_s: float = 120.0) -> list[dict]:
    """Ask every task once at temperature 0 and score the replies. Order follows `tasks`."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(tasks)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(i: int, task: dict):
            body = {"model": model, "temperature": 0.0, "stream": False,
                    "max_tokens": MAX_TOKENS[task["scorer"]],
                    "messages": [{"role": "user", "content": prompt_for(task)}]}
            async with sem:
                reply, error = "", None
                try:
                    async with session.post(url, json=body,
                                            timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                        if resp.status != 200:
                            error = f"HTTP {resp.status}"
                        else:
                            data = await resp.json()
                            reply = data["choices"][0]["message"]["content"] or ""
                except (TimeoutError, aiohttp.ClientError, KeyError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            results[i] = {
                "id": task["id"], "set": task["set"], "kind": task.get("kind", task["scorer"]),
                "ok": error is None, "error": error, "reply": reply,
                "correct": bool(error is None and score_task(task, reply)),
            }

        await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    return [r for r in results if r is not None]


def summarize_quality(results: list[dict]) -> dict:
    """Accuracy per set and per kind, with 95% Wilson intervals. Failed requests count as wrong."""
    def block(rows):
        n = len(rows)
        k = sum(1 for r in rows if r["correct"])
        lo, hi = wilson_interval(k, n)
        return {"n": n, "correct": k, "accuracy": (k / n) if n else None,
                "ci95": [lo, hi], "request_errors": sum(1 for r in rows if not r["ok"])}

    by_set: dict[str, list[dict]] = defaultdict(list)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_set[r["set"]].append(r)
        by_kind[f"{r['set']}/{r['kind']}"].append(r)
    return {"all": block(results),
            "by_set": {k: block(v) for k, v in sorted(by_set.items())},
            "by_kind": {k: block(v) for k, v in sorted(by_kind.items())}}
