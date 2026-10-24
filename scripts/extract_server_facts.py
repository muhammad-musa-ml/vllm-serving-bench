"""Read an arm's startup log (results/server/<arm>/startup.log) into results/server/<arm>/facts.json.

    python scripts/extract_server_facts.py fp16 [--results results]

Only values printed by the server are recorded; a value the log does not contain is null.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATTERNS = {
    "model_loading_gib": (r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", 1, float),
    "model_loading_s": (r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", 2, float),
    "kv_cache_gib": (r"Available KV cache memory: ([0-9.]+) GiB", 1, float),
    "kv_cache_tokens": (r"GPU KV cache size: ([0-9,]+) tokens", 1, lambda s: int(s.replace(",", ""))),
    "max_concurrency_at_max_len": (r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", 1,
                                   float),
    "engine_init_s": (r"init engine \(profile, create kv cache, warmup model\) took ([0-9.]+) s", 1, float),
    "compilation_s": (r"init engine .* \(compilation: ([0-9.]+) s\)", 1, float),
    "free_memory_gib_at_startup": (r"Free memory on device \(([0-9.]+)/[0-9.]+ GiB\) on startup", 1, float),
    "total_memory_gib": (r"Free memory on device \([0-9.]+/([0-9.]+) GiB\) on startup", 1, float),
    "attention_backend": (r"Using (\w+) attention backend", 1, str),
    "vllm_version": (r"Initializing a V1 LLM engine \(v([0-9.]+)\)", 1, str),
}
INTERESTING = re.compile(r"quantiz|marlin|awq|gptq|kernel", re.I)
NOISE = re.compile(r"Capturing CUDA graphs|Fetching|it/s\]|Route:|GET /|POST /")


def parse(text):
    facts = {name: None for name in PATTERNS}
    for name, (pattern, group, cast) in PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            facts[name] = cast(m.group(group))
    lines = [ln.split("|", 1)[-1].strip() for ln in text.splitlines()]
    facts["quantization_lines"] = [ln for ln in lines
                                   if INTERESTING.search(ln) and not NOISE.search(ln)
                                   and "non-default args" not in ln and "Initializing a V1" not in ln][:12]
    warnings = [re.sub(r"^\(\w+ pid=\d+\) ", "", ln)
                for ln in lines if " WARNING " in ln or ln.startswith("Warning:")]
    cast = [w for w in warnings if "Attempted to load weight" in w]
    facts["weight_dtype_cast_warnings"] = len(cast)
    # one line per distinct message: timestamps dropped, the per-layer weight warnings counted above
    distinct = {re.sub(r"^WARNING \d\d-\d\d \d\d:\d\d:\d\d ", "WARNING ", w) for w in warnings
                if "Attempted to load weight" not in w}
    facts["warnings"] = sorted(distinct)
    return facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    arm_dir = ROOT / args.results / "server" / args.arm
    log = arm_dir / "startup.log"
    if not log.exists():
        sys.exit(f"{log} not found")
    facts = {"arm": args.arm, "source": f"{args.results}/server/{args.arm}/startup.log",
             **parse(log.read_text(encoding="utf-8", errors="replace"))}
    out = arm_dir / "facts.json"
    out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
