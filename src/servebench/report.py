"""Turn the run files in `results/` into summary.json, summary.md, runs.csv and plots.

Layout read:  results/runs/<arm>/run-NN.json (+ run-NN.requests.jsonl.gz), results/quality/<arm>/
run-NN.json (+ run-NN.items.jsonl.gz). `warmup.json` files are ignored. Every figure is a median
with the range over the runs of one configuration; pooled percentiles come from all requests of
all runs of that configuration.
"""
from __future__ import annotations

import csv
import datetime
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

from .stats import fmt_range, mcnemar_exact, median_range, summarize

RUN_FILE = re.compile(r"^run-(\d{2})\.json$")
ARM_ORDER = ["fp16", "awq", "gptq"]
KNEE_GAIN = 1.10          # closed loop: saturated when doubling the workers gains under 10%
OPEN_TTFT_S = 1.0         # open loop: saturated when the median time to first token passes 1 s


def dig(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


CLOSED_METRICS = {
    "out_tok_s": lambda lv: lv.get("out_tok_s"),
    "req_s": lambda lv: lv.get("req_s"),
    "ttft_p50_s": lambda lv: dig(lv, "ttft_s", "p50"),
    "ttft_p95_s": lambda lv: dig(lv, "ttft_s", "p95"),
    "ttft_p99_s": lambda lv: dig(lv, "ttft_s", "p99"),
    "e2e_p50_s": lambda lv: dig(lv, "e2e_s", "p50"),
    "e2e_p95_s": lambda lv: dig(lv, "e2e_s", "p95"),
    "e2e_p99_s": lambda lv: dig(lv, "e2e_s", "p99"),
    "tpot_p50_s": lambda lv: dig(lv, "tpot_s", "p50"),
    "gpu_mem_peak_mib": lambda lv: dig(lv, "gpu", "mem_mib", "max"),
    "gpu_util_mean_pct": lambda lv: dig(lv, "gpu", "util_pct", "mean"),
    "gpu_sm_mhz_mean": lambda lv: dig(lv, "gpu", "sm_mhz", "mean"),
    "gpu_temp_max_c": lambda lv: dig(lv, "gpu", "temp_c", "max"),
    "gpu_power_mean_w": lambda lv: dig(lv, "gpu", "power_w", "mean"),
    "n_ok": lambda lv: lv.get("n_ok"),
    "n_failed": lambda lv: lv.get("n_failed"),
    "short_outputs": lambda lv: lv.get("short_outputs"),
}
OPEN_METRICS = {
    "realized_rate": lambda lv: lv.get("realized_rate"),
    "req_s_overall": lambda lv: lv.get("req_s_overall"),
    "out_tok_s_overall": lambda lv: lv.get("out_tok_s_overall"),
    "completion_ratio": lambda lv: lv.get("completion_ratio"),
    "ttft_p50_s": lambda lv: dig(lv, "ttft_s", "p50"),
    "ttft_p95_s": lambda lv: dig(lv, "ttft_s", "p95"),
    "e2e_p50_s": lambda lv: dig(lv, "e2e_s", "p50"),
    "e2e_p95_s": lambda lv: dig(lv, "e2e_s", "p95"),
    "e2e_p99_s": lambda lv: dig(lv, "e2e_s", "p99"),
    "gpu_mem_peak_mib": lambda lv: dig(lv, "gpu", "mem_mib", "max"),
    "n_unfinished": lambda lv: lv.get("n_unfinished"),
    "n_failed": lambda lv: lv.get("n_failed"),
    "n_dropped": lambda lv: lv.get("n_dropped"),
}


def read_jsonl_gz(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_arm_runs(root: Path) -> dict[str, dict[str, dict]]:
    """{arm: {run name: run json}} for every results/<root>/<arm>/run-NN.json."""
    out: dict[str, dict[str, dict]] = {}
    if not root.exists():
        return out
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for f in sorted(arm_dir.iterdir()):
            if RUN_FILE.match(f.name):
                runs[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        if runs:
            out[arm_dir.name] = runs
    return out


def order_arms(arms) -> list[str]:
    known = [a for a in ARM_ORDER if a in arms]
    return known + sorted(a for a in arms if a not in ARM_ORDER)


def aggregate_levels(runs: dict[str, dict], request_rows: dict[str, list[dict]]) -> dict:
    """Median and range of every metric per closed concurrency / open rate, over the runs."""
    closed: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    opened: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in runs.values():
        for lvl in run["levels"]:
            if lvl["mode"] == "closed":
                for name, fn in CLOSED_METRICS.items():
                    closed[lvl["concurrency"]][name].append(fn(lvl))
            else:
                for name, fn in OPEN_METRICS.items():
                    opened[lvl["offered_rate"]][name].append(fn(lvl))

    # pooled percentiles from the per-request records of every run
    pool: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rows in request_rows.values():
        for r in rows:
            if not r.get("ok"):
                continue
            key = ("closed", r["concurrency"]) if r["mode"] == "closed" else ("open", r["offered_rate"])
            pool[key]["ttft"].append(r["ttft_s"])
            pool[key]["e2e"].append(r["e2e_s"])
            if r.get("tpot_s") is not None:
                pool[key]["tpot"].append(r["tpot_s"])

    def finish(mode, key, metrics):
        out = {name: median_range(vals) for name, vals in metrics.items()}
        p = pool.get((mode, key))
        if p:
            out["pooled"] = {
                "n_requests": len(p["ttft"]),
                "ttft_s": summarize(p["ttft"]),
                "e2e_s": summarize(p["e2e"]),
                "tpot_s": summarize(p["tpot"]),
            }
        return out

    return {
        "closed": {str(c): finish("closed", c, closed[c]) for c in sorted(closed)},
        "open": {str(r): finish("open", r, opened[r]) for r in sorted(opened)},
    }


def closed_saturation(closed: dict) -> dict:
    """Smallest concurrency whose next level (double the workers) gains under 10% in tokens/s."""
    levels = sorted(int(c) for c in closed)
    tput = {c: dig(closed[str(c)], "out_tok_s", "median") for c in levels}
    peak_c = max(levels, key=lambda c: tput[c] or 0) if levels else None
    for lo, hi in zip(levels, levels[1:], strict=False):
        if tput[lo] and tput[hi] is not None and tput[hi] < tput[lo] * KNEE_GAIN:
            return {"concurrency": lo, "rule": f"next level gains under {int((KNEE_GAIN - 1) * 100)}%",
                    "tok_s_at_knee": tput[lo], "tok_s_at_next": tput[hi],
                    "peak_concurrency": peak_c, "peak_tok_s": tput[peak_c]}
    return {"concurrency": None, "rule": "not reached: throughput still rising at the highest level",
            "peak_concurrency": peak_c, "peak_tok_s": tput[peak_c] if peak_c else None}


def open_saturation(opened: dict) -> dict:
    """Lowest offered rate at which requests wait for a slot: median TTFT over 1 s, or unfinished work.

    TTFT is measured from the scheduled arrival, so it stays near the prefill time while the server
    keeps up and jumps once a queue forms. (A first rule, p95 end-to-end latency doubling, fired at the
    same rate for every arm because per-request latency also grows with batch size before any queue.)
    Also reports the highest completed throughput seen at any tested rate.
    """
    rates = sorted(float(r) for r in opened)
    if not rates:
        return {"rate": None, "rule": "no open-loop data"}
    best = max(rates, key=lambda r: dig(opened[str(r)], "req_s_overall", "median") or 0)
    peaks = {"peak_req_s": dig(opened[str(best)], "req_s_overall", "median"),
             "peak_out_tok_s": dig(opened[str(best)], "out_tok_s_overall", "median"),
             "peak_at_offered": best}
    rule = f"median TTFT over {OPEN_TTFT_S:g} s, or requests left unfinished after the drain limit"
    for r in rates:
        m = opened[str(r)]
        ttft = dig(m, "ttft_p50_s", "median")
        unfinished_max = dig(m, "n_unfinished", "max") or 0
        reasons = []
        if ttft is not None and ttft > OPEN_TTFT_S:
            reasons.append(f"median time to first token {ttft:.2f} s: requests wait for a slot")
        if unfinished_max > 0:
            reasons.append(f"up to {int(unfinished_max)} requests unfinished after the drain limit")
        if reasons:
            return {"rate": r, "reasons": reasons, "rule": rule, **peaks}
    return {"rate": None, "rule": "not reached within the tested rates", **peaks}


def aggregate_quality(runs: dict[str, dict]) -> dict:
    sets: dict[str, list[float]] = defaultdict(list)
    kinds: dict[str, list[float]] = defaultdict(list)
    allacc: list[float] = []
    errors = 0
    for run in runs.values():
        s = run["summary"]
        allacc.append(s["all"]["accuracy"])
        errors += s["all"]["request_errors"]
        for k, v in s["by_set"].items():
            sets[k].append(v["accuracy"])
        for k, v in s["by_kind"].items():
            kinds[k].append(v["accuracy"])
    return {"n_runs": len(runs), "all": median_range(allacc), "request_errors": errors,
            "by_set": {k: median_range(v) for k, v in sorted(sets.items())},
            "by_kind": {k: median_range(v) for k, v in sorted(kinds.items())}}


def majority_correct(items_by_run: list[list[dict]]) -> dict[str, bool]:
    """Item id -> correct in more than half the runs."""
    votes: dict[str, list[bool]] = defaultdict(list)
    for items in items_by_run:
        for it in items:
            votes[it["id"]].append(bool(it["correct"]))
    return {k: sum(v) * 2 > len(v) for k, v in votes.items()}


def instability(items_by_run: list[list[dict]]) -> int:
    """Items whose correctness differs between runs of one arm."""
    votes: dict[str, set] = defaultdict(set)
    for items in items_by_run:
        for it in items:
            votes[it["id"]].add(bool(it["correct"]))
    return sum(1 for v in votes.values() if len(v) > 1)


def paired_against(base: dict[str, bool], other: dict[str, bool]) -> dict:
    ids = sorted(set(base) & set(other))
    only_base = sum(1 for i in ids if base[i] and not other[i])
    only_other = sum(1 for i in ids if other[i] and not base[i])
    return {"n": len(ids), "only_baseline_right": only_base, "only_arm_right": only_other,
            "both_right": sum(1 for i in ids if base[i] and other[i]),
            "both_wrong": sum(1 for i in ids if not base[i] and not other[i]),
            "mcnemar_p": mcnemar_exact(only_base, only_other)}


def quality_pairs(root: Path, quality_runs: dict[str, dict[str, dict]]) -> dict:
    items: dict[str, list[list[dict]]] = {}
    for arm, runs in quality_runs.items():
        items[arm] = [read_jsonl_gz(root / "quality" / arm / f"{name}.items.jsonl.gz")
                      for name in sorted(runs)]
    out = {"instability_items": {a: instability(v) for a, v in items.items()}, "vs_fp16": {}}
    if "fp16" in items and items["fp16"] and items["fp16"][0]:
        base = majority_correct(items["fp16"])
        for arm in items:
            if arm != "fp16":
                out["vs_fp16"][arm] = paired_against(base, majority_correct(items[arm]))
    return out


def ratio(a, b):
    return (a / b) if (a is not None and b) else None


def summarize_results(results: Path, plots: bool = True) -> dict:
    results = Path(results)
    runs = load_arm_runs(results / "runs")
    quality = load_arm_runs(results / "quality")
    summary: dict = {"generated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "arms": {}}
    for arm in order_arms(runs):
        rows = {name: read_jsonl_gz(results / "runs" / arm / f"{name}.requests.jsonl.gz")
                for name in runs[arm]}
        agg = aggregate_levels(runs[arm], rows)
        agg["n_runs"] = len(runs[arm])
        agg["run_names"] = sorted(runs[arm])
        agg["saturation"] = {"closed": closed_saturation(agg["closed"]),
                             "open": open_saturation(agg["open"])}
        summary["arms"][arm] = agg
    for arm in order_arms(quality):
        summary["arms"].setdefault(arm, {})["quality"] = aggregate_quality(quality[arm])
    summary["quality_pairs"] = quality_pairs(results, quality)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8",
                                          newline="\n")
    (results / "summary.md").write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    write_runs_csv(results, runs)
    if plots:
        from .plots import make_plots
        make_plots(summary, results / "plots")
    return summary


def write_runs_csv(results: Path, runs: dict[str, dict[str, dict]]) -> None:
    cols = ["arm", "run", "mode", "level", "n_ok", "n_failed", "out_tok_s", "req_s", "ttft_p50_s",
            "ttft_p95_s", "ttft_p99_s", "e2e_p50_s", "e2e_p95_s", "e2e_p99_s", "gpu_mem_peak_mib",
            "n_unfinished"]
    with open(results / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for arm in order_arms(runs):
            for name in sorted(runs[arm]):
                for lvl in runs[arm][name]["levels"]:
                    closed = lvl["mode"] == "closed"
                    w.writerow([
                        arm, name, lvl["mode"],
                        lvl["concurrency"] if closed else lvl["offered_rate"],
                        lvl.get("n_ok"), lvl.get("n_failed"),
                        lvl.get("out_tok_s") if closed else lvl.get("out_tok_s_overall"),
                        lvl.get("req_s") if closed else lvl.get("req_s_overall"),
                        dig(lvl, "ttft_s", "p50"), dig(lvl, "ttft_s", "p95"),
                        dig(lvl, "ttft_s", "p99"), dig(lvl, "e2e_s", "p50"),
                        dig(lvl, "e2e_s", "p95"), dig(lvl, "e2e_s", "p99"),
                        dig(lvl, "gpu", "mem_mib", "max"), lvl.get("n_unfinished"),
                    ])


def ms(summary):
    """median_range dict in seconds -> same dict in milliseconds."""
    if not summary or summary.get("median") is None:
        return summary
    return {**summary, "median": summary["median"] * 1000, "min": summary["min"] * 1000,
            "max": summary["max"] * 1000}


def pct(summary):
    if not summary or summary.get("median") is None:
        return summary
    return {**summary, "median": summary["median"] * 100, "min": summary["min"] * 100,
            "max": summary["max"] * 100}


def render_markdown(summary: dict) -> str:
    L = ["# Results summary", "",
         "Each cell is the median over the runs of one configuration, with the range (min-max) in "
         "brackets. Times are per-run percentiles taken over that run's requests; `pooled` tables "
         "take the percentile over all requests of all runs instead. Generated by "
         "`servebench summarize` from the run files; do not edit by hand.", ""]
    arms = summary["arms"]
    for arm, a in arms.items():
        if "closed" not in a:
            continue
        L += [f"## {arm}: closed loop ({a['n_runs']} runs)", "",
              "| workers | output tok/s | req/s | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms | "
              "e2e p50 s | e2e p95 s | e2e p99 s | TPOT p50 ms | GPU mem peak MiB | failed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, m in a["closed"].items():
            L.append(f"| {c} | {fmt_range(m['out_tok_s'], 1)} | {fmt_range(m['req_s'], 2)} | "
                     f"{fmt_range(ms(m['ttft_p50_s']), 0)} | {fmt_range(ms(m['ttft_p95_s']), 0)} | "
                     f"{fmt_range(ms(m['ttft_p99_s']), 0)} | {fmt_range(m['e2e_p50_s'], 2)} | "
                     f"{fmt_range(m['e2e_p95_s'], 2)} | {fmt_range(m['e2e_p99_s'], 2)} | "
                     f"{fmt_range(ms(m['tpot_p50_s']), 1)} | {fmt_range(m['gpu_mem_peak_mib'], 0)} | "
                     f"{fmt_range(m['n_failed'], 0)} |")
        L.append("")
        L += [f"### {arm}: closed loop, pooled over all runs", "",
              "| workers | requests | TTFT p50 ms | TTFT p95 ms | TTFT p99 ms | e2e p50 s | "
              "e2e p95 s | e2e p99 s |", "|---|---|---|---|---|---|---|---|"]
        for c, m in a["closed"].items():
            p = m.get("pooled")
            if not p:
                continue
            L.append(f"| {c} | {p['n_requests']} | {p['ttft_s']['p50'] * 1000:.0f} | "
                     f"{p['ttft_s']['p95'] * 1000:.0f} | {p['ttft_s']['p99'] * 1000:.0f} | "
                     f"{p['e2e_s']['p50']:.2f} | {p['e2e_s']['p95']:.2f} | {p['e2e_s']['p99']:.2f} |")
        L.append("")
        L += [f"## {arm}: open loop (Poisson arrivals, {a['n_runs']} runs)", "",
              "| offered req/s (nominal) | realized req/s | completed req/s | output tok/s | "
              "completed % | TTFT p50 ms | TTFT p95 ms | e2e p50 s | e2e p95 s | e2e p99 s | "
              "unfinished | failed |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r, m in a["open"].items():
            L.append(f"| {r} | {fmt_range(m['realized_rate'], 2)} | {fmt_range(m['req_s_overall'], 2)} | "
                     f"{fmt_range(m['out_tok_s_overall'], 1)} | {fmt_range(pct(m['completion_ratio']), 0)} | "
                     f"{fmt_range(ms(m['ttft_p50_s']), 0)} | {fmt_range(ms(m['ttft_p95_s']), 0)} | "
                     f"{fmt_range(m['e2e_p50_s'], 2)} | {fmt_range(m['e2e_p95_s'], 2)} | "
                     f"{fmt_range(m['e2e_p99_s'], 2)} | {fmt_range(m['n_unfinished'], 0)} | "
                     f"{fmt_range(m['n_failed'], 0)} |")
        L.append("")
    cmp_arms = [a for a in arms if "closed" in arms[a]]
    if "fp16" in cmp_arms and len(cmp_arms) > 1:
        L += ["## Against fp16 (ratio of medians)", "",
              "| arm | workers | output tok/s | e2e p50 | TTFT p50 | GPU mem peak |",
              "|---|---|---|---|---|---|"]
        base = arms["fp16"]["closed"]
        for arm in cmp_arms:
            if arm == "fp16":
                continue
            for c, m in arms[arm]["closed"].items():
                b = base.get(c)
                if not b:
                    continue
                r = [ratio(dig(m, k, "median"), dig(b, k, "median"))
                     for k in ("out_tok_s", "e2e_p50_s", "ttft_p50_s", "gpu_mem_peak_mib")]
                L.append(f"| {arm} | {c} | " + " | ".join("-" if x is None else f"{x:.2f}x" for x in r)
                         + " |")
        L.append("")
    L += ["## Saturation", ""]
    for arm, a in arms.items():
        if "saturation" not in a:
            continue
        c, o = a["saturation"]["closed"], a["saturation"]["open"]
        cc = (f"the knee is at {c['concurrency']} workers ({c['tok_s_at_knee']:.1f} tok/s; the next level "
              f"gives {c['tok_s_at_next']:.1f})" if c["concurrency"] else "no knee up to 16 workers")
        oo = (f"requests start to wait at {o['rate']:g} req/s offered ({'; '.join(o['reasons'])})"
              if o.get("rate") else o["rule"])
        pk = (f" The most it completed at any tested rate was {o['peak_req_s']:.2f} req/s "
              f"({o['peak_out_tok_s']:.0f} tok/s) at {o['peak_at_offered']:g} req/s offered."
              if o.get("peak_req_s") else "")
        L += [f"- **{arm}**: closed loop: {cc}, peak {c['peak_tok_s']:.1f} tok/s at "
              f"{c['peak_concurrency']} workers. Open loop: {oo}.{pk}"]
    L += ["", "Rules: the closed loop is saturated at the smallest worker count whose next level "
          f"(double the workers) gains under {int((KNEE_GAIN - 1) * 100)}% in tokens/s. The open loop is "
          f"saturated at the lowest offered rate where the median time to first token, measured from "
          f"the scheduled arrival, is over {OPEN_TTFT_S:g} s, or where any run left requests "
          f"unfinished after the drain limit.", ""]
    q_arms = [a for a in arms if "quality" in arms[a]]
    if q_arms:
        L += ["## Quality (fixed question sets, temperature 0)", "",
              "Accuracy, median (min-max) over the runs.", "",
              "| arm | runs | all | " + " | ".join(
                  sorted({k for a in q_arms for k in arms[a]["quality"]["by_set"]})) + " |"]
        sets = sorted({k for a in q_arms for k in arms[a]["quality"]["by_set"]})
        L.append("|---|---|---|" + "---|" * len(sets))
        for arm in q_arms:
            q = arms[arm]["quality"]
            L.append(f"| {arm} | {q['n_runs']} | {fmt_range(pct(q['all']), 1)} | " + " | ".join(
                fmt_range(pct(q['by_set'].get(s)), 1) for s in sets) + " |")
        L.append("")
        kinds = sorted({k for a in q_arms for k in arms[a]["quality"]["by_kind"]})
        L += ["| arm | " + " | ".join(kinds) + " |", "|---|" + "---|" * len(kinds)]
        for arm in q_arms:
            q = arms[arm]["quality"]
            L.append(f"| {arm} | " + " | ".join(fmt_range(pct(q['by_kind'].get(k)), 1) for k in kinds)
                     + " |")
        L.append("")
        qp = summary.get("quality_pairs", {})
        if qp.get("vs_fp16"):
            L += ["Paired comparison with fp16 on the items each arm got right in most of its runs "
                  "(exact McNemar test on the items where they differ):", "",
                  "| arm | items | both right | only fp16 right | only arm right | both wrong | p |",
                  "|---|---|---|---|---|---|---|"]
            for arm, p in qp["vs_fp16"].items():
                L.append(f"| {arm} | {p['n']} | {p['both_right']} | {p['only_baseline_right']} | "
                         f"{p['only_arm_right']} | {p['both_wrong']} | {p['mcnemar_p']:.3f} |")
            L.append("")
        if qp.get("instability_items"):
            L += ["Items whose correctness changed between runs of the same arm "
                  "(greedy decoding is not bit-for-bit repeatable when batches differ): "
                  + ", ".join(f"{a} {n}" for a, n in qp["instability_items"].items()) + ".", ""]
    return "\n".join(L) + "\n"
