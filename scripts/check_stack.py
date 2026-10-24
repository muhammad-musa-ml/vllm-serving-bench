"""Check the monitoring stack against a live run and write the evidence to results/evidence/.

    python scripts/check_stack.py --arm fp16 [--out results/evidence]

What it checks, and writes to <out>/<arm>-stack-check.json:
  * Prometheus' targets (the model server and the GPU exporter) are up;
  * every query in the provisioned Grafana dashboard evaluates in Prometheus without error, and how
    many series it returned (a panel with no series is listed, not hidden);
  * Grafana has the dashboard and its Prometheus datasource answers a query sent through Grafana;
  * the server's own token counter, read from Prometheus, against the tokens the load harness
    counted over the same window (two views of the same traffic).
"""
import argparse
import datetime
import gzip
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROM = "http://127.0.0.1:18490"
GRAFANA = "http://127.0.0.1:18491"
DASHBOARD = ROOT / "observability" / "grafana" / "dashboards" / "vllm-serving.json"


def get_json(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url, body, timeout=20):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def panel_targets(dashboard):
    """(panel title, refId, expr) for every Prometheus target, rows included."""
    out = []

    def walk(panels):
        for p in panels:
            for t in p.get("targets", []):
                if t.get("expr"):
                    out.append((p.get("title", "?"), t.get("refId", "A"), t["expr"]))
            walk(p.get("panels", []))

    walk(dashboard.get("panels", []))
    return out


def expand_macros(expr):
    return expr.replace("$__rate_interval", "30s").replace("$__interval", "15s")


def prom_query(expr, at=None):
    params = {"query": expand_macros(expr)}
    if at is not None:
        params["time"] = f"{at:.3f}"
    return get_json(f"{PROM}/api/v1/query?" + urllib.parse.urlencode(params))


def read_gz(path):
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_utc(text):
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.UTC).timestamp()


def token_crosscheck(results, arm):
    """Server-counted vs harness-counted generation tokens over the timed passes of one arm."""
    run_dir = results / "runs" / arm
    runs = sorted(run_dir.glob("run-[0-9][0-9].json"))
    if not runs:
        return {"available": False, "reason": "no run files"}
    meta = [json.loads(p.read_text(encoding="utf-8")) for p in runs]
    start = min(parse_utc(m["started_utc"]) for m in meta)
    end = max(parse_utc(m["finished_utc"]) for m in meta)
    harness_tokens = 0
    harness_requests = 0
    for p in runs:
        for r in read_gz(p.with_name(p.stem + ".requests.jsonl.gz")):
            harness_requests += 1
            if r.get("ok"):
                harness_tokens += r.get("completion_tokens") or 0
    window = int(end - start) + 10
    # a small margin after the last pass lets the final scrape land
    at = end + 12
    expr = f"sum(increase(vllm:generation_tokens_total[{window + 12}s]))"
    result = prom_query(expr, at=at)
    series = result.get("data", {}).get("result", [])
    server_tokens = float(series[0]["value"][1]) if series else None
    return {"available": server_tokens is not None, "window_start_utc": runs and meta[0]["started_utc"],
            "window_end_utc": max(m["finished_utc"] for m in meta), "expr": expr,
            "server_generation_tokens": server_tokens, "harness_completion_tokens": harness_tokens,
            "harness_requests": harness_requests,
            "ratio_server_to_harness": (server_tokens / harness_tokens)
            if server_tokens is not None and harness_tokens else None,
            "note": "increase() extrapolates at the window edges and counts tokens of requests the "
                    "harness cancelled; expect the server figure to be slightly higher"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default="results/evidence")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    results = (ROOT / args.results).resolve()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    evidence = {"arm": args.arm,
                "checked_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}
    ok = True

    targets = get_json(f"{PROM}/api/v1/targets")["data"]["activeTargets"]
    evidence["targets"] = {t["labels"]["job"]: {"health": t["health"], "last_error": t["lastError"],
                                                "scrape_url": t["scrapeUrl"]} for t in targets}
    for job in ("vllm", "gpu"):
        if evidence["targets"].get(job, {}).get("health") != "up":
            ok = False

    if DASHBOARD.exists():
        dash = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        queries = []
        for title, ref, expr in panel_targets(dash):
            entry = {"panel": title, "refId": ref, "expr": expr}
            try:
                res = prom_query(expr)
                entry["status"] = res.get("status")
                entry["series"] = len(res.get("data", {}).get("result", []))
            except Exception as exc:  # noqa: BLE001 - recorded as evidence, then reported
                entry["status"] = f"error: {exc}"
                entry["series"] = 0
            if entry["status"] != "success":
                ok = False
            queries.append(entry)
        evidence["dashboard_queries"] = {"total": len(queries),
                                         "errors": sum(1 for q in queries if q["status"] != "success"),
                                         "without_series": [q["panel"] for q in queries if not q["series"]],
                                         "queries": queries}
        try:
            health = get_json(f"{GRAFANA}/api/health")
            found = get_json(f"{GRAFANA}/api/search?" + urllib.parse.urlencode({"type": "dash-db"}))
            via = post_json(f"{GRAFANA}/api/ds/query", {
                "queries": [{"refId": "A", "datasource": {"uid": "prometheus", "type": "prometheus"},
                             "expr": "up", "instant": True}], "from": "now-5m", "to": "now"})
            frames = via.get("results", {}).get("A", {}).get("frames", [])
            evidence["grafana"] = {"version": health.get("version"),
                                   "dashboards": [d["title"] for d in found],
                                   "datasource_query_frames": len(frames)}
            if not found or not frames:
                ok = False
        except Exception as exc:  # noqa: BLE001
            evidence["grafana"] = {"error": str(exc)}
            ok = False
    else:
        evidence["dashboard_queries"] = {"error": f"{DASHBOARD.name} not found"}
        ok = False

    try:
        evidence["token_crosscheck"] = token_crosscheck(results, args.arm)
    except Exception as exc:  # noqa: BLE001
        evidence["token_crosscheck"] = {"available": False, "reason": str(exc)}

    evidence["ok"] = ok
    path = out / f"{args.arm}-stack-check.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}: ok={ok}; dashboard queries "
          f"{evidence.get('dashboard_queries', {}).get('total')}, errors "
          f"{evidence.get('dashboard_queries', {}).get('errors')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
