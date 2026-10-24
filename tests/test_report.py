import csv
import gzip
import json

import pytest

from servebench import report


def closed_level(c, tok_s, e2e_p50=2.0):
    return {"mode": "closed", "concurrency": c, "out_tok_s": tok_s, "req_s": tok_s / 128,
            "ttft_s": {"p50": 0.1, "p95": 0.2, "p99": 0.3}, "e2e_s": {"p50": e2e_p50, "p95": 3.0, "p99": 4.0},
            "tpot_s": {"p50": 0.02}, "gpu": {"mem_mib": {"max": 4000.0 + c}, "util_pct": {"mean": 90.0},
                                              "sm_mhz": {"mean": 2000.0}, "temp_c": {"max": 70.0},
                                              "power_w": {"mean": 50.0}},
            "n_ok": 32, "n_failed": 0, "short_outputs": 0}


def open_level(rate, p95, completion=1.0, unfinished=0):
    return {"mode": "open", "offered_rate": rate, "realized_rate": rate * 1.01,
            "req_s_overall": rate * 0.95, "out_tok_s_overall": rate * 120, "completion_ratio": completion,
            "ttft_s": {"p50": 0.1, "p95": 0.2}, "e2e_s": {"p50": 1.0, "p95": p95, "p99": p95 * 1.2},
            "gpu": {"mem_mib": {"max": 4100.0}}, "n_unfinished": unfinished, "n_failed": 0,
            "n_dropped": 0}


def write_arm(results, arm, tputs, opens, n_runs=5):
    """tputs: {concurrency: [tok/s per run]}; opens: {rate: [p95 per run]}."""
    d = results / "runs" / arm
    d.mkdir(parents=True)
    for i in range(n_runs):
        levels = [closed_level(c, vals[i]) for c, vals in tputs.items()]
        levels += [open_level(r, vals[i]) for r, vals in opens.items()]
        (d / f"run-{i + 1:02d}.json").write_text(json.dumps({"arm": arm, "levels": levels}))
        rows = []
        for c in tputs:
            for k in range(4):
                rows.append({"mode": "closed", "concurrency": c, "ok": True, "ttft_s": 0.1 * (k + 1),
                             "e2e_s": float(k + 1), "tpot_s": 0.02})
        with gzip.open(d / f"run-{i + 1:02d}.requests.jsonl.gz", "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (d / "warmup.json").write_text("not a run file and must be ignored")


def test_median_and_range_per_level(tmp_path):
    write_arm(tmp_path, "fp16", {1: [10, 11, 12, 13, 14], 4: [30, 31, 32, 33, 34]}, {})
    s = report.summarize_results(tmp_path, plots=False)
    c1 = s["arms"]["fp16"]["closed"]["1"]["out_tok_s"]
    assert (c1["median"], c1["min"], c1["max"], c1["n"]) == (12, 10, 14, 5)
    assert s["arms"]["fp16"]["n_runs"] == 5
    assert s["arms"]["fp16"]["run_names"] == [f"run-0{i}" for i in range(1, 6)]
    assert s["arms"]["fp16"]["closed"]["4"]["gpu_mem_peak_mib"]["median"] == 4004.0


def test_pooled_percentiles_use_every_request_of_every_run(tmp_path):
    write_arm(tmp_path, "fp16", {1: [10] * 5}, {})
    s = report.summarize_results(tmp_path, plots=False)
    pooled = s["arms"]["fp16"]["closed"]["1"]["pooled"]
    assert pooled["n_requests"] == 20           # 4 rows per run x 5 runs
    assert pooled["e2e_s"]["max"] == 4.0 and pooled["e2e_s"]["p50"] == 2.5


def test_closed_saturation_finds_the_first_flat_step():
    closed = {str(c): {"out_tok_s": {"median": v}} for c, v in
              {1: 40.0, 2: 78.0, 4: 150.0, 8: 160.0, 16: 162.0}.items()}
    sat = report.closed_saturation(closed)
    assert sat["concurrency"] == 4 and sat["tok_s_at_next"] == 160.0
    assert sat["peak_concurrency"] == 16


def test_closed_saturation_not_reached_when_still_rising():
    closed = {str(c): {"out_tok_s": {"median": v}} for c, v in
              {1: 10.0, 2: 19.0, 4: 36.0}.items()}
    sat = report.closed_saturation(closed)
    assert sat["concurrency"] is None and "not reached" in sat["rule"]


def test_open_saturation_by_latency_doubling_and_by_unfinished_work():
    def lvl(p95, unfinished=0, completion=1.0):
        return {"e2e_p95_s": {"median": p95}, "n_unfinished": {"max": unfinished},
                "completion_ratio": {"median": completion}}

    by_latency = {"2.0": lvl(3.0), "4.0": lvl(3.4), "6.0": lvl(6.5), "8.0": lvl(20.0)}
    assert report.open_saturation(by_latency)["rate"] == 6.0
    by_unfinished = {"2.0": lvl(3.0), "4.0": lvl(3.1, unfinished=2)}
    assert report.open_saturation(by_unfinished)["rate"] == 4.0
    by_completion = {"2.0": lvl(3.0), "4.0": lvl(3.2, completion=0.9)}
    assert report.open_saturation(by_completion)["rate"] == 4.0
    calm = {"2.0": lvl(3.0), "4.0": lvl(3.5)}
    assert report.open_saturation(calm)["rate"] is None
    assert report.open_saturation({})["rate"] is None


def test_open_levels_aggregate_and_saturation_flows_into_summary(tmp_path):
    write_arm(tmp_path, "awq", {1: [10] * 5},
              {2.0: [3.0] * 5, 4.0: [3.1] * 5, 8.0: [9.0, 9.5, 10, 10.5, 11]})
    s = report.summarize_results(tmp_path, plots=False)
    a = s["arms"]["awq"]
    assert a["open"]["8.0"]["e2e_p95_s"]["median"] == 10
    assert a["saturation"]["open"]["rate"] == 8.0


def test_runs_csv_has_one_row_per_run_and_level(tmp_path):
    write_arm(tmp_path, "fp16", {1: [10] * 5, 4: [30] * 5}, {2.0: [3.0] * 5})
    report.summarize_results(tmp_path, plots=False)
    rows = list(csv.DictReader(open(tmp_path / "runs.csv", encoding="utf-8")))
    assert len(rows) == 5 * 3
    assert {r["mode"] for r in rows} == {"closed", "open"}
    assert rows[0]["arm"] == "fp16" and rows[0]["run"] == "run-01"


def write_quality(results, arm, per_run_correct):
    """per_run_correct: list (one per run) of sets of correct item ids out of ids a..j."""
    d = results / "quality" / arm
    d.mkdir(parents=True)
    ids = list("abcdefghij")
    for i, ok_ids in enumerate(per_run_correct):
        items = [{"id": x, "set": "s", "kind": "k", "ok": True, "correct": x in ok_ids} for x in ids]
        acc = len(ok_ids) / 10
        summary = {"all": {"accuracy": acc, "request_errors": 0},
                   "by_set": {"s": {"accuracy": acc}}, "by_kind": {"s/k": {"accuracy": acc}}}
        (d / f"run-{i + 1:02d}.json").write_text(json.dumps({"arm": arm, "summary": summary}))
        with gzip.open(d / f"run-{i + 1:02d}.items.jsonl.gz", "wt") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")


def test_quality_aggregation_and_paired_comparison(tmp_path):
    base = set("abcdefgh")                      # fp16 gets 8 of 10 in every run
    write_quality(tmp_path, "fp16", [base] * 3)
    other = set("abcdef") | {"i"}               # awq: loses g,h and gains i; one run also flips j
    write_quality(tmp_path, "awq", [other, other, other | {"j"}])
    s = report.summarize_results(tmp_path, plots=False)
    q = s["arms"]["awq"]["quality"]
    assert q["n_runs"] == 3
    assert q["all"]["median"] == pytest.approx(0.7) and q["all"]["max"] == pytest.approx(0.8)
    pair = s["quality_pairs"]["vs_fp16"]["awq"]
    assert (pair["only_baseline_right"], pair["only_arm_right"]) == (2, 1)
    assert pair["both_right"] == 6 and pair["both_wrong"] == 1 and pair["n"] == 10
    assert pair["mcnemar_p"] == pytest.approx(1.0)
    assert s["quality_pairs"]["instability_items"] == {"awq": 1, "fp16": 0}


def test_markdown_has_tables_and_the_rules(tmp_path):
    write_arm(tmp_path, "fp16", {1: [10] * 5, 2: [19] * 5, 4: [20] * 5}, {2.0: [3.0] * 5})
    write_arm(tmp_path, "awq", {1: [20] * 5, 2: [38] * 5, 4: [40] * 5}, {2.0: [2.0] * 5})
    write_quality(tmp_path, "fp16", [set("abcdefgh")] * 2)
    report.summarize_results(tmp_path, plots=False)
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## fp16: closed loop (5 runs)" in md and "## awq: open loop" in md
    assert "## Against fp16 (ratio of medians)" in md and "2.00x" in md
    assert "## Saturation" in md and "Rules:" in md
    assert "## Quality" in md and "80.0 (80.0-80.0)" in md


def test_plots_are_written(tmp_path):
    write_arm(tmp_path, "fp16", {1: [10] * 5, 2: [19] * 5}, {2.0: [3.0] * 5, 4.0: [5.0] * 5})
    report.summarize_results(tmp_path, plots=True)
    names = {p.name for p in (tmp_path / "plots").iterdir()}
    assert names == {"throughput-vs-concurrency.png", "latency-vs-concurrency.png", "open-loop.png"}
    assert all((tmp_path / "plots" / n).stat().st_size > 2000 for n in names)


def test_empty_results_directory_does_not_crash(tmp_path):
    s = report.summarize_results(tmp_path, plots=False)
    assert s["arms"] == {}
    assert (tmp_path / "summary.md").exists()
