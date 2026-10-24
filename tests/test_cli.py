import asyncio
import gzip
import json

from servebench import cli


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_cli(*argv):
    # the CLI starts its own event loop, so it runs in a worker thread beside the fake server
    return await asyncio.to_thread(cli.main, [str(a) for a in argv])


async def test_bench_writes_run_file_requests_and_docker_ps(fake, tmp_path):
    fake.token_delay = 0.001
    out = tmp_path / "runs" / "fp16"
    code = await run_cli("bench", "--url", fake.url, "--arm", "fp16", "--run", 3, "--out", out,
                         "--levels", "1,2", "--open-rates", "10", "--open-duration", 1, "--drain", 5,
                         "--cooldown", 0, "--max-tokens", 8)
    assert code == 0
    run = json.loads((out / "run-03.json").read_text(encoding="utf-8"))
    assert run["arm"] == "fp16" and run["name"] == "run-03"
    assert [(lv["mode"], lv.get("concurrency") or lv.get("offered_rate")) for lv in run["levels"]] == [
        ("closed", 1), ("closed", 2), ("open", 10.0)]
    assert run["settings"]["fixed_length"] is True and run["settings"]["max_tokens"] == 8
    assert run["settings"]["prompt_set"]["count"] == 256
    assert run["harness"]["version"] and run["started_utc"].endswith("Z")
    assert run["lock"] is None
    rows = read_rows(out / "run-03.requests.jsonl.gz")
    assert len(rows) == fake.calls
    assert {r["mode"] for r in rows} == {"closed", "open"}
    assert all("ttft_s" in r and "concurrency" in r or "offered_rate" in r for r in rows)
    assert (out / "run-03.docker-ps.txt").read_text(encoding="utf-8").startswith("# captured ")


async def test_bench_holds_and_releases_the_lock(fake, tmp_path):
    fake.token_delay = 0.001
    lock = tmp_path / "m.lock"
    out = tmp_path / "runs" / "awq"
    await run_cli("bench", "--url", fake.url, "--arm", "awq", "--out", out, "--levels", "1",
                  "--open-rates", "", "--cooldown", 0, "--max-tokens", 4, "--lock-file", lock,
                  "--lock-owner", "P4")
    assert not lock.exists()                       # released
    run = json.loads((out / "run-01.json").read_text(encoding="utf-8"))
    assert run["lock"]["owner"] == "P4" and run["lock"]["waited_s"] >= 0
    assert run["levels"][0]["n_ok"] == 32


async def test_warmup_is_written_apart_from_the_numbered_runs(fake, tmp_path):
    fake.token_delay = 0.001
    out = tmp_path / "runs" / "gptq"
    await run_cli("bench", "--url", fake.url, "--arm", "gptq", "--out", out, "--warmup",
                  "--cooldown", 0, "--max-tokens", 4)
    run = json.loads((out / "warmup.json").read_text(encoding="utf-8"))
    assert run["name"] == "warmup" and [lv["n_requests"] for lv in run["levels"]] == [24]
    assert not list(out.glob("run-*.json"))


async def test_quality_command_scores_and_writes_items(fake, tmp_path):
    fake.reply_fn = lambda body: "A"
    out = tmp_path / "quality" / "fp16"
    code = await run_cli("quality", "--url", fake.url, "--arm", "fp16", "--run", 2, "--out", out,
                         "--sets", "synthetic")
    assert code == 0
    run = json.loads((out / "run-02.json").read_text(encoding="utf-8"))
    assert run["settings"]["n_tasks"] == 120 and run["settings"]["temperature"] == 0.0
    assert run["summary"]["all"]["n"] == 120
    items = read_rows(out / "run-02.items.jsonl.gz")
    assert len(items) == 120 and all("reply" in i for i in items)


async def test_summarize_command_builds_the_summary_files(fake, tmp_path):
    fake.token_delay = 0.001
    results = tmp_path / "results"
    for n in (1, 2):
        await run_cli("bench", "--url", fake.url, "--arm", "fp16", "--run", n,
                      "--out", results / "runs" / "fp16", "--levels", "1", "--open-rates", "",
                      "--cooldown", 0, "--max-tokens", 4)
    code = await run_cli("summarize", "--results", results, "--no-plots")
    assert code == 0
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert summary["arms"]["fp16"]["n_runs"] == 2
    assert summary["arms"]["fp16"]["closed"]["1"]["out_tok_s"]["n"] == 2
    assert (results / "summary.md").exists() and (results / "runs.csv").exists()


def test_gzip_output_is_byte_identical_for_identical_rows(tmp_path):
    rows = [{"a": 1}, {"b": [1, 2]}]
    cli.write_jsonl_gz(tmp_path / "x.gz", rows)
    cli.write_jsonl_gz(tmp_path / "y.gz", rows)
    assert (tmp_path / "x.gz").read_bytes() == (tmp_path / "y.gz").read_bytes()


def test_parse_list():
    assert cli.parse_list("1,2, 4", int) == [1, 2, 4]
    assert cli.parse_list("", float) == []
    assert cli.parse_list("2.5,3", float) == [2.5, 3.0]
