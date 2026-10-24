import importlib.util
import json
from pathlib import Path

import pytest

from servebench import report
from test_report import write_arm, write_quality

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("render_readme_results",
                                              ROOT / "scripts" / "render_readme_results.py")
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)

README = "# title\n\nintro text\n\n<!-- results:start -->\nOLD\n<!-- results:end -->\n\nafter text\n"


@pytest.fixture
def results(tmp_path):
    write_arm(tmp_path, "fp16", {1: [10] * 5, 2: [19] * 5, 16: [100, 101, 102, 103, 104]},
              {2.0: [3.0] * 5, 8.0: [9.0, 9.5, 10, 10.5, 11]})
    write_arm(tmp_path, "awq", {1: [20] * 5, 2: [38] * 5, 16: [150] * 5}, {2.0: [2.0] * 5})
    write_quality(tmp_path, "fp16", [set("abcdefgh")] * 3)
    write_quality(tmp_path, "awq", [set("abcdefg")] * 3)
    for arm, gib, tokens in (("fp16", 2.98, 42432), ("awq", 1.10, 115072)):
        d = tmp_path / "server" / arm
        d.mkdir(parents=True)
        (d / "facts.json").write_text(json.dumps({"model_loading_gib": gib, "kv_cache_tokens": tokens}))
    report.summarize_results(tmp_path, plots=False)
    return tmp_path


def block_for(results):
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    return render.render(summary, results)


def test_block_is_wrapped_in_the_markers_and_carries_the_measured_cells(results):
    block = block_for(results)
    assert block.startswith(render.START) and block.endswith(render.END)
    assert "| weights in GPU memory, GiB | 2.98 | 1.10 |" in block
    assert "| KV cache that fits, tokens | 42,432 | 115,072 |" in block
    assert "102.0 (100.0-104.0)" in block                       # fp16 tok/s at 16 workers, from the files
    assert "| 16 | 150.0 (150.0-150.0) |" in block               # awq closed-loop row
    assert "**fp16**" in block and "**awq**" in block
    assert "### Quality" in block and "80.0 (80.0-80.0)" in block


def test_every_number_in_the_block_comes_from_the_summary(results):
    """No stray literal: change one run's value and the block changes with it."""
    before = block_for(results)
    path = results / "runs" / "fp16" / "run-01.json"
    data = json.loads(path.read_text())
    for lvl in data["levels"]:
        if lvl["mode"] == "closed" and lvl["concurrency"] == 16:
            lvl["out_tok_s"] = 999.0
    path.write_text(json.dumps(data))
    report.summarize_results(results, plots=False)
    after = block_for(results)
    assert before != after and "999.0" in after


def test_replace_block_keeps_text_outside_and_is_idempotent(results):
    block = block_for(results)
    once = render.replace_block(README, block)
    twice = render.replace_block(once, block)
    assert once == twice
    assert once.startswith("# title\n\nintro text\n\n") and once.endswith("\n\nafter text\n")
    assert "OLD" not in once


def test_missing_markers_are_refused():
    with pytest.raises(SystemExit):
        render.replace_block("no markers here", "x")


def test_saturation_text():
    assert "not reached" in render.sat_closed({"concurrency": None, "peak_tok_s": 600.0})
    assert render.sat_closed({"concurrency": 8, "tok_s_at_knee": 700.0, "tok_s_at_next": 720.0}).startswith(
        "8 workers")
    assert render.sat_open({"rate": 8.0}) == "8 req/s offered"
    assert render.sat_open({"rate": None}).startswith("not reached")
