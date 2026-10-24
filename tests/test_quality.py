import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from servebench import quality

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("reply,expected", [
    ("B", "B"), ("B.", "B"), ("(C)", "C"), ("D) the fourth option", "D"),
    ("The answer is A.", "A"), ("Answer: C", "C"), ("answer is D", "D"),
    ("I think it is B because of that", "B"),
    ("  \n A", "A"), ("None of these", None), ("", None), ("E", None),
])
def test_extract_choice(reply, expected):
    assert quality.extract_choice(reply) == expected


def test_extract_choice_does_not_take_the_article_a_in_lowercase():
    assert quality.extract_choice("a bird in the hand") is None


@pytest.mark.parametrize("reply,expected", [
    ("636", 636), ("The answer is 636.", 636), ("347 + 289 = 636", 636), ("-12", -12),
    ("1,630", 1630), ("no number", None), ("", None), ("3094 apples and 12 pears", 12),
])
def test_extract_int(reply, expected):
    assert quality.extract_int(reply) == expected


def test_score_int_and_text():
    assert quality.score_task({"scorer": "int", "expected": 714}, "714")
    assert not quality.score_task({"scorer": "int", "expected": 714}, "715")
    assert not quality.score_task({"scorer": "int", "expected": 714}, "")
    text = {"scorer": "text", "expected": "Leeds"}
    assert quality.score_task(text, "Leeds.") and quality.score_task(text, '"leeds"')
    assert not quality.score_task(text, "Porto")
    num = {"scorer": "text", "expected": "39451"}
    assert quality.score_task(num, "#39451")


def test_score_exact_is_strict():
    task = {"scorer": "exact", "expected": "QWERTY"}
    assert quality.score_task(task, "QWERTY\n")
    assert not quality.score_task(task, "qwerty")
    assert not quality.score_task(task, "QWERTY.")


def test_score_json_accepts_fences_and_rejects_extras():
    task = {"scorer": "json", "expected": {"name": "Ana", "count": 7}}
    assert quality.score_task(task, '{"name": "Ana", "count": 7}')
    assert quality.score_task(task, '```json\n{"count": 7, "name": "Ana"}\n```')
    assert not quality.score_task(task, '{"name": "Ana", "count": "7"}')
    assert not quality.score_task(task, '{"name": "Ana", "count": 7, "x": 1}')
    assert not quality.score_task(task, "name Ana count 7")
    assert not quality.score_task(task, 'Here you go: {"name": "Ana", "count": 7}')


def test_score_lines():
    task = {"scorer": "lines", "expected": {"n": 3, "prefix": "- "}}
    assert quality.score_task(task, "- apple\n- pear\n- plum")
    assert quality.score_task(task, "- apple\n\n- pear\n- plum\n")
    assert not quality.score_task(task, "- apple\n- pear")
    assert not quality.score_task(task, "1. apple\n2. pear\n3. plum")
    assert not quality.score_task(task, "- apple\n- pear\n- plum\n- fig")


def test_score_mcq_uses_answer_field_and_unknown_scorer_raises():
    assert quality.score_task({"scorer": "mcq", "answer": "C"}, "C")
    assert not quality.score_task({"scorer": "mcq", "answer": "C"}, "A")
    with pytest.raises(ValueError):
        quality.score_task({"scorer": "nope", "expected": 1}, "x")


def test_mcq_prompt_layout():
    task = {"subject": "college_biology", "question": "What is a cell?",
            "choices": ["w", "x", "y", "z"], "scorer": "mcq", "answer": "A"}
    p = quality.mcq_prompt(task)
    assert p.startswith("The following is a multiple choice question about college biology.")
    assert "\nA. w\nB. x\nC. y\nD. z\nAnswer:" in p
    assert p.endswith("Answer:")


def test_committed_mmlu_subset_shape():
    tasks = quality.load_tasks(ROOT / "quality" / "mmlu_subset.jsonl")
    assert len(tasks) == 285
    per_subject = Counter(t["subject"] for t in tasks)
    assert len(per_subject) == 57 and set(per_subject.values()) == {5}
    assert all(len(t["choices"]) == 4 and t["answer"] in "ABCD" for t in tasks)
    assert len({t["id"] for t in tasks}) == 285
    source = json.loads((ROOT / "quality" / "mmlu_subset.source.json").read_text(encoding="utf-8"))
    assert source["n_questions"] == 285 and source["licence"] == "MIT"
    assert len(source["download_sha256"]) == 64


def test_committed_synthetic_tasks_match_the_generator():
    gen = _load_script("make_synthetic_tasks")
    committed = quality.load_tasks(ROOT / "quality" / "synthetic_tasks.jsonl")
    assert committed == json.loads(json.dumps(gen.build(20261024)))
    assert len(committed) == 120
    kinds = Counter(t["kind"] for t in committed)
    assert kinds == {"arith": 40, "extract": 40, "format-exact": 14, "format-json": 13,
                     "format-lines": 13}


def test_synthetic_arithmetic_answers_are_right():
    for t in quality.load_tasks(ROOT / "quality" / "synthetic_tasks.jsonl"):
        if t["kind"] != "arith":
            continue
        a, op, b = t["prompt"].removeprefix("What is ").split("?")[0].split()
        value = {"+": int(a) + int(b), "-": int(a) - int(b), "*": int(a) * int(b)}[op]
        assert t["expected"] == value


def test_synthetic_extraction_answers_appear_in_the_passage():
    for t in quality.load_tasks(ROOT / "quality" / "synthetic_tasks.jsonl"):
        if t["kind"] == "extract":
            passage = t["prompt"].split("\n\n")[0]
            assert str(t["expected"]) in passage


def test_every_scorer_accepts_the_reference_answer_of_its_own_task():
    """A task whose own reference scores as wrong would make a perfect model look wrong."""
    for t in quality.load_tasks(ROOT / "quality" / "synthetic_tasks.jsonl"):
        s = t["scorer"]
        if s == "json":
            reference = json.dumps(t["expected"])
        elif s == "lines":
            want = t["expected"]
            reference = "\n".join(f"{want['prefix']}fruit{i}" for i in range(want["n"]))
        else:
            reference = str(t["expected"])
        assert quality.score_task(t, reference), t["id"]


async def test_run_quality_scores_replies_and_counts_errors(fake):
    tasks = quality.load_tasks(ROOT / "quality" / "mmlu_subset.jsonl")[:10]
    answers = {t["id"]: t["answer"] for t in tasks}
    # the fake answers with the right letter for the first five questions, a wrong one afterwards
    seen = []

    def reply(body):
        seen.append(body)
        return "A" if len(seen) <= 5 else "Z"

    fake.reply_fn = reply
    results = await quality.run_quality(fake.url, tasks, model="bench", concurrency=1)
    assert [r["id"] for r in results] == [t["id"] for t in tasks]
    assert all(r["ok"] for r in results)
    for r in results[:5]:
        assert r["correct"] == (answers[r["id"]] == "A")
    assert not any(r["correct"] for r in results[5:])
    assert seen[0]["temperature"] == 0.0 and seen[0]["stream"] is False
    assert seen[0]["max_tokens"] == quality.MAX_TOKENS["mcq"]


async def test_run_quality_marks_http_failures_as_wrong(fake):
    fake.fail_every = 1
    tasks = quality.load_tasks(ROOT / "quality" / "synthetic_tasks.jsonl")[:4]
    results = await quality.run_quality(fake.url, tasks, model="bench", concurrency=2)
    assert not any(r["ok"] for r in results) and not any(r["correct"] for r in results)
    summary = quality.summarize_quality(results)
    assert summary["all"]["request_errors"] == 4 and summary["all"]["accuracy"] == 0.0


def test_summarize_quality_groups_and_intervals():
    rows = ([{"set": "a", "kind": "x", "ok": True, "correct": True}] * 3
            + [{"set": "a", "kind": "y", "ok": True, "correct": False}]
            + [{"set": "b", "kind": "z", "ok": True, "correct": True}])
    s = quality.summarize_quality(rows)
    assert s["all"]["n"] == 5 and s["all"]["correct"] == 4
    assert s["by_set"]["a"]["accuracy"] == 0.75
    assert s["by_kind"]["a/x"]["accuracy"] == 1.0 and s["by_kind"]["a/y"]["accuracy"] == 0.0
    lo, hi = s["by_set"]["a"]["ci95"]
    assert lo < 0.75 < hi
