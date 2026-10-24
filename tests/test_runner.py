import asyncio

import pytest

from servebench import runner
from servebench.workload import build_prompts

PROMPTS = build_prompts(16, seed=3)


def test_requests_for_level():
    assert [runner.requests_for_level(c) for c in (1, 2, 4, 8, 16)] == [32, 32, 64, 128, 256]


def test_poisson_arrivals_deterministic_sorted_and_inside_the_window():
    a = runner.poisson_arrivals(5.0, 20.0, seed=11)
    b = runner.poisson_arrivals(5.0, 20.0, seed=11)
    c = runner.poisson_arrivals(5.0, 20.0, seed=12)
    assert a == b and a != c
    assert a == sorted(a)
    assert all(0 <= t < 20.0 for t in a)
    # about rate * duration arrivals; wide bound, this is only a sanity check on the generator
    assert 60 <= len(a) <= 140


def test_poisson_arrivals_mean_gap_is_one_over_rate():
    a = runner.poisson_arrivals(50.0, 200.0, seed=1)
    gaps = [y - x for x, y in zip(a, a[1:], strict=False)]
    assert sum(gaps) / len(gaps) == pytest.approx(1 / 50.0, rel=0.05)


def test_poisson_arrivals_rejects_nonpositive_arguments():
    with pytest.raises(ValueError):
        runner.poisson_arrivals(0, 10, 1)
    with pytest.raises(ValueError):
        runner.poisson_arrivals(1, 0, 1)


def test_parse_gauge():
    text = ('# TYPE vllm:num_requests_running gauge\n'
            'vllm:num_requests_running{model_name="a"} 3.0\n'
            'vllm:num_requests_running{model_name="b"} 1.0\n'
            'vllm:num_requests_waiting{model_name="a"} 0.0\n')
    assert runner.parse_gauge(text, "vllm:num_requests_running") == 4.0
    assert runner.parse_gauge(text, "vllm:num_requests_waiting") == 0.0
    assert runner.parse_gauge(text, "vllm:nothing") is None


async def test_closed_loop_counts_and_throughput(fake):
    level, records = await runner.measure_closed(
        fake.url, PROMPTS, concurrency=4, n_requests=24, max_tokens=16, model="bench")
    assert level["mode"] == "closed" and level["concurrency"] == 4
    assert level["n_requests"] == 24 == len(records)
    assert level["n_ok"] == 24 and level["n_failed"] == 0 and level["short_outputs"] == 0
    assert fake.calls == 24
    # throughput is output tokens over the first-send-to-last-completion time
    ok = [r for r in records if r.ok]
    wall = max(r.t_end for r in records) - min(r.t_send for r in records)
    assert level["wall_s"] == pytest.approx(wall)
    assert level["out_tok_s"] == pytest.approx(sum(r.completion_tokens for r in ok) / wall)
    assert level["req_s"] == pytest.approx(24 / wall)
    assert level["ttft_s"]["n"] == 24 and level["e2e_s"]["p99"] >= level["e2e_s"]["p50"]
    assert level["gpu"] is None
    assert [r.idx for r in records] == list(range(24))


async def test_closed_loop_never_exceeds_the_worker_count(fake):
    fake.token_delay = 0.003
    await runner.measure_closed(fake.url, PROMPTS, concurrency=3, n_requests=15, max_tokens=8,
                                model="bench")
    assert fake.peak_running <= 3


async def test_closed_loop_reports_failures_and_short_outputs(fake):
    fake.fail_every = 5
    fake.stop_early_every = 4
    level, _ = await runner.measure_closed(fake.url, PROMPTS, concurrency=2, n_requests=20,
                                           max_tokens=16, model="bench")
    assert level["n_failed"] == 4  # calls 5, 10, 15, 20
    assert level["errors"]
    assert level["short_outputs"] > 0
    assert level["n_ok"] + level["n_failed"] == 20


async def test_saturation_shows_up_as_more_waiting_not_more_throughput(fake):
    """With 2 server slots, going from 2 to 8 workers must not raise throughput much."""
    fake.token_delay = 0.004
    fake._sem = asyncio.Semaphore(2)
    low, _ = await runner.measure_closed(fake.url, PROMPTS, concurrency=2, n_requests=24,
                                         max_tokens=16, model="bench")
    high, _ = await runner.measure_closed(fake.url, PROMPTS, concurrency=8, n_requests=24,
                                          max_tokens=16, model="bench")
    assert high["out_tok_s"] < low["out_tok_s"] * 1.4
    assert high["ttft_s"]["p50"] > low["ttft_s"]["p50"]


async def test_open_loop_light_load_completes_everything(fake):
    level, records = await runner.measure_open(
        fake.url, PROMPTS, rate=20.0, duration_s=1.5, seed=5, max_tokens=8, model="bench",
        drain_s=10)
    assert level["mode"] == "open"
    assert level["n_arrivals"] == len(records) == fake.calls
    assert level["n_unfinished"] == 0 and level["n_dropped"] == 0
    assert level["completion_ratio"] == pytest.approx(1.0)
    assert level["realized_rate"] == pytest.approx(level["n_arrivals"] / 1.5)
    ok = [r for r in records if r.ok]
    span = max(r.t_end for r in ok) - min(r.t_sched for r in ok)
    assert level["span_s"] == pytest.approx(span)
    assert level["req_s_overall"] == pytest.approx(len(ok) / span)
    # every request's latency is measured from its scheduled arrival
    assert all(r.t_send >= r.t_sched - 1e-6 for r in records)


async def test_open_loop_overload_leaves_unfinished_and_a_long_tail(fake):
    fake.token_delay = 0.01
    fake._sem = asyncio.Semaphore(1)  # capacity of about 1 / (8 * 0.01) req/s
    level, _ = await runner.measure_open(
        fake.url, PROMPTS, rate=30.0, duration_s=1.0, seed=6, max_tokens=8, model="bench",
        drain_s=0.2)
    assert level["n_unfinished"] > 0
    assert level["completion_ratio"] < 0.6
    assert level["req_s_overall"] < level["realized_rate"] * 0.6


async def test_open_loop_drops_when_too_many_in_flight(fake):
    fake.token_delay = 0.02
    fake._sem = asyncio.Semaphore(1)
    level, _ = await runner.measure_open(
        fake.url, PROMPTS, rate=40.0, duration_s=1.0, seed=7, max_tokens=8, model="bench",
        max_in_flight=3, drain_s=0.2)
    assert level["n_dropped"] > 0


async def test_wait_idle_true_when_server_is_idle_false_when_busy(fake):
    assert await runner.wait_idle(fake.url, timeout_s=2) is True
    fake.running = 1
    assert await runner.wait_idle(fake.url, timeout_s=0.6, poll_s=0.1) is False


async def test_run_pass_covers_every_level_and_rate(fake):
    fake.token_delay = 0.001
    lines = []
    summaries, raw = await runner.run_pass(
        fake.url, PROMPTS, model="bench", levels=[1, 2], open_rates=[10.0], max_tokens=8,
        open_duration_s=1.0, drain_s=5, seed=1, sampler=None, cooldown_s=0.0, log=lines.append)
    assert [(s["mode"], s.get("concurrency") or s.get("offered_rate")) for s in summaries] == [
        ("closed", 1), ("closed", 2), ("open", 10.0)]
    assert all(s["idle_before_next"] for s in summaries)
    assert len(raw) == 3 and len(lines) == 3
    assert raw[0][0] == {"mode": "closed", "concurrency": 1}
