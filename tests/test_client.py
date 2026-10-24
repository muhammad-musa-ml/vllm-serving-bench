import time

import aiohttp
import pytest

from servebench import client
from servebench.workload import chat_body


def test_parse_sse_line_variants():
    assert client.parse_sse_line(b"") is None
    assert client.parse_sse_line(b": keep-alive\n") is None
    assert client.parse_sse_line(b"event: ping\n") is None
    assert client.parse_sse_line(b"data: [DONE]\n") == client.DONE
    assert client.parse_sse_line('data: {"a": 1}\n') == {"a": 1}
    assert client.parse_sse_line(b"data: {broken\n") is None


def test_chunk_text_handles_role_chunk_usage_chunk_and_content():
    assert client.chunk_text({"choices": [{"delta": {"role": "assistant", "content": ""}}]}) == ""
    assert client.chunk_text({"choices": []}) == ""
    assert client.chunk_text({"usage": {}}) == ""
    assert client.chunk_text({"choices": [{"delta": {"content": "hi"}}]}) == "hi"
    assert client.chunk_text({"choices": [{"delta": {"content": None}}]}) == ""


async def _one(fake, *, max_tokens=8, t_sched=None, timeout_s=10.0):
    body = chat_body("hello", model="bench", max_tokens=max_tokens, fixed_length=True)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        return await client.stream_request(
            session, fake.url + "/v1/chat/completions", body, idx=3, prompt_id=7, t0=t0,
            t_sched=t_sched, timeout_s=timeout_s)


async def test_successful_request_has_consistent_timings(fake):
    rec = await _one(fake, max_tokens=8)
    assert rec.ok and rec.error is None and rec.status == 200
    assert rec.idx == 3 and rec.prompt_id == 7
    assert rec.completion_tokens == 8 and rec.prompt_tokens == fake.prompt_tokens
    assert rec.n_chunks == 8
    assert 0 < rec.ttft_s < rec.e2e_s
    assert rec.t_send <= rec.t_end
    # time per output token excludes the first token and is about the fake per-token delay
    assert rec.tpot_s is not None and 0.001 < rec.tpot_s < 0.05
    assert rec.itl_max_s >= rec.tpot_s * 0.5


async def test_request_body_reaches_the_server(fake):
    await _one(fake, max_tokens=5)
    body = fake.bodies[0]
    assert body["ignore_eos"] is True and body["max_tokens"] == 5
    assert body["stream_options"] == {"include_usage": True}


async def test_http_error_is_recorded_not_raised(fake):
    fake.fail_every = 1
    rec = await _one(fake)
    assert not rec.ok
    assert rec.status == 500
    assert "injected failure" in rec.error
    assert rec.ttft_s is None and rec.e2e_s is None


async def test_connection_refused_is_recorded_not_raised(unused_tcp_port):
    body = chat_body("x", model="bench", max_tokens=4, fixed_length=True)
    async with aiohttp.ClientSession() as session:
        rec = await client.stream_request(
            session, f"http://127.0.0.1:{unused_tcp_port}/v1/chat/completions", body, idx=0,
            prompt_id=0, t0=time.perf_counter(), timeout_s=2)
    assert not rec.ok and rec.error and rec.status is None


async def test_timeout_is_recorded(fake):
    fake.token_delay = 0.2
    rec = await _one(fake, max_tokens=50, timeout_s=0.3)
    assert not rec.ok
    assert "Timeout" in rec.error or "timeout" in rec.error.lower()


async def test_scheduled_time_in_the_past_is_charged_to_latency(fake):
    """Open loop: latency runs from the scheduled arrival, not from when the client sent."""
    body = chat_body("hello", model="bench", max_tokens=4, fixed_length=True)
    t0 = time.perf_counter() - 1.0  # the run started a second ago
    async with aiohttp.ClientSession() as session:
        rec = await client.stream_request(
            session, fake.url + "/v1/chat/completions", body, idx=0, prompt_id=0, t0=t0,
            t_sched=0.0, timeout_s=10)
    assert rec.ok
    assert rec.t_send - rec.t_sched >= 0.99          # the client was a second late
    assert rec.ttft_s >= 1.0 and rec.e2e_s > rec.ttft_s  # and the lateness is in the latency


async def test_stream_without_done_is_not_ok():
    # a server that closes the stream early leaves no [DONE] and no usage chunk
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def broken(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", broken)
    server = TestServer(app)
    await server.start_server()
    try:
        body = chat_body("x", model="bench", max_tokens=4, fixed_length=True)
        async with aiohttp.ClientSession() as session:
            rec = await client.stream_request(
                session, str(server.make_url("/v1/chat/completions")), body, idx=0, prompt_id=0,
                t0=time.perf_counter(), timeout_s=5)
        assert not rec.ok
        assert "without" in rec.error
    finally:
        await server.close()


@pytest.fixture
def unused_tcp_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
