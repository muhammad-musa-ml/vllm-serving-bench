"""An in-process stand-in for the model server: streams chat completions as server-sent events.

It can be given a capacity (how many requests run at once, the rest wait), a per-token delay and a
failure pattern, so the harness can be tested for saturation and error handling without a GPU.
"""
from __future__ import annotations

import asyncio
import json

from aiohttp import web
from aiohttp.test_utils import TestServer


def sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


class FakeModelServer:
    def __init__(self, *, token_delay: float = 0.002, first_token_delay: float = 0.01,
                 max_concurrent: int = 4, fail_every: int = 0, prompt_tokens: int = 40,
                 stop_early_every: int = 0):
        self.token_delay = token_delay
        self.first_token_delay = first_token_delay
        self.fail_every = fail_every
        self.stop_early_every = stop_early_every
        self.prompt_tokens = prompt_tokens
        self.reply_fn = lambda body: "B"
        self.running = 0
        self.waiting = 0
        self.calls = 0
        self.peak_running = 0
        self.bodies: list[dict] = []
        self._sem = asyncio.Semaphore(max_concurrent)
        self.app = web.Application()
        self.app.router.add_post("/v1/chat/completions", self.chat)
        self.app.router.add_get("/metrics", self.metrics)
        self._server: TestServer | None = None

    async def chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.bodies.append(body)
        self.calls += 1
        call_no = self.calls
        if self.fail_every and call_no % self.fail_every == 0:
            return web.Response(status=500, text="injected failure")
        if body.get("stream") is False:
            reply = self.reply_fn(body)
            message = {"role": "assistant", "content": reply}
            return web.json_response({"choices": [{"message": message}]})
        n = int(body.get("max_tokens", 16))
        if self.stop_early_every and call_no % self.stop_early_every == 0:
            n = max(1, n // 2)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        self.waiting += 1
        async with self._sem:
            self.waiting -= 1
            self.running += 1
            self.peak_running = max(self.peak_running, self.running)
            try:
                await asyncio.sleep(self.first_token_delay)
                role_chunk = {"choices": [{"delta": {"role": "assistant", "content": ""}}]}
                await resp.write(sse(role_chunk))
                for _ in range(n):
                    await asyncio.sleep(self.token_delay)
                    await resp.write(sse({"choices": [{"delta": {"content": "tok "}}]}))
                await resp.write(sse({"choices": [], "usage": {
                    "prompt_tokens": self.prompt_tokens, "completion_tokens": n,
                    "total_tokens": self.prompt_tokens + n}}))
                await resp.write(b"data: [DONE]\n\n")
            finally:
                self.running -= 1
        return resp

    async def metrics(self, request: web.Request) -> web.Response:
        text = (
            "# HELP vllm:num_requests_running Requests running\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f'vllm:num_requests_running{{model_name="bench"}} {float(self.running)}\n'
            "# HELP vllm:num_requests_waiting Requests waiting\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            f'vllm:num_requests_waiting{{model_name="bench"}} {float(self.waiting)}\n'
        )
        return web.Response(text=text, content_type="text/plain")

    async def start(self) -> str:
        self._server = TestServer(self.app)
        await self._server.start_server()
        return str(self._server.make_url("")).rstrip("/")

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.close()
