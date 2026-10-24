# Design

This is the plan I wrote before building the harness. Where a measurement
changed it, the change is noted in the section it affects.

## What I want to find out

I serve a small open-weights chat model from the 6 GB GPU in my laptop and
measure what the server does under load. Three questions:

1. How do output tokens per second, requests per second, time to first token
   and end-to-end latency change as the number of requests in flight goes
   from 1 to 16?
2. What does 4-bit weight quantization (AWQ, GPTQ) change compared with
   fp16: throughput, latency, GPU memory, and the answers on a fixed question
   set that a script can score?
3. At what load does the server saturate, when requests arrive one after the
   other (closed loop) and when they arrive on their own schedule (open
   loop)?

Everything runs on one machine, so none of the numbers say anything about a
data-center GPU. What they do show is the shape of the curves and the cost of
each precision on a small card.

## Stack

| piece | choice | why |
|---|---|---|
| inference server | vLLM `v0.32.0` (`vllm/vllm-openai:v0.32.0`, the CUDA 13 build) in Docker | OpenAI-compatible HTTP API, continuous batching, Prometheus metrics built in, AWQ and GPTQ kernels |
| GPU access | Docker Desktop with the NVIDIA runtime | checked first: `docker run --gpus all` shows the RTX 4050 (`results/environment.json`) |
| metrics | Prometheus scrapes the server's `/metrics` and a small GPU exporter | the dashboard and the load harness measure the same run from two sides |
| dashboard | Grafana, provisioned from files | the dashboard is code, so a fresh `docker compose up` shows it |
| load harness | Python 3, `asyncio` and `aiohttp` | one process can hold 16 streaming requests and timestamp every chunk |
| CI | GitHub Actions, no GPU | runs the harness's unit tests against a fake server and checks the configuration files |

If the GPU had not been usable from the container, the fallback was
llama.cpp's server (its official CUDA image) with GGUF quantizations of the
same model. Not needed: the GPU probe passed.

All services publish host ports 18400-18499 only: 18480 (model server),
18490 (Prometheus), 18491 (Grafana). They bind to 127.0.0.1.

## Model and precision arms

One model family at three precisions, so the arms differ in the weights and
nothing else.

| arm | Hugging Face repo | weights |
|---|---|---|
| `fp16` | `Qwen/Qwen2.5-1.5B-Instruct` | 16-bit |
| `awq` | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | 4-bit AWQ |
| `gptq` | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | 4-bit GPTQ |

All three are the Qwen organisation's own repositories, Apache-2.0, not gated
(checked through the Hugging Face API on 2026-10-24), pinned to the commit
hashes in `arms.json`. The 4-bit repos are the same model quantized by its
authors, so I did not quantize anything myself.

Models I looked at and left out: Qwen2.5-3B-Instruct (its licence is
`qwen-research`, not Apache-2.0) and Llama 3.2 1B (the repo is gated behind
an accepted licence on an account, and I do not use accounts for this).

Server flags are the same for every arm: `--max-model-len 2048`,
`--gpu-memory-utilization 0.80`, `--max-num-seqs 32`,
`--no-enable-prefix-caching`. Prefix caching is off because the five runs
reuse the same prompts; with it on, every run after the first would skip the
prompt work and the runs would stop being repeats of each other.

## The load harness

`servebench` is a small package (`src/servebench/`) with one command line.

Each request is a streamed chat completion. The harness records the time of
the first content chunk (time to first token), the time of the last chunk
(end-to-end), the gaps between chunks, and the token counts the server
reports in its final usage chunk.

**Workload.** Prompts are synthetic: an instruction plus a made-up document of
100 to 400 words built from a fixed word list with a seeded random generator,
so every run and every arm sends the same 256 prompts. Every request asks for
exactly 128 new tokens and tells the server to ignore the end-of-sequence
token, so each request does the same amount of decode work. That makes the
arms comparable on tokens per second. The quality set (below) does not do
this; it stops at the end of the answer.

**Closed loop.** A fixed number of workers (1, 2, 4, 8, 16), each sending its
next request as soon as the last one finishes. Throughput is the tokens (or
requests) completed divided by the wall time from the first send to the last
completion, so the ramp-down at the end counts against it. The number of
requests per level is `max(32, 16 x workers)`.

**Open loop.** Requests arrive as a Poisson process at a fixed rate,
whether or not the server has kept up. Latency is measured from the
scheduled arrival time, not from when the client got round to sending, so a
slow server cannot hide its queue (coordinated omission). Rates are fixed and
the same for every arm; they were picked from a first exploratory pass
(`results/exploratory/`) so that the list brackets every arm's saturation.

**GPU memory.** `nvidia-smi` is sampled five times a second while a level
runs. The figure is total memory in use on the card, which includes what the
desktop uses; the idle figure before the server starts is recorded next to it.

**Repeats.** Each configuration (arm x level) is measured in 5 passes. One
warm-up pass runs first and is kept in the results folder but not counted.
Tables report the median and the range of the 5 per-pass values. The
per-request records are kept, so pooled percentiles over all five passes can
be computed too.

**Measurement lock.** On a shared machine a neighbour's build can steal CPU
from the client or Docker. The harness can take an exclusive lock file
(`--lock-file`) for the duration of a pass, waiting if another run holds it,
and writes the output of `docker ps` beside every results file so the
conditions are on record.

## Quality set

Throughput numbers mean little if the quantized model has stopped answering
correctly, so each arm also answers a fixed set of questions at temperature 0,
and a script scores the answers:

- `mmlu-subset`: 285 four-choice questions, 5 from each of the 57 MMLU
  subjects (test split, MIT licence, `cais/mmlu` on Hugging Face), drawn with
  a fixed seed. A reply is correct if its first A/B/C/D letter is the right
  one.
- `synthetic-tasks`: 120 generated items (40 arithmetic with an exact integer
  answer, 40 extraction with an exact string answer, 40 format-following items
  checked by a script). Synthetic, seeded.

Greedy decoding on a GPU is not bit-for-bit repeatable when the batch
changes, so the set runs 5 times per arm like everything else and I report
the range too.

## Metrics pipeline

```
load harness ──HTTP──> vLLM :8000 ──/metrics──┐
                                              ├─> Prometheus ─> Grafana
nvidia-smi ──> gpu-exporter :9835 ────────────┘
```

Prometheus scrapes every 5 s. The Grafana dashboard `vllm-serving` shows
requests running and waiting, KV-cache use, generation and prompt tokens per
second, time to first token, end-to-end latency and inter-token latency
quantiles, and GPU memory and utilization. A check script evaluates every
panel query against Prometheus while a run is going, so a panel that queries
a metric the server does not export shows up as an error.

## What the unit tests cover

No GPU is needed: the harness is tested against an in-process fake server that
streams server-sent events. Tests cover the percentile and median/range
code, the stream parser, the Poisson schedule (deterministic for a seed),
coordinated-omission accounting, the lock file, the workload generator, the
quality scorers, and the dashboard and Prometheus files (every metric a panel
queries must appear in a snapshot taken from the real server).

## Limits I expect

- One laptop GPU that also drives the display; thermal state and power plan
  move the numbers. Clock, power and temperature are sampled with the memory.
- Synthetic prompts of one length range and a fixed output length of 128
  tokens. Real traffic has other shapes.
- One model family. No claim about other models or other GPUs.
