# vllm-serving-bench

A small open-weights language model served from the 6 GB GPU in my laptop with
vLLM, plus the tools I used to measure it: an async load harness, a fixed set
of questions with scripted scoring, and a Prometheus and Grafana stack that
watches the server while it runs.

I built it to see how a model server behaves under load on a small card: how
throughput and latency change as more requests arrive, where the server stops
scaling, and what 4-bit weights buy and cost compared with 16-bit weights.

**The load prompts are synthetic**: made-up documents built from a word list
with a seeded random generator. The quality questions are 285 from the public
MMLU test set and 120 generated ones (see [Quality set](#quality-set)). Every
number below was measured on one Windows 11 laptop; none of it says anything
about other GPUs.

## What it does

```
load harness ──HTTP──> vLLM :8000 ──/metrics──┐
                                              ├─> Prometheus ─> Grafana
nvidia-smi ──> gpu-exporter :9835 ────────────┘
```

| piece | what it does | host port |
|---|---|---|
| vLLM (`vllm/vllm-openai:v0.32.0`) | serves one model at a time through the OpenAI-compatible API; one Compose profile per precision arm | 18480 |
| gpu-exporter | publishes GPU memory, utilization, clock, temperature and power from `nvidia-smi` | none |
| Prometheus | scrapes the server and the exporter every 5 s | 18490 |
| Grafana | dashboard "vLLM serving", provisioned from `observability/grafana/` | 18491 |
| `servebench` | the load harness, quality runner and results summary (`src/servebench/`) | |

Three precision arms of one model, all from the Qwen organisation's own
repositories (Apache-2.0, not gated), pinned to a commit hash in
[`arms.json`](arms.json):

| arm | repository | weights |
|---|---|---|
| `fp16` | `Qwen/Qwen2.5-1.5B-Instruct` | 16-bit (the repository's bfloat16 weights, cast to float16) |
| `awq` | `Qwen/Qwen2.5-1.5B-Instruct-AWQ` | 4-bit AWQ |
| `gptq` | `Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4` | 4-bit GPTQ |

I did not quantize anything myself. Why this model and the design of the
measurements: [docs/design.md](docs/design.md).

## Running it

Needs Docker with GPU support (the NVIDIA runtime), an NVIDIA GPU with 6 GB,
and Python 3.11 or newer for the harness. The first start of each arm downloads
the weights from Hugging Face (public repositories, no token).

```
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,report]"      # .venv/bin/pip on Linux and macOS
docker compose --profile fp16 up -d --wait        # or awq, or gptq; one at a time
.venv/Scripts/python -m servebench.cli bench --arm fp16 --out results/try
docker compose down
```

Grafana is at http://127.0.0.1:18491 (anonymous read access), Prometheus at
http://127.0.0.1:18490. Only one arm can run at a time: they share port 18480
and the card does not have the memory for two.

`scripts/run_arm.py <arm> --runs 5` does everything for one arm: starts the
server, waits for its health check, keeps the startup log, runs a warm-up pass,
five timed passes and five quality runs, and checks the dashboard against
Prometheus. `servebench summarize` turns the run files into the tables.

## How the numbers are produced

- **Closed loop.** 1, 2, 4, 8 and 16 workers, each sending its next request as
  soon as the last one finishes; `max(32, 16 x workers)` requests per level.
  Throughput is output tokens (or requests) divided by the time from the first
  send to the last completion.
- **Open loop.** Requests arrive as a Poisson process at a fixed rate whether or
  not the server keeps up, for 40 s per rate, and latency is measured from the
  scheduled arrival, so a slow server cannot hide its queue. After the last
  arrival the queue gets 60 s to drain; whatever is still unfinished is counted.
- **Every request** asks for exactly 128 new tokens (`ignore_eos`), so every
  arm does the same decode work. Prompts are 256 seeded synthetic documents of
  about 100 to 400 words, the same for every run and every arm.
- **Time to first token** is the first content chunk of the stream; end-to-end
  latency is the last chunk; both are percentiles per run, and the tables give
  the median of the 5 runs with the range in brackets. Per-request records are
  kept (`results/runs/<arm>/run-NN.requests.jsonl.gz`), so pooled percentiles
  over all five runs are in the summary too.
- **GPU memory** is sampled from `nvidia-smi` four times a second during every
  level. It is total memory in use on the card, desktop included. vLLM reserves
  its memory when it starts, so this figure barely moves with load; the
  per-arm differences are in how much of the reservation holds weights and how
  much holds the KV cache, from each arm's startup log
  (`results/server/<arm>/facts.json`).
- **Shared machine.** Other projects were using the same laptop. Every timed
  pass takes an exclusive lock file first (`--lock-file`) so two measurements
  do not overlap, and `docker ps` is saved beside each run file. Background
  load that is not locked (the desktop, containers that are idle) is still
  there.
- **Prefix caching is off** (the five runs reuse the same prompts, and with
  caching on the later runs would skip the prompt work), and the server's
  `--generation-config vllm` ignores the sampling defaults in the model
  repository (a repetition penalty of 1.1, top-k 20, top-p 0.8), which the
  first start of the server was applying to every request.

### Quality set

Each arm answers the same 405 questions at temperature 0, five times, and a
script scores the replies (`src/servebench/quality.py`; no model judges
anything):

- `mmlu-subset`: 285 four-choice questions, 5 from each of the 57 MMLU
  subjects, drawn with a fixed seed from the test split of `cais/mmlu` (MIT
  licence, dataset revision `c30699e8356da336a370243923dbaf21066bb9fe`,
  downloaded 2026-10-24). A reply is right if the first A, B, C or D in it is
  the answer. [`quality/mmlu_subset.source.json`](quality/mmlu_subset.source.json)
  has the URL, the hash of the download and the seed.
- `synthetic-tasks`: 120 generated items: 40 integer arithmetic questions, 40
  extractions of one fact from a made-up order note, and 40 format-following
  items (exact text, a JSON object, a number of lines) checked by code. Generated
  by `scripts/make_synthetic_tasks.py` with a fixed seed.

Greedy decoding on a GPU is not bit-for-bit repeatable when batches differ, so
the tables give the range over the five runs and the number of questions whose
result changed between runs.

## Tests

```
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest -q
```

No GPU is needed. The harness is tested against an in-process fake server that
streams server-sent events: percentiles and median/range, the stream parser,
the Poisson schedule, latency measured from the scheduled arrival, failures,
saturation, the lock file, the workload generator, the quality scorers (and a
check that each synthetic task's own answer scores as correct), the results
summary, the command line, the Compose file against `arms.json`, and that every
metric a dashboard panel queries appears in a snapshot taken from the real
server (`tests/fixtures/vllm-metrics-v0.32.0.txt`). `.github/workflows/ci.yml`
runs the same lint and tests on GitHub Actions, and checks the Compose profiles
and the Prometheus configuration. The measurements are not part of CI: they need
the GPU.

## Things that went wrong

- `vllm/vllm-openai:v0.32.0-cu129`, the first image I pulled (11 GB, about half
  an hour on this machine), does not start: `vllm serve` fails at import because
  its `torchcodec` needs `libnvrtc.so.13`, which a CUDA 12.9 image does not
  carry (`docs/notes/cu129-import-error.log`). The plain `v0.32.0` tag is the CUDA 13
  build and works with this driver (reports CUDA 13.1).
- The first start of the server printed that it was overriding vLLM's sampling
  defaults with the model repository's `generation_config.json`. That would
  have put a repetition penalty on every "temperature 0" request, so every
  arm now starts with `--generation-config vllm`.
- A unit test showed that counting open-loop completions only inside the arrival
  window under-counts under load, because requests that arrive late finish after
  the window. Open-loop throughput now uses the closed loop's basis: completed
  work over the time from the first arrival to the last completion.

## Repository layout

```
src/servebench/   load harness, quality runner, results summary, plots
quality/          the two question sets and where they came from
scripts/          arm runner, stack check, environment recorder, set builders
docker/           the GPU exporter image
observability/    Prometheus and Grafana configuration and the dashboard
tests/            unit tests and the metrics snapshot they check the dashboard against
results/          every run (JSON), per-request records, summary, evidence
docs/             design notes
```
