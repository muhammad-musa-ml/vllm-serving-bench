# First starts of each server, compile cache empty

The first time each arm started, its Torch compile cache (the `vllm-cache` volume) was empty. These records
come from those starts. The headline runs started every server again with a warm cache, and
`results/server/<arm>/` holds those starts.

| arm | file | KV cache that fit | engine init |
|---|---|---|---|
| fp16 | `fp16-startup.log` (whole log; the first start also applied the model repository's sampling defaults, fixed before the headline runs) | 42,432 tokens | 50.09 s |
| awq | `awq-facts.json` (output of `scripts/extract_server_facts.py`; its log was not kept) | 115,072 tokens | 67.14 s |
| gptq | `gptq-facts.json` (same) | 114,528 tokens | 50.3 s |

With the cache warm the same flags gave a larger KV cache (fp16 54,352 tokens, awq 127,280; see
`results/server/<arm>/facts.json`) and a much shorter engine init (17 s for fp16). Free memory at start
was 4.95 GiB in both cases. So the KV-cache capacity of a server depends on whether the compile workspace
was still allocated when vLLM measured what was left; compare arms only at the same cache state.
