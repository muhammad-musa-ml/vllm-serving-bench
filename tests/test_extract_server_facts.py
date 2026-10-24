import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("extract_server_facts",
                                              ROOT / "scripts" / "extract_server_facts.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)

EXCERPT = (ROOT / "tests" / "fixtures" / "fp16-startup-excerpt.log").read_text(encoding="utf-8")


def test_values_printed_by_the_server_are_read_back():
    f = extract.parse(EXCERPT)
    assert f["model_loading_gib"] == 2.98
    assert f["model_loading_s"] == 14.488121
    assert f["kv_cache_gib"] == 1.13
    assert f["kv_cache_tokens"] == 42432
    assert f["max_concurrency_at_max_len"] == 20.72
    assert f["engine_init_s"] == 42.17
    assert f["compilation_s"] == 26.87
    assert f["free_memory_gib_at_startup"] == 4.95
    assert f["total_memory_gib"] == 6.0
    assert f["attention_backend"] == "FLASH_ATTN"
    assert f["vllm_version"] == "0.32.0"


def test_warnings_are_listed_without_timestamps():
    f = extract.parse(EXCERPT)
    assert any("Pinned memory is not available" in w for w in f["warnings"])
    assert any("Casting torch.bfloat16 to torch.float16" in w for w in f["warnings"])
    assert all(not w.split()[1].startswith("10-") for w in f["warnings"] if w.startswith("WARNING"))


def test_per_layer_weight_cast_warnings_are_counted_not_listed():
    line = ("vllm-fp16-1  | (EngineCore pid=1) WARNING 10-24 07:53:38 [utils.py:333] Attempted to load "
            "weight layers.{n}.mlp.down_proj.weight with dtype torch.bfloat16 into parameter with dtype "
            "torch.float16")
    text = "\n".join(line.format(n=n) for n in range(5))
    f = extract.parse(text)
    assert f["weight_dtype_cast_warnings"] == 5 and f["warnings"] == []


def test_missing_values_are_null_not_guessed():
    f = extract.parse("nothing useful here\n")
    for key in ("model_loading_gib", "kv_cache_tokens", "engine_init_s", "attention_backend"):
        assert f[key] is None
    assert f["warnings"] == [] and f["quantization_lines"] == []


def test_quantization_kernel_is_read_from_the_log_line_the_server_prints():
    text = ("vllm-awq-1  | (EngineCore pid=178) INFO 10-24 08:09:26 [auto_awq.py:451] "
            "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod\n"
            "vllm-awq-1  | (APIServer pid=1) INFO 10-24 08:08:45 [api_utils.py:347]   model   "
            "Qwen/Qwen2.5-1.5B-Instruct-AWQ")
    f = extract.parse(text)
    assert f["linear_kernel"] == "MarlinLinearKernel"
    assert f["linear_method"] == "AutoAWQMarlinLinearMethod"
    assert len(f["quantization_lines"]) == 1 and "Kernel for" in f["quantization_lines"][0]


def test_fp16_log_has_no_quantization_kernel():
    f = extract.parse(EXCERPT)
    assert f["linear_kernel"] is None and f["quantization_lines"] == []
