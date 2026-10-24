import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "vllm-metrics-v0.32.0.txt"
DASHBOARD = ROOT / "observability" / "grafana" / "dashboards" / "vllm-serving.json"
TOKEN = re.compile(r"\b(vllm:[A-Za-z0-9_:]+|gpu_[a-z_]+)\b")


def sample_names(text):
    names = set()
    for line in text.splitlines():
        if line and not line.startswith("#"):
            names.add(re.split(r"[{ ]", line, maxsplit=1)[0])
    return names


def exporter_names():
    path = ROOT / "docker" / "gpu-exporter" / "exporter.py"
    spec = importlib.util.spec_from_file_location("exporter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {name for name, _ in mod.NAMES}


def dashboard_metrics():
    dash = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    found = set()
    for panel in dash["panels"]:
        for target in panel["targets"]:
            found |= set(TOKEN.findall(target["expr"]))
    return found


def test_snapshot_is_a_real_server_scrape():
    text = FIXTURE.read_text(encoding="utf-8")
    assert "# TYPE vllm:num_requests_running gauge" in text
    assert 'model_name="bench"' in text
    assert any(n.startswith("vllm:time_to_first_token_seconds_bucket") for n in sample_names(text))


def test_every_metric_the_dashboard_queries_is_exported():
    exported = sample_names(FIXTURE.read_text(encoding="utf-8")) | exporter_names()
    queried = dashboard_metrics()
    assert queried, "the dashboard queries no metrics"
    missing = sorted(m for m in queried if m not in exported)
    assert not missing, f"queried but not exported: {missing}"


def test_dashboard_covers_the_server_and_the_gpu():
    queried = dashboard_metrics()
    assert any(m.startswith("vllm:") for m in queried)
    assert any(m.startswith("gpu_") for m in queried)
    for needed in ("vllm:time_to_first_token_seconds_bucket", "vllm:e2e_request_latency_seconds_bucket",
                   "vllm:generation_tokens_total", "vllm:kv_cache_usage_perc"):
        assert needed in queried, needed
