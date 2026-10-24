import importlib.util
from pathlib import Path

from servebench import gpu


def test_parse_full_line():
    s = gpu.parse_gpu_line("2817, 63, 61, 2370, 52.31\n")
    assert s == {"mem_mib": 2817.0, "util_pct": 63.0, "temp_c": 61.0, "sm_mhz": 2370.0,
                 "power_w": 52.31}


def test_parse_line_with_na_field_keeps_the_rest():
    s = gpu.parse_gpu_line("2817, 63, 61, 2370, [N/A]")
    assert s["power_w"] is None
    assert s["mem_mib"] == 2817.0


def test_parse_rejects_short_and_memoryless_lines():
    assert gpu.parse_gpu_line("1, 2, 3") is None
    assert gpu.parse_gpu_line("[N/A], 1, 2, 3, 4") is None
    assert gpu.parse_gpu_line("") is None


def test_summarize_samples():
    samples = [
        {"mem_mib": 100.0, "util_pct": 10.0, "temp_c": 50.0, "sm_mhz": 2000.0, "power_w": 20.0},
        {"mem_mib": 300.0, "util_pct": 90.0, "temp_c": 60.0, "sm_mhz": 1500.0, "power_w": None},
    ]
    s = gpu.summarize_samples(samples)
    assert s["n"] == 2
    assert s["mem_mib"] == {"mean": 200.0, "min": 100.0, "max": 300.0}
    assert s["power_w"] == {"mean": 20.0, "min": 20.0, "max": 20.0}
    assert gpu.summarize_samples([]) == {"n": 0}


def test_sampler_without_nvidia_smi_is_unavailable_and_harmless():
    sampler = gpu.GpuSampler(smi="no-such-binary-xyz")
    assert sampler.available is False
    sampler.start()
    assert sampler.window(0, 1e12) == []
    sampler.stop()


def _load_exporter():
    path = Path(__file__).resolve().parent.parent / "docker" / "gpu-exporter" / "exporter.py"
    spec = importlib.util.spec_from_file_location("exporter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exporter_renders_prometheus_gauges_and_nan():
    exporter = _load_exporter()
    values = exporter.parse_line("2817, 6141, 63, 61, 2370, [N/A]")
    text = exporter.render(values)
    assert "gpu_memory_used_mib 2817.0" in text
    assert "gpu_memory_total_mib 6141.0" in text
    assert "gpu_power_watts nan" in text
    assert text.count("# TYPE") == len(exporter.NAMES)
