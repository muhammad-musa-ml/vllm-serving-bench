import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
ARMS = json.loads((ROOT / "arms.json").read_text(encoding="utf-8"))
PROM = yaml.safe_load((ROOT / "observability/prometheus/prometheus.yml").read_text(encoding="utf-8"))


def flag_value(command, flag):
    return command[command.index(flag) + 1]


def test_each_arm_has_a_service_whose_command_matches_arms_json():
    for name, arm in ARMS["arms"].items():
        svc = COMPOSE["services"][arm["service"]]
        assert svc["profiles"] == [arm["profile"]] == [name]
        cmd = [str(x) for x in svc["command"]]
        assert flag_value(cmd, "--model") == arm["model"]
        assert flag_value(cmd, "--revision") == arm["revision"]
        assert flag_value(cmd, "--dtype") == arm["dtype"]
        assert flag_value(cmd, "--served-model-name") == ARMS["served_model_name"]
        assert svc["image"] == ARMS["image"]


def test_server_flags_are_identical_across_arms():
    shared = ARMS["server_flags"]
    for arm in ARMS["arms"].values():
        cmd = [str(x) for x in COMPOSE["services"][arm["service"]]["command"]]
        for flag, value in shared.items():
            assert f"--{flag}" in cmd, flag
            if value is not True:
                assert flag_value(cmd, f"--{flag}") == value


def test_arms_differ_only_in_model_and_revision():
    commands = {}
    for name, arm in ARMS["arms"].items():
        cmd = [str(x) for x in COMPOSE["services"][arm["service"]]["command"]]
        for flag in ("--model", "--revision"):
            i = cmd.index(flag)
            cmd[i + 1] = "<arm>"
        commands[name] = cmd
    assert len({tuple(c) for c in commands.values()}) == 1


def test_models_are_pinned_and_apache_licensed():
    for arm in ARMS["arms"].values():
        assert len(arm["revision"]) == 40 and int(arm["revision"], 16) >= 0
        assert arm["license"] == "apache-2.0"
        assert arm["model"].startswith("Qwen/")


def test_published_ports_stay_in_the_assigned_range_on_localhost():
    published = []
    for name, svc in COMPOSE["services"].items():
        for p in svc.get("ports", []):
            host_ip, host_port, _container = str(p).split(":")
            assert host_ip == "127.0.0.1", f"{name} publishes on {host_ip}"
            published.append(int(host_port))
    assert published and all(18400 <= p <= 18499 for p in published)
    # the arms share one host port because only one runs at a time
    assert sorted(set(published)) == [18480, 18490, 18491]


def test_gpu_exporter_publishes_nothing_and_listens_where_prometheus_looks():
    exporter = COMPOSE["services"]["gpu-exporter"]
    assert "ports" not in exporter
    targets = {job["job_name"]: job["static_configs"][0]["targets"] for job in PROM["scrape_configs"]}
    assert targets["gpu"] == ["gpu-exporter:9835"]
    dockerfile = (ROOT / "docker/gpu-exporter/Dockerfile").read_text(encoding="utf-8")
    assert "9835" in dockerfile


def test_prometheus_scrapes_the_alias_every_arm_shares():
    targets = {job["job_name"]: job["static_configs"][0]["targets"] for job in PROM["scrape_configs"]}
    assert targets["vllm"] == ["vllm:8000"]
    for arm in ARMS["arms"].values():
        svc = COMPOSE["services"][arm["service"]]
        assert "vllm" in svc["networks"]["default"]["aliases"]
        assert flag_value([str(x) for x in svc["command"]], "--port") == "8000"
    assert PROM["global"]["scrape_interval"] == "5s"


def test_grafana_dashboard_provisioning_paths_exist():
    grafana = COMPOSE["services"]["grafana"]
    for vol in grafana["volumes"]:
        host = vol.split(":")[0]
        assert (ROOT / host).exists(), host
    default_home = grafana["environment"]["GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH"]
    assert (ROOT / "observability/grafana/dashboards" / Path(default_home).name).exists()


def test_every_service_that_needs_the_gpu_reserves_it():
    for name in ("vllm-fp16", "vllm-awq", "vllm-gptq", "gpu-exporter"):
        dev = COMPOSE["services"][name]["deploy"]["resources"]["reservations"]["devices"][0]
        assert dev["driver"] == "nvidia"
