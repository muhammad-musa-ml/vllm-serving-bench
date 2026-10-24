import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check = load("check_stack")


def test_panel_targets_walks_rows_and_skips_targets_without_expr():
    dash = {"panels": [
        {"title": "a", "targets": [{"refId": "A", "expr": "up"}, {"refId": "B"}]},
        {"title": "row", "panels": [{"title": "b", "targets": [{"refId": "A", "expr": "rate(x[1m])"}]}]},
    ]}
    assert check.panel_targets(dash) == [("a", "A", "up"), ("b", "A", "rate(x[1m])")]


def test_expand_macros():
    assert check.expand_macros("rate(x[$__rate_interval])") == "rate(x[30s])"
    assert check.expand_macros("avg_over_time(x[$__interval])") == "avg_over_time(x[15s])"


def test_the_shipped_dashboard_has_unique_panel_ids_and_a_query_per_panel():
    path = ROOT / "observability/grafana/dashboards/vllm-serving.json"
    dash = json.loads(path.read_text(encoding="utf-8"))
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))
    for p in dash["panels"]:
        assert p["targets"], p["title"]
        assert p["datasource"]["uid"] == "prometheus"
        g = p["gridPos"]
        assert 0 <= g["x"] and g["x"] + g["w"] <= 24
    assert dash["uid"] == "vllm-serving"


def test_parse_utc_round_trips():
    assert check.parse_utc("2026-10-24T07:00:00Z") == 1792825200.0
