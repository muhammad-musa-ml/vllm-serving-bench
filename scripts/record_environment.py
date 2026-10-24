"""Write results/environment.json: the machine and tool versions the results were produced on.

Records no host name and no user name. Run it once per measurement session:
    python scripts/record_environment.py [--out results/environment.json]
"""
import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ["vllm/vllm-openai:v0.32.0", "prom/prometheus:v3.5.0", "grafana/grafana:12.0.2"]


def run(cmd, timeout=30):
    exe = shutil.which(cmd[0])
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def gpu():
    fields = "name,memory.total,driver_version,pci.bus_id,power.limit,clocks.max.sm,clocks.max.mem"
    line = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    header = run(["nvidia-smi"]) or ""
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    if not line:
        return {"available": False}
    parts = [p.strip() for p in line.splitlines()[0].split(",")]
    keys = ["name", "memory_total_mib", "driver_version", "pci_bus_id", "power_limit_w",
            "max_sm_clock_mhz", "max_mem_clock_mhz"]
    return {"available": True, **dict(zip(keys, parts, strict=False)),
            "cuda_version_reported_by_driver": cuda.group(1) if cuda else None}


def power_state():
    """AC or battery on Windows; None elsewhere."""
    if sys.platform != "win32":
        return None
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    ac = {0: "battery", 1: "AC", 255: "unknown"}.get(status.ACLineStatus, "unknown")
    plan = run(["powercfg", "/getactivescheme"])
    return {"source": ac, "battery_percent": status.BatteryLifePercent,
            "active_power_scheme": plan}


def docker_info():
    version = run(["docker", "version", "--format", "{{.Server.Version}}"])
    compose = run(["docker", "compose", "version", "--short"])
    info = run(["docker", "info", "--format", "{{.NCPU}} {{.MemTotal}} {{.OperatingSystem}}"])
    ncpu = mem = osname = None
    if info:
        a = info.split(" ", 2)
        ncpu, mem, osname = int(a[0]), int(a[1]), a[2]
    images = {}
    for ref in IMAGES:
        digest = run(["docker", "image", "inspect", ref, "--format",
                      "{{index .RepoDigests 0}}|{{.Id}}|{{.Size}}"])
        if digest:
            d, image_id, size = digest.split("|")
            images[ref] = {"repo_digest": d, "image_id": image_id, "size_bytes": int(size)}
    return {"server_version": version, "compose_version": compose, "ncpu": ncpu,
            "mem_total_bytes": mem, "operating_system": osname, "images": images}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/environment.json")
    args = ap.parse_args()
    import aiohttp

    env = {
        "recorded_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "logical_cpus": os.cpu_count(), "power": power_state()},
        "gpu": gpu(),
        "docker": docker_info(),
        "python": {"version": sys.version.split()[0], "aiohttp": aiohttp.__version__},
        "git": {"git_version": run(["git", "--version"])},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
