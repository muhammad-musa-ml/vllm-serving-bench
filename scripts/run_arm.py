"""Run one precision arm end to end and leave its evidence under results/.

    python scripts/run_arm.py fp16 [--runs 5] [--lock-file PATH --lock-owner NAME]

Steps: stop any other arm's server, start this arm with its Compose profile (plus Prometheus,
Grafana and the GPU exporter), wait for the server's health check, keep the startup log and a
/metrics sample, record the GPU memory with the model loaded and idle, then `--runs` timed passes
(one batch: the measurement lock, if one is given, is taken once and held across them, and each
pass starts with a short uncounted warm-up) and `--runs` quality runs (not timed, no lock), and
finally ask check_stack.py to evaluate the dashboard against Prometheus.

`--first-run N` and `--reuse-server` continue an arm whose server is already up and whose first
N-1 runs are on disk.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from servebench.lock import MeasurementLock

ROOT = Path(__file__).resolve().parent.parent
ARMS = json.loads((ROOT / "arms.json").read_text(encoding="utf-8"))
URL = "http://127.0.0.1:18480"


def utc_now():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def compose(*args, check=True, capture=False):
    cmd = ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), *args]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, encoding="utf-8",
                          errors="replace", cwd=ROOT)


def servebench(*args):
    cmd = [sys.executable, "-m", "servebench.cli", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def stop_all_arms():
    services = [a["service"] for a in ARMS["arms"].values()]
    profiles = [x for a in ARMS["arms"].values() for x in ("--profile", a["profile"])]
    compose(*profiles, "rm", "-sf", *services, check=False)


def fetch(path, timeout=15):
    with urllib.request.urlopen(URL + path, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def gpu_reading():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return out.stdout.strip()


def start_server(args, arm, server_dir):
    stop_all_arms()
    (server_dir / "gpu-before-server.txt").write_text(
        f"# memory.used MiB, memory.total MiB, utilization.gpu % ({utc_now()}, no model server running)\n"
        f"{gpu_reading()}\n", encoding="utf-8", newline="\n")
    t_up = time.time()
    compose("--profile", arm["profile"], "up", "-d", "--wait", "--wait-timeout", "1500")
    startup_s = time.time() - t_up
    log = compose("logs", "--no-color", arm["service"], capture=True)
    (server_dir / "startup.log").write_text(log.stdout + log.stderr, encoding="utf-8", newline="\n")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "extract_server_facts.py"), args.arm,
                    "--results", args.results], check=False, cwd=ROOT, capture_output=True)
    (server_dir / "model.json").write_text(json.dumps({
        "arm": args.arm, "model": arm["model"], "revision": arm["revision"], "dtype": arm["dtype"],
        "weights": arm["weights"], "license": arm["license"], "image": ARMS["image"],
        "server_flags": ARMS["server_flags"], "start_to_healthy_s": round(startup_s, 1),
        "started_utc": utc_now(),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    (server_dir / "gpu-idle-model-loaded.txt").write_text(
        f"# memory.used MiB, memory.total MiB, utilization.gpu % ({utc_now()}, model loaded, idle)\n"
        f"{gpu_reading()}\n", encoding="utf-8", newline="\n")
    print(f"{args.arm}: healthy after {startup_s:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list(ARMS["arms"]))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--first-run", type=int, default=1, help="continue from this run number")
    ap.add_argument("--reuse-server", action="store_true", help="the arm's server is already up")
    ap.add_argument("--pause-between-runs", type=float, default=0.0,
                    help="seconds to wait between two timed passes (the lock is kept)")
    ap.add_argument("--results", default="results")
    ap.add_argument("--lock-file", default=None)
    ap.add_argument("--lock-owner", default="servebench")
    ap.add_argument("--open-rates", default="2,4,6,8")
    ap.add_argument("--open-duration", type=float, default=40.0)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--keep-up", action="store_true", help="leave the stack running afterwards")
    args = ap.parse_args()

    arm = ARMS["arms"][args.arm]
    results = (ROOT / args.results).resolve()
    server_dir = results / "server" / args.arm
    server_dir.mkdir(parents=True, exist_ok=True)
    # The whole arm is one batch: the lock is taken first and held through the server start (a CPU-heavy
    # compile), the timed passes, the quality runs and the shutdown; the other users of the machine hold it
    # the same way, a batch at a time. Every timed pass starts with a short uncounted warm-up.
    lock = None
    batch_args = []
    if args.lock_file:
        lock = MeasurementLock(args.lock_file, args.lock_owner, poll_s=0.05)
        print(f"{args.arm}: waiting for the measurement lock ...", flush=True)
        lock.acquire()
        print(f"{args.arm}: lock taken after {lock.waited_s:.0f} s at {utc_now()}", flush=True)
        batch_args = ["--batch-lock-owner", args.lock_owner]
    try:
        if args.reuse_server:
            fetch("/health")          # raises if the server is not answering
            print(f"{args.arm}: reusing the running server", flush=True)
        else:
            start_server(args, arm, server_dir)
        for n in range(args.first_run, args.runs + 1):
            if n > args.first_run:
                time.sleep(args.pause_between_runs)
            servebench("bench", "--url", URL, "--arm", args.arm, "--run", str(n),
                       "--out", str(results / "runs" / args.arm), "--open-rates", args.open_rates,
                       "--open-duration", str(args.open_duration), "--warmup-first", *batch_args)
        (server_dir / "metrics-sample.txt").write_text(fetch("/metrics"), encoding="utf-8", newline="\n")
        if not args.skip_quality:
            for n in range(1, args.runs + 1):
                servebench("quality", "--url", URL, "--arm", args.arm, "--run", str(n),
                           "--out", str(results / "quality" / args.arm))
        check = ROOT / "scripts" / "check_stack.py"
        if check.exists():
            subprocess.run([sys.executable, str(check), "--arm", args.arm,
                            "--out", str(results / "evidence")], check=False, cwd=ROOT)
        if not args.keep_up:
            stop_all_arms()
    finally:
        if lock:
            lock.release()
            print(f"{args.arm}: lock released at {utc_now()}", flush=True)
    print(f"{args.arm}: done {utc_now()}", flush=True)


if __name__ == "__main__":
    main()
