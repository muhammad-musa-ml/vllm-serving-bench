"""Expose a few nvidia-smi readings as Prometheus gauges on :9835/metrics.

Standard library only. nvidia-smi is run on every scrape, so the readings are as old as the scrape.
"""
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

QUERY = "memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw"
NAMES = [
    ("gpu_memory_used_mib", "GPU memory in use, MiB (all processes, including the desktop)"),
    ("gpu_memory_total_mib", "GPU memory installed, MiB"),
    ("gpu_utilization_percent", "GPU utilization, percent"),
    ("gpu_temperature_celsius", "GPU temperature, degrees Celsius"),
    ("gpu_sm_clock_mhz", "Streaming multiprocessor clock, MHz"),
    ("gpu_power_watts", "Board power draw, watts"),
]


def parse_line(line):
    """Parse one `--format=csv,noheader,nounits` line into a list of floats (NaN for [N/A])."""
    out = []
    for part in line.split(","):
        part = part.strip()
        try:
            out.append(float(part))
        except ValueError:
            out.append(float("nan"))
    return out


def render(values):
    lines = []
    for (name, help_text), value in zip(NAMES, values, strict=False):
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    return "\n".join(lines) + "\n"


def read_gpu():
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True)
    first = result.stdout.strip().splitlines()[0]
    return parse_line(first)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = render(read_gpu()).encode()
            status = 200
        except (subprocess.SubprocessError, OSError, IndexError) as exc:
            body = f"nvidia-smi failed: {exc}\n".encode()
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9835
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
