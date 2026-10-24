"""PNG charts of the summary: throughput, latency and the open-loop sweep, one line per arm."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def plain_log_axis(axis):
    """Log scale with ticks written as plain numbers (1, 2, 5, 10 ...) instead of powers of ten."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    axis.set_major_formatter(fmt)
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "235" else ""))


COLORS = {"fp16": "#1f4e79", "awq": "#c0504d", "gptq": "#4f8f3a"}
MARKERS = {"fp16": "o", "awq": "s", "gptq": "^"}


def _series(arm_data: dict, level_key: str, metric: str, scale: float = 1.0):
    xs, med, lo, hi = [], [], [], []
    for x, m in arm_data[level_key].items():
        v = m.get(metric)
        if not v or v.get("median") is None:
            continue
        xs.append(float(x))
        med.append(v["median"] * scale)
        lo.append(v["min"] * scale)
        hi.append(v["max"] * scale)
    return xs, med, lo, hi


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25)


def _errline(ax, arm, xs, med, lo, hi, label=None, style="-"):
    yerr = [[m - low for m, low in zip(med, lo, strict=True)],
            [h - m for m, h in zip(med, hi, strict=True)]]
    ax.errorbar(xs, med, yerr=yerr, color=COLORS.get(arm), marker=MARKERS.get(arm, "o"),
                linestyle=style, capsize=3, label=label or arm)


def make_plots(summary: dict, outdir: Path) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = {a: d for a, d in summary["arms"].items() if "closed" in d}
    paths = []

    fig, ax = plt.subplots(figsize=(6.2, 4))
    for arm, d in arms.items():
        xs, med, lo, hi = _series(d, "closed", "out_tok_s")
        _errline(ax, arm, xs, med, lo, hi)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    _style(ax, "concurrent requests (closed loop)", "output tokens / s",
           "Throughput, median and range of 5 runs")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = outdir / "throughput-vs-concurrency.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4))
    for arm, d in arms.items():
        xs, med, lo, hi = _series(d, "closed", "e2e_p50_s")
        _errline(axes[0], arm, xs, med, lo, hi, label=f"{arm} p50")
        xs, med, lo, hi = _series(d, "closed", "e2e_p95_s")
        _errline(axes[0], arm, xs, med, lo, hi, label=f"{arm} p95", style="--")
        xs, med, lo, hi = _series(d, "closed", "ttft_p50_s", 1000)
        _errline(axes[1], arm, xs, med, lo, hi, label=f"{arm} p50")
        xs, med, lo, hi = _series(d, "closed", "ttft_p95_s", 1000)
        _errline(axes[1], arm, xs, med, lo, hi, label=f"{arm} p95", style="--")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    _style(axes[0], "concurrent requests (closed loop)", "end-to-end latency, s",
           "End-to-end latency for 128 output tokens")
    _style(axes[1], "concurrent requests (closed loop)", "time to first token, ms",
           "Time to first token")
    axes[1].set_yscale("log")
    plain_log_axis(axes[1].yaxis)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    p = outdir / "latency-vs-concurrency.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    open_arms = {a: d for a, d in arms.items() if d.get("open")}
    if open_arms:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4))
        for arm, d in open_arms.items():
            xs, med, lo, hi = _series(d, "open", "e2e_p95_s")
            _errline(axes[0], arm, xs, med, lo, hi)
            xs, med, lo, hi = _series(d, "open", "req_s_overall")
            _errline(axes[1], arm, xs, med, lo, hi)
        any_arm = next(iter(open_arms.values()))
        offered = sorted(float(r) for r in any_arm["open"])
        axes[1].plot(offered, offered, color="#888888", linestyle=":", label="offered")
        _style(axes[0], "offered load, requests / s (Poisson arrivals)", "p95 end-to-end latency, s",
               "Open loop: latency from the scheduled arrival")
        axes[0].set_yscale("log")
        plain_log_axis(axes[0].yaxis)
        _style(axes[1], "offered load, requests / s (Poisson arrivals)", "completed requests / s",
               "Open loop: completed against offered")
        axes[0].legend(frameon=False)
        axes[1].legend(frameon=False)
        fig.tight_layout()
        p = outdir / "open-loop.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths
