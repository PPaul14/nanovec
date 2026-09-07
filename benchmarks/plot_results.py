"""
Render the committed scaling results as PNG charts.

Run from the project root:
    venv/Scripts/python.exe -m benchmarks.plot_results
    venv/Scripts/python.exe -m benchmarks.plot_results --csv benchmarks/results/scaling_other.csv

Requires matplotlib, which lives in requirements-dev.txt rather than
requirements.txt -- the Docker image has no reason to carry a plotting library
and its font/image dependencies. Install with:
    venv/Scripts/python.exe -m pip install -r requirements-dev.txt

Every number is read from the CSV. Nothing is hardcoded, so re-running the
benchmark and re-running this script keeps the charts and the README tables in
agreement. The tables in the README remain the precise values; these charts are
the shape.

Charts produced (into the same directory as the CSV):

    crossover.png    p50 query latency vs N. The headline: brute force wins at
                     small N and loses at scale.
    recall_vs_n.png  Recall@10 vs N. HNSW holds ~0.93-0.97; IVF+PQ declines.
    memory.png       Memory vs N. IVF+PQ's compression advantage widening.

Design notes
------------
Colour encodes the INDEX and nothing else -- flat, hnsw, ivfpq keep the same
hue in all three charts, so a reader who learns the mapping once can read all
of them. Where a chart shows two parameter settings of the same index, the
parameter is carried by line style, not by a fourth and fifth hue. That keeps
the palette at three slots, which is the number that validates for
colourblind separation across all pairs.

Every line is also directly labelled at its right end. That is a requirement,
not decoration: the aqua slot sits below 3:1 contrast against a white surface,
so identity has to be recoverable without relying on colour.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# Non-interactive backend: this script writes files and must not try to open a
# window (it would fail outright in CI or over SSH).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DEFAULT_CSV = Path("benchmarks") / "results" / "scaling_scale100k_clean.csv"

# Categorical slots 1-3 of the validated default palette. Assigned to entities,
# never to rank, and identical across all three charts.
DISPLAY = {"flat": "Flat", "hnsw": "HNSW", "ivfpq": "IVF+PQ"}

COLOR = {
    "flat": "#2a78d6",   # slot 1, blue
    "hnsw": "#eb6834",   # slot 2, orange
    "ivfpq": "#1baf7a",  # slot 3, aqua
}

# Text wears ink colours, never the series colour.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#d8d7d2"
SURFACE = "#ffffff"  # opaque white: transparent PNGs render broken on LinkedIn

FIGSIZE = (8.0, 5.0)
DPI = 150
LINEWIDTH = 2.0
MARKERSIZE = 7.0


# ---------------------------------------------------------------------- #
# data
# ---------------------------------------------------------------------- #


def load_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    return rows


def series(
    rows: Sequence[Dict[str, str]], index: str, param: Optional[str], field: str
) -> Tuple[List[int], List[float]]:
    """(n, value) pairs for one index/param, sorted by n.

    `param=None` takes whichever param is present for that index -- used for
    memory, which is a property of the index rather than of a query setting.
    """
    seen: Dict[int, float] = {}
    for r in rows:
        if r["index"] != index:
            continue
        if param is not None and r["param"] != param:
            continue
        seen[int(r["n"])] = float(r[field])
    ns = sorted(seen)
    return ns, [seen[n] for n in ns]


def find_crossings(
    ns: Sequence[int], a: Sequence[float], b: Sequence[float]
) -> List[Tuple[float, float]]:
    """Where series `a` and `b` swap places, interpolated in log-x space.

    The x axis is logarithmic, so interpolating linearly in N would place the
    marker in the wrong spot on screen. Returns (n, value) for each crossing.
    """
    import math

    out: List[Tuple[float, float]] = []
    for i in range(1, len(ns)):
        d0, d1 = a[i - 1] - b[i - 1], a[i] - b[i]
        if d0 == 0.0 or (d0 > 0) == (d1 > 0):
            continue
        t = d0 / (d0 - d1)  # fraction of the way from i-1 to i
        x0, x1 = math.log10(ns[i - 1]), math.log10(ns[i])
        n_cross = 10 ** (x0 + t * (x1 - x0))
        val = a[i - 1] + t * (a[i] - a[i - 1])
        out.append((n_cross, val))
    return out


def cluster_crossings(
    crossings: Sequence[Tuple[float, float]], log_gap: float = 0.05
) -> List[List[Tuple[float, float]]]:
    """Group crossings that sit within `log_gap` decades of each other.

    When two lines touch and separate again over a short span -- a near-tie --
    the strict answer is "two crossings", but drawing two labels a few pixels
    apart is unreadable and overstates what happened. One label per cluster.
    """
    import math

    clusters: List[List[Tuple[float, float]]] = []
    for c in sorted(crossings):
        if clusters and abs(math.log10(c[0]) - math.log10(clusters[-1][-1][0])) <= log_gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return clusters


# ---------------------------------------------------------------------- #
# styling
# ---------------------------------------------------------------------- #


def fmt_count(value: float, _pos: int = 0) -> str:
    if value >= 1000 and value % 1000 == 0:
        return f"{int(value // 1000)}k"
    return f"{value:g}"


def new_axes(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.set_title(title, fontsize=13, color=INK, pad=14, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10, color=INK_SECONDARY, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_SECONDARY, labelpad=8)

    # Recessive grid and axes: the data should be the most prominent thing.
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    return fig, ax


def log_x(ax, ns: Sequence[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(ns))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_count))
    ax.minorticks_off()


def label_end(ax, ns, values, text: str, color: str, dy: float = 0.0) -> None:
    """Direct label at the right end of a line.

    Not decoration -- with one palette slot below the 3:1 contrast floor,
    identity must not depend on colour alone.
    """
    ax.annotate(
        text,
        xy=(ns[-1], values[-1]),
        xytext=(8, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=color,
    )


def save(fig, path: Path) -> None:
    fig.savefig(
        path,
        dpi=DPI,
        facecolor=SURFACE,   # opaque, not transparent
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close(fig)


# ---------------------------------------------------------------------- #
# charts
# ---------------------------------------------------------------------- #


def chart_crossover(rows, out: Path) -> Path:
    flat_n, flat_v = series(rows, "flat", "exact", "p50")
    hnsw_n, hnsw_v = series(rows, "hnsw", "ef=100", "p50")
    ivf_n, ivf_v = series(rows, "ivfpq", "nprobe=4", "p50")

    fig, ax = new_axes(
        "Query latency vs index size",
        "vectors indexed (N, log scale)",
        "p50 query latency (ms, lower is better)",
    )
    log_x(ax, flat_n)

    ax.plot(flat_n, flat_v, color=COLOR["flat"], lw=LINEWIDTH, marker="o",
            ms=MARKERSIZE, label="Flat (exact)")
    ax.plot(hnsw_n, hnsw_v, color=COLOR["hnsw"], lw=LINEWIDTH, marker="s",
            ms=MARKERSIZE, label="HNSW (ef=100)")
    ax.plot(ivf_n, ivf_v, color=COLOR["ivfpq"], lw=LINEWIDTH, marker="^",
            ms=MARKERSIZE, label="IVF+PQ (nprobe=4)")

    label_end(ax, flat_n, flat_v, "Flat", COLOR["flat"])
    label_end(ax, hnsw_n, hnsw_v, "HNSW", COLOR["hnsw"])
    label_end(ax, ivf_n, ivf_v, "IVF+PQ", COLOR["ivfpq"])

    # Crossings are computed, not asserted. Two crossings a hair apart are a
    # near-tie, not two separate crossovers, so cluster them -- otherwise the
    # labels print on top of each other and say something the data does not.
    for cluster in cluster_crossings(find_crossings(flat_n, flat_v, hnsw_v)):
        n_cross = sum(c[0] for c in cluster) / len(cluster)
        val = sum(c[1] for c in cluster) / len(cluster)
        text = (
            f"lines cross\n~N={fmt_count(round(n_cross, -2))}"
            if len(cluster) == 1
            else f"near-tie: lines cross\ntwice around N={fmt_count(round(n_cross, -3))}"
        )
        ax.plot([n_cross], [val], marker="o", ms=11, mfc="none",
                mec=INK_SECONDARY, mew=1.6, zorder=5)
        ax.annotate(
            text,
            xy=(n_cross, val),
            xytext=(0, 46),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_SECONDARY,
            # Opaque backing so the note stays readable where it crosses a line.
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                            shrinkA=0, shrinkB=6),
        )

    # If the approximate index already leads at the smallest N measured, the
    # real crossover is off the left of this chart -- say so rather than let
    # the reader assume the first plotted point is where it happens.
    if hnsw_v[0] < flat_v[0]:
        ax.annotate(
            f"HNSW already ahead at N={fmt_count(flat_n[0])};\n"
            "the crossover sits left of this range",
            xy=(0.02, 0.94),
            xycoords="axes fraction",
            fontsize=8.5,
            color=INK_MUTED,
            va="top",
        )

    ax.set_ylim(bottom=0)
    ax.set_xlim(right=flat_n[-1] * 1.55)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                    bbox_to_anchor=(0.02, 0.80))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    save(fig, out)
    return out


def chart_recall(rows, out: Path) -> Path:
    fig, ax = new_axes(
        "Recall@10 vs index size",
        "vectors indexed (N, log scale)",
        "Recall@10 (vs exact search, higher is better)",
    )

    # Colour = index, line style = parameter. A fourth and fifth hue would
    # break the validated three-slot palette for no gain in meaning.
    spec = [
        ("flat", "exact", "-", "o", "Flat (exact)"),
        ("hnsw", "ef=100", "-", "s", "HNSW (ef=100)"),
        ("hnsw", "ef=25", "--", "s", "HNSW (ef=25)"),
        ("ivfpq", "nprobe=16", "-", "^", "IVF+PQ (nprobe=16)"),
        ("ivfpq", "nprobe=4", "--", "^", "IVF+PQ (nprobe=4)"),
    ]
    ns_ref: List[int] = []
    for index, param, ls, marker, label in spec:
        ns, vals = series(rows, index, param, "recall")
        if not ns:
            continue
        ns_ref = ns
        ax.plot(ns, vals, color=COLOR[index], lw=LINEWIDTH, ls=ls,
                marker=marker, ms=MARKERSIZE, label=label)

    log_x(ax, ns_ref)

    # Label one endpoint per index rather than all five -- selective labels,
    # not a number on every point.
    for index, param, dy in (("flat", "exact", 5), ("hnsw", "ef=100", -7),
                             ("ivfpq", "nprobe=16", 0)):
        ns, vals = series(rows, index, param, "recall")
        if ns:
            label_end(ax, ns, vals, DISPLAY[index], COLOR[index], dy=dy)

    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(right=ns_ref[-1] * 1.5)
    leg = ax.legend(frameon=False, fontsize=9, loc="lower left",
                    bbox_to_anchor=(0.02, 0.03), ncol=2)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    save(fig, out)
    return out


def chart_memory(rows, out: Path) -> Path:
    fig, ax = new_axes(
        "Index memory vs index size",
        "vectors indexed (N, log scale)",
        "index memory (MB, lower is better)",
    )

    ns_ref: List[int] = []
    finals: Dict[str, float] = {}
    for index, marker, label in (("hnsw", "s", "HNSW (vectors + graph)"),
                                 ("flat", "o", "Flat (raw float32)"),
                                 ("ivfpq", "^", "IVF+PQ (PQ codes)")):
        ns, vals = series(rows, index, None, "memory_mb")
        if not ns:
            continue
        ns_ref = ns
        finals[index] = vals[-1]
        ax.plot(ns, vals, color=COLOR[index], lw=LINEWIDTH, marker=marker,
                ms=MARKERSIZE, label=label)
        label_end(ax, ns, vals, f"{vals[-1]:.1f} MB", COLOR[index])

    log_x(ax, ns_ref)

    # The point of the chart is the widening gap, so state it as a ratio taken
    # from the data rather than leaving the reader to measure it off the axis.
    if "hnsw" in finals and "ivfpq" in finals and finals["ivfpq"] > 0:
        ratio = finals["hnsw"] / finals["ivfpq"]
        ax.annotate(
            f"at N={fmt_count(ns_ref[-1])}, HNSW holds {ratio:.0f}x\n"
            f"the memory of IVF+PQ",
            xy=(0.02, 0.94),
            xycoords="axes fraction",
            fontsize=9,
            color=INK_SECONDARY,
            va="top",
        )

    ax.set_ylim(bottom=0)
    ax.set_xlim(right=ns_ref[-1] * 1.6)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left",
                    bbox_to_anchor=(0.02, 0.78))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    save(fig, out)
    return out


# ---------------------------------------------------------------------- #
# entry point
# ---------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render committed scaling results as PNG charts."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help=f"results CSV to read (default: {DEFAULT_CSV})")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: alongside the CSV)")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"no such CSV: {args.csv}")

    rows = load_rows(args.csv)
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = sorted({int(r["n"]) for r in rows})
    print(f"read {len(rows)} rows from {args.csv}")
    print(f"  sizes   : {sizes}")
    print(f"  indexes : {sorted({r['index'] for r in rows})}")
    print()

    for fn, name in (
        (chart_crossover, "crossover.png"),
        (chart_recall, "recall_vs_n.png"),
        (chart_memory, "memory.png"),
    ):
        path = fn(rows, out_dir / name)
        size = path.stat().st_size
        status = "OK" if size > 0 else "EMPTY -- FAILED"
        print(f"  [{status}] {path}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
