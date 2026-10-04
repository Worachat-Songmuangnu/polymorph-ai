"""Demo + benchmark for @adaptive_exec.   Run from the project root:

    python examples/demo.py              # stage demo: 8 everyday jobs, polymorph_ai vs every fixed mode
    python examples/demo.py --overhead   # first-call cost of polymorph_ai by job length

The jobs below are ordinary code written for this demo, NOT the 52 training
templates, so the model has never seen them.
Results are saved to examples/results/ (CSV + PNG for the slides).
"""

import argparse
import json
import os
import platform
import random
import statistics
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polymorph_ai import adaptive_exec, run, warm_up       # noqa: E402
from polymorph_ai.features import usable_cpu_count            # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODES = ["sequential", "threading", "multiprocessing"]


# ---------------------------------------------------------------- the jobs

@adaptive_exec
def count_primes(limit):
    """Pure Python maths (holds the GIL)."""
    count = 0
    for n in range(2, limit):
        if all(n % d for d in range(2, int(n ** 0.5) + 1)):
            count += 1
    return count


@adaptive_exec
def call_api(user_id):
    """Pretend web request: mostly waiting."""
    time.sleep(0.03)
    return {"id": user_id, "ok": True}


@adaptive_exec
def compress(block):
    """zlib releases the GIL while it compresses."""
    return len(zlib.compress(block, 9))


@adaptive_exec
def word_count(text):
    """Text processing on a ~50 KB string per item."""
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return max(counts.values())


@adaptive_exec
def parse_json(doc):
    """Parse a JSON document and sum a field."""
    return sum(row["price"] for row in json.loads(doc))


@adaptive_exec
def checksum(block):
    """Big input (4 MB) but very little work: sending it to a process costs more than the work."""
    return zlib.crc32(block)


@adaptive_exec
def sort_rows(rows):
    """Sort 20,000 numbers and keep the top 10."""
    return sorted(rows)[:10]


@adaptive_exec
def add_tax(price):
    """Tiny work per item: nothing beats a plain loop."""
    return round(price * 1.07, 2)


def make_jobs():
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 900).split()
    texts = [" ".join(words[i:] + words[:i]) for i in range(120)]
    docs = [json.dumps([{"id": j, "price": j * 0.5, "name": f"item{j}"} for j in range(2000)]) for _ in range(150)]
    blocks = [os.urandom(200_000) + bytes(600_000) for _ in range(48)]
    return [
        ("count primes   (CPU, pure Python)", count_primes, [30_000] * 48),
        ("call API       (waiting on network)", call_api, list(range(120))),
        ("compress       (zlib, releases GIL)", compress, blocks),
        ("word count     (text, 50 KB items)", word_count, texts),
        ("parse JSON     (150 docs)", parse_json, docs),
        ("checksum       (4 MB items, little work)", checksum, [os.urandom(4_000_000) for _ in range(40)]),
        ("sort rows      (20k numbers per item)", sort_rows, [[random.random() for _ in range(20_000)] for _ in range(100)]),
        ("add tax        (tiny items)", add_tax, [i * 0.1 for i in range(20_000)]),
    ]


def best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def demo(repeats):
    print(f"{platform.system()} | {usable_cpu_count()} CPUs | Python {platform.python_version()}")
    print("starting worker pools once (a long-running program pays this only once)...\n")
    warm_up()
    rows = []
    for name, f, items in make_jobs():
        expected = [f.func(x) for x in items[:3]]
        f.map(items)                                    # store the decision for the repeat-call timing
        t = {m: best_of(lambda: run(f.func, items, mode=m, _target=f), repeats) for m in MODES}
        t["first_call"] = best_of(lambda: f.map(items, cache=False), repeats)    # probe + model every time
        t["polymorph_ai"] = best_of(lambda: f.map(items), repeats)                 # repeat calls reuse the decision
        assert f.map(items)[:3] == expected
        f.map(items, cache=False)
        d = f.last_run                                  # the model's own decision (not the cached copy)
        fastest = min(MODES, key=t.get)
        rows.append({"job": name.split("(")[0].strip(), **{f"t_{k}": v for k, v in t.items()},
                     "picked": d.mode, "fastest": fastest, "overhead_s": d.overhead_s, "reason": d.reason})
        print(f"{name}")
        print("   " + "   ".join(f"{m[:5]} {t[m]:6.3f}s" for m in MODES)
              + f"   | polymorph_ai 1st {t['first_call']:6.3f}s  repeat {t['polymorph_ai']:6.3f}s -> {d.mode}"
              + ("  (fastest)" if d.mode == fastest else f"  (fastest: {fastest})"))

    total = {k: sum(r[f"t_{k}"] for r in rows) for k in MODES + ["first_call", "polymorph_ai"]}
    best_total = sum(min(r[f"t_{m}"] for m in MODES) for r in rows)
    print("\nTotal time for all jobs (lower is better):")
    for k, v in total.items():
        label = {"first_call": "polymorph_ai, 1st call", "polymorph_ai": "polymorph_ai, repeat"}.get(k, "always " + k)
        print(f"   {label:22s} {v:6.2f} s   (+{(v / best_total - 1) * 100:5.1f}% vs perfect picks)")
    save(rows, "demo")
    plot_demo(rows)


def overhead(repeats):
    """First-call cost of polymorph_ai (probe + model, no cache) by job length.

    For each job length: plain loop, the best fixed mode, and polymorph_ai's first call.
    Answers "how long must a job be before polymorph_ai pays off?".
    """
    warm_up()
    jobs = [("count primes", count_primes, lambda n: [30_000] * n, "multiprocessing"),
            ("call API", call_api, lambda n: list(range(n)), "threading"),
            ("add tax", add_tax, lambda n: [i * 0.1 for i in range(n)], "sequential")]
    sizes = {"count primes": [2, 4, 8, 16, 32, 64, 128, 256],
             "call API": [4, 8, 16, 32, 64, 128, 256, 512],
             "add tax": [10, 100, 1_000, 10_000, 100_000, 1_000_000]}
    rows = []
    print(f"{'job':14s} {'items':>9s} {'plain loop':>10s} {'best mode':>10s} {'polymorph_ai':>10s}  vs best  picked")
    for name, f, make, best_mode in jobs:
        for n in sizes[name]:
            items = make(n)
            plain = best_of(lambda: [f.func(x) for x in items], repeats)
            best = best_of(lambda: run(f.func, items, mode=best_mode, _target=f), repeats)
            auto = best_of(lambda: f.map(items, cache=False), repeats)
            best = min(best, plain)
            rows.append({"job": name, "n_items": n, "plain_loop_s": plain, "best_fixed_s": best,
                         "conductor_first_call_s": auto, "picked": f.last_run.mode,
                         "decision_overhead_s": f.last_run.overhead_s})
            print(f"{name:14s} {n:9,d} {plain:9.3f}s {best:9.3f}s {auto:9.3f}s  {(auto / best - 1) * 100:+6.0f}%  "
                  f"{f.last_run.mode}")
    save(rows, "overhead")
    plot_overhead(rows)


def plot_overhead(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"count primes": "#1baf7a", "call API": "#eb6834", "add tax": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for job, color in colors.items():
        r = [x for x in rows if x["job"] == job]
        ax.plot([x["plain_loop_s"] for x in r], [x["conductor_first_call_s"] / x["best_fixed_s"] for x in r],
                "o-", color=color, lw=2, label=job)
    ax.axhline(1, color="#52514e", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("job length as a plain loop (seconds, log scale)")
    ax.set_ylabel("polymorph_ai 1st call / best fixed mode")
    ax.set_title(f"The probe costs a lot on short jobs and almost nothing on long ones ({platform.system()})",
                 loc="left", fontweight="bold", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(RESULTS, f"overhead_{platform.system().lower()}.png")
    fig.savefig(path, dpi=150)
    print("saved", path)


def save(rows, name):
    import csv
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}_{platform.system().lower()}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved", path)


def plot_demo(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"sequential": "#2a78d6", "threading": "#eb6834", "multiprocessing": "#1baf7a",
              "first_call": "#8a8985", "polymorph_ai": "#0b0b0b"}
    names = {"first_call": "polymorph_ai (1st call)", "polymorph_ai": "polymorph_ai (repeat)"}
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.16
    for k, key in enumerate(MODES + ["first_call", "polymorph_ai"]):
        rel = [r[f"t_{key}"] / min(r[f"t_{m}"] for m in MODES) for r in rows]
        ax.bar([i + (k - 2) * width for i in range(len(rows))], rel, width, color=colors[key],
               label=names.get(key, f"always {key}"))
    ax.axhline(1, color="#52514e", lw=1, ls="--")
    ax.set_xticks(range(len(rows)), [r["job"] for r in rows])
    ax.set_ylabel("time / fastest mode (1.0 = best)")
    ax.set_title(f"polymorph_ai vs fixed modes on jobs the model never saw ({platform.system()})", loc="left",
                 fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper left", fontsize=8)
    ax.set_ylim(0, min(ax.get_ylim()[1], 6))
    fig.tight_layout()
    path = os.path.join(RESULTS, f"demo_{platform.system().lower()}.png")
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--overhead", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    overhead(args.repeats) if args.overhead else demo(args.repeats)
