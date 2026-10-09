"""Live progress views for the demo and use-case notebooks.

`race(func, items)` runs a plain for loop and then `func.map(items)`, and redraws
one HTML panel while they run. Both bars share one time axis (the for loop's
time), so the @adaptive_exec bar visibly stops short when it is faster.

`Steps(total)` is a step counter for long cells: call `.step(label)` between
measurements. It redraws only between steps, so it adds nothing to the timings.

`explain(func)` replays what the last func.map() call did (data -> probe ->
features -> MLP -> decision -> run) as a GIF, using the real numbers it stored.

Plain HTML (no ipywidgets), explicit colours so it reads the same in dark themes.
"""

import textwrap
import threading
import time

from IPython.display import HTML, display

COLORS = {"sequential": "#2a78d6", "threading": "#eb6834", "multiprocessing": "#1baf7a"}
GREY, TRACK, INK, INK2, PAPER = "#9a9994", "#e4e3df", "#0b0b0b", "#52514e", "#fcfcfb"
TICK_S = 0.25            # redraw interval of the background ticker


def _fmt(seconds):
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"


def _row(label, frac, color, text):
    frac = min(max(frac, 0.0), 1.0)
    return (f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
            f'<div style="width:120px;font-family:monospace;color:{INK}">{label}</div>'
            f'<div style="flex:1;max-width:460px;height:18px;background:{TRACK};border-radius:4px">'
            f'<div style="width:{frac * 100:.1f}%;height:100%;background:{color};border-radius:4px"></div></div>'
            f'<div style="min-width:240px;color:{INK2}">{text}</div></div>')


def _panel(title, rows, footer=""):
    return HTML(f'<div style="background:{PAPER};color:{INK};padding:10px 14px;border:1px solid {TRACK};'
                f'border-radius:8px;font-size:13px;margin:4px 0">'
                f'<div style="font-weight:bold;margin-bottom:4px">{title}</div>{"".join(rows)}'
                f'<div style="color:{INK2};margin-top:4px;font-size:12px">{footer}</div></div>')


def race(func, items, same=None, title=None, chunks=100):
    """For loop vs func.map(items, cache=False), drawn live. Returns loop/smart/mode/job."""
    items = list(items)
    n = len(items)
    title = f"{title or func.__name__} · {n:,} items"
    s = {"phase": "loop", "done": 0, "start": time.perf_counter(), "t_loop": None, "t_smart": None, "mode": None}

    def render():
        now = time.perf_counter() - s["start"]
        if s["phase"] == "loop":
            rows = [_row("for loop", s["done"] / n, GREY, f"{s['done']:,} / {n:,} items · {_fmt(now)}"),
                    _row("@adaptive_exec", 0, GREY, "waiting")]
        else:
            t_loop = s["t_loop"]
            t_smart = s["t_smart"] if s["t_smart"] is not None else now
            if s["mode"] is None:
                smart_text, color = f"running · {_fmt(t_smart)}", "#8f6bd6"
            else:
                speed = t_loop / t_smart
                if speed > 1.15:
                    gain = f"{speed:.1f}x faster"
                elif speed > 0.87:
                    gain = "about the same"
                else:
                    gain = f"+{_fmt(t_smart - t_loop)} to decide"
                smart_text, color = f"{_fmt(t_smart)} · AI picked <b>{s['mode']}</b> · {gain}", COLORS[s["mode"]]
            scale = max(t_loop, t_smart)
            rows = [_row("for loop", t_loop / scale, GREY, f"{_fmt(t_loop)}"),
                    _row("@adaptive_exec", t_smart / scale, color, smart_text)]
        return _panel(title, rows, "bar length = time used")

    handle = display(render(), display_id=True)
    stop = threading.Event()

    def ticker():
        while not stop.wait(TICK_S):
            handle.update(render())

    thread = threading.Thread(target=ticker, daemon=True)
    thread.start()
    try:
        step = max(1, n // chunks)
        plain = []
        t0 = time.perf_counter()
        for i in range(0, n, step):              # same work as [func(x) for x in items], in ~100 chunks
            plain.extend([func(x) for x in items[i:i + step]])
            s["done"] = len(plain)
        s["t_loop"] = time.perf_counter() - t0

        s["phase"], s["start"] = "map", time.perf_counter()
        t0 = time.perf_counter()
        smart = func.map(items, cache=False)     # the AI picks the mode
        s["t_smart"] = time.perf_counter() - t0
        s["mode"] = func.last_run.mode
    except BaseException:
        stop.set()
        thread.join()
        handle.update(_panel(title, [], "stopped by an error"))
        raise
    stop.set()
    thread.join()
    handle.update(render())

    ok = same(smart, plain) if same else smart == plain
    assert ok, "results must match the for loop"
    return {"job": func.__name__, "loop": s["t_loop"], "smart": s["t_smart"], "mode": s["mode"]}


class Steps:
    """Step counter for long cells; redraws only when .step() is called."""

    def __init__(self, total, title="progress"):
        self.total, self.title, self.n = total, title, 0
        self.start = time.perf_counter()
        self.handle = display(self._render("starting"), display_id=True)

    def _render(self, label):
        elapsed = time.perf_counter() - self.start
        eta = elapsed / self.n * (self.total - self.n) if self.n else None
        right = f"{self.n} / {self.total} · {elapsed:.0f} s" + (f" · ~{eta:.0f} s left" if eta and self.n < self.total else "")
        return _panel(self.title, [_row("", self.n / self.total, "#5b3fb0", right)], label)

    def step(self, label):
        """Show what runs next (call before the measurement)."""
        self.handle.update(self._render(label))

    def done(self, label=None):
        """Mark one step finished."""
        self.n += 1
        self.handle.update(self._render(label or ("finished" if self.n >= self.total else "")))


# ------------------------------------------------------------ decision flow

STAGES = ["DATA", "PROBE", "FEATURES", "MLP MODEL", "DECISION", "RUN"]
KEY_FEATURES = [("time_per_item", "time/item", lambda v: _fmt(v)),
                ("cpu_ratio", "CPU busy", lambda v: f"{v:.0%}"),
                ("thread_probe_speedup", "2 threads", lambda v: f"{v:.2f}x"),
                ("pickle_time_ratio", "send/work", lambda v: f"{v:.2f}"),
                ("cpu_count", "cores", lambda v: f"{v:.0f}")]


def _stage_text(d, func_name):
    """What each box says, taken from the real Decision of the last .map() call."""
    rest = d.n_items - d.probed_items
    used = {"sequential": f"for loop\n{rest:,} items, 1 core",
            "threading": f"ThreadPool({d.n_workers}).map\n{rest:,} items",
            "multiprocessing": f"Pool({d.n_workers}).map\n{rest:,} items, {d.n_workers} processes"}[d.mode]
    feats = ("\n".join(f"{label}: {fmt(d.features[k])}" for k, label, fmt in KEY_FEATURES)
             if d.features else "not needed")
    return [f"{d.n_items:,} items\n\n{func_name}(item)",
            f"ran the first\n{d.probed_items:,} items for real\n(results kept)",
            feats + ("\n+ 8 more" if d.features else ""),
            "13 -> hidden -> 3\nneural network" if d.probabilities else "skipped:\n" + textwrap.fill(d.reason, 18),
            "",
            used + f"\n\n= {d.n_items:,} results,\nsame order"]


def explain(func, fps=12, path=None):
    """Animated replay of what the last func.map() call did, with its real numbers.

    Returns an Image (a GIF) that plays in the notebook; `path` also saves it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyBboxPatch
    from IPython.display import Image

    d = func.last_run
    texts = _stage_text(d, func.__name__)
    color = COLORS[d.mode]
    hold, per_stage = fps * 2, int(fps * 0.9)          # frames per stage, frames held at the end
    n_frames = per_stage * len(STAGES) + hold

    fig, ax = plt.subplots(figsize=(13, 4.2), dpi=80)
    fig.patch.set_facecolor(PAPER)
    ax.set_xlim(0, 6.2); ax.set_ylim(0, 3); ax.axis("off")
    xs = [0.1 + i * 1.03 for i in range(6)]
    w, h, y0 = 0.86, 2.0, 0.45

    def draw(frame):
        ax.clear(); ax.set_xlim(0, 6.2); ax.set_ylim(0, 3); ax.axis("off")
        stage, t = divmod(frame, per_stage)
        t = min(t / (per_stage - 1), 1.0) if stage < len(STAGES) else 1.0
        stage = min(stage, len(STAGES) - 1)
        ax.text(0.1, 2.8, f"What happened inside {func.__name__}.map()", fontsize=14, fontweight="bold", color=INK)
        ax.text(6.1, 2.8, f"-> {d.mode}", fontsize=14, fontweight="bold", color=color if stage >= 4 else TRACK, ha="right")
        for i, (x, name) in enumerate(zip(xs, STAGES)):
            on = i <= stage
            skipped = (i == 2 and not d.features) or (i in (3, 4) and not d.probabilities)
            edge = color if (i == stage and i >= 4) else ("#5b3fb0" if i == stage else (INK2 if on else TRACK))
            ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                        fc="white" if on else PAPER, ec=edge, lw=2.5 if i == stage else 1.2))
            ax.text(x + w / 2, y0 + h - 0.18, name, ha="center", fontsize=10, fontweight="bold",
                    color=(GREY if skipped else INK) if on else TRACK)
            if i < 5:                                   # arrow to the next box, with a moving dot
                ax.annotate("", (xs[i + 1] - 0.01, y0 + h / 2), (x + w + 0.01, y0 + h / 2),
                            arrowprops=dict(arrowstyle="->", color=INK2 if i < stage else TRACK, lw=1.5))
                if i == stage - 1 and t < 1:
                    ax.plot(x + w + 0.17 * t, y0 + h / 2, "o", color="#5b3fb0", ms=7)
            if not on:
                continue
            if i == 3 and d.probabilities:              # tiny network, lights up layer by layer
                cols = [(x + 0.18, 5), (x + 0.43, 4), (x + 0.68, 3)]
                pts = [[(cx, y0 + 1.45 - k * (1.0 / max(n - 1, 1))) for k in range(n)] for cx, n in cols]
                lit = 2 if i < stage else int(t * 2.99)
                for L in range(2):
                    for a in pts[L]:
                        for b in pts[L + 1]:
                            ax.plot([a[0], b[0]], [a[1], b[1]], color="#5b3fb0" if L < lit else TRACK, lw=0.6, alpha=0.7)
                for L, layer in enumerate(pts):
                    for k, (px, py) in enumerate(layer):
                        c = COLORS[list(COLORS)[k]] if L == 2 else ("#5b3fb0" if L <= lit else GREY)
                        big = L == 2 and list(COLORS)[k] == d.mode and stage > 3
                        ax.plot(px, py, "o", color=c, ms=10 if big else 6)
                ax.text(x + w / 2, y0 + 0.12, texts[3], ha="center", fontsize=8, color=INK2)
            elif i == 4:                                # probabilities grow
                if d.probabilities:
                    grow = t if i == stage else 1.0
                    for k, m in enumerate(COLORS):
                        p = d.probabilities[m] * grow
                        yy = y0 + h - 0.6 - k * 0.42
                        ax.add_patch(plt.Rectangle((x + 0.06, yy), 0.74 * p, 0.22, color=COLORS[m]))
                        ax.text(x + 0.06, yy + 0.27, f"{m} {p:.0%}", fontsize=7.5, color=INK)
                else:
                    ax.text(x + w / 2, y0 + h / 2, f"{d.mode}\n(no model needed)", ha="center", va="center",
                            fontsize=9, color=color)
            else:
                ax.text(x + w / 2, y0 + h / 2 - 0.12, texts[i], ha="center", va="center", fontsize=8.5,
                        color=GREY if skipped else INK, linespacing=1.4)
        if stage == len(STAGES) - 1:
            ax.text(0.1, 0.1, f"decision overhead {d.overhead_s * 1e3:.0f} ms  |  whole call {_fmt(d.total_s)}"
                    f"  |  reason: {d.reason}", fontsize=9, color=INK2)

    anim = FuncAnimation(fig, draw, frames=n_frames)
    tmp = path or "_flow.gif"
    anim.save(tmp, writer=PillowWriter(fps=fps))
    plt.close(fig)
    with open(tmp, "rb") as f:
        data = f.read()
    if path is None:
        import os
        os.remove(tmp)
    return Image(data=data, format="gif")
