"""Feature extraction shared by the data collector and the decorator.

The same code must be used in both places, otherwise the model is trained on
features that the decorator measures differently at run time.

13 features in 4 groups (see CLAUDE.md):
  call-site    : n_items, input_bytes, cpu_count, n_workers
  ast          : max_loop_depth, arith_op_count, has_io_call, has_gil_release_call
  probe        : time_per_item, cpu_ratio, time_cv, pickle_time_ratio
  thread probe : thread_probe_speedup

A workload is "apply func to every item in a list", like map(func, items).
"""

import ast
import functools
import inspect
import os
import pickle
import statistics
import textwrap
import threading
import time

FEATURE_NAMES = [
    # call-site
    "n_items", "input_bytes", "cpu_count", "n_workers",
    # ast
    "max_loop_depth", "arith_op_count", "has_io_call", "has_gil_release_call",
    # probe
    "time_per_item", "cpu_ratio", "time_cv", "pickle_time_ratio",
    # thread probe
    "thread_probe_speedup",
]

# The probe runs real items one by one until it has used this much time
# (at least PROBE_MIN_ITEMS, at most a quarter of all items or PROBE_MAX_ITEMS). The outputs are
# kept, so the decorator does not waste this work.
PROBE_TARGET_S = 0.05
PROBE_MIN_ITEMS = 3
PROBE_MAX_ITEMS = 2000      # tiny items: more adds bookkeeping, not information
PICKLE_SAMPLE = 20         # items used to measure the pickling cost


# ---------------------------------------------------------------- cpu clock
# CPU time used by the current thread, in seconds.
# Linux: time.thread_time() is precise. Windows: thread_time()/process_time()
# only tick every 15.6 ms, useless for a probe of a few ms. Instead we read
# the thread's CPU cycle counter (QueryThreadCycleTime) and convert cycles to
# seconds with a rate measured once by a short busy loop.

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32")
    _kernel32.GetCurrentThread.restype = wintypes.HANDLE
    _kernel32.QueryThreadCycleTime.argtypes = [wintypes.HANDLE,
                                               ctypes.POINTER(ctypes.c_ulonglong)]

    def _thread_cycles():
        cycles = ctypes.c_ulonglong()
        _kernel32.QueryThreadCycleTime(_kernel32.GetCurrentThread(), ctypes.byref(cycles))
        return cycles.value

    @functools.lru_cache(maxsize=1)
    def _cycles_per_second():
        # Busy loop: the thread uses 100% CPU, so cycles / wall = rate.
        c0, t0 = _thread_cycles(), time.perf_counter()
        while time.perf_counter() - t0 < 0.02:
            pass
        return (_thread_cycles() - c0) / (time.perf_counter() - t0)

    def thread_cpu_time():
        rate = _cycles_per_second()       # calibrate first, outside the reading
        return _thread_cycles() / rate
else:
    thread_cpu_time = time.thread_time


# ---------------------------------------------------------------- call-site

def usable_cpu_count():
    """Number of CPUs this process may actually use (not just installed)."""
    if hasattr(os, "process_cpu_count"):          # Python 3.13+
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):        # Linux
        n = len(os.sched_getaffinity(0))
    else:                                         # Windows, older Python
        n = os.cpu_count()
    return n or 1


# ---------------------------------------------------------------------- ast

# Calls that mean "waits for I/O". Matched against the dotted call name,
# e.g. `time.sleep(...)` -> "time.sleep".
IO_CALLS = {"open", "print", "input", "time.sleep", "os.read", "os.write",
            "os.fsync", "asyncio.sleep"}
IO_PREFIXES = ("socket.", "requests.", "urllib.", "http.", "subprocess.",
               "shutil.", "ftplib.", "smtplib.", "sqlite3.")
# Method names that are almost always file/socket I/O: f.read(), s.recv() ...
IO_METHODS = {"read", "write", "readline", "readlines", "recv", "send",
              "sendall", "flush"}

# C functions that do heavy work and release the GIL, so threads can run
# them in parallel. Only functions we have checked; plain `math` is not here.
GIL_RELEASE_CALLS = {"np.dot", "numpy.dot", "np.matmul", "numpy.matmul",
                     "np.sort", "numpy.sort"}
GIL_RELEASE_PREFIXES = ("zlib.", "bz2.", "lzma.", "hashlib.",
                        "np.linalg.", "numpy.linalg.", "np.fft.", "numpy.fft.")

LOOP_NODES = (ast.For, ast.AsyncFor, ast.While)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _dotted_name(node):
    """`zlib.compress` -> "zlib.compress", `f.read` -> "?.read" if f is complex."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    parts.append(node.id if isinstance(node, ast.Name) else "?")
    return ".".join(reversed(parts))


def _loop_depth(node, depth=0):
    if isinstance(node, LOOP_NODES):
        depth += 1
    elif isinstance(node, COMPREHENSIONS):
        depth += len(node.generators)
    child_depths = [_loop_depth(child, depth) for child in ast.iter_child_nodes(node)]
    return max([depth] + child_depths)


@functools.lru_cache(maxsize=256)
def ast_features(func):
    """Static features read from the function's source code (cached per function).

    Only the function's own body is inspected, not the functions it calls.
    """
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        # No source available (builtin, lambda in REPL, ...): neutral values.
        return {"max_loop_depth": 0, "arith_op_count": 0,
                "has_io_call": 0, "has_gil_release_call": 0}

    arith = 0
    has_io = False
    has_gil_release = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            arith += 1
            if isinstance(node.op, ast.MatMult):      # a @ b on numpy arrays
                has_gil_release = True
        elif isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            method = name.rsplit(".", 1)[-1]
            if (name in IO_CALLS or name.startswith(IO_PREFIXES)
                    or ("." in name and method in IO_METHODS)):
                has_io = True
            if name in GIL_RELEASE_CALLS or name.startswith(GIL_RELEASE_PREFIXES):
                has_gil_release = True

    return {"max_loop_depth": _loop_depth(tree), "arith_op_count": arith,
            "has_io_call": int(has_io), "has_gil_release_call": int(has_gil_release)}


# -------------------------------------------------------------------- probe

def probe(func, items):
    """Run the first few real items sequentially and measure them.

    Returns (features, results) where results are the outputs of the probed
    items, so the decorator can reuse them instead of computing them twice.
    """
    max_items = min(len(items), PROBE_MAX_ITEMS, max(PROBE_MIN_ITEMS, len(items) // 4))
    item_times, results, pickle_times, input_sizes = [], [], [], []

    cpu_start = thread_cpu_time()
    wall_start = time.perf_counter()
    for item in items[:max_items]:
        t0 = time.perf_counter()
        out = func(item)
        item_times.append(time.perf_counter() - t0)
        results.append(out)
        if (len(item_times) >= PROBE_MIN_ITEMS
                and time.perf_counter() - wall_start >= PROBE_TARGET_S):
            break
    wall = time.perf_counter() - wall_start
    cpu = thread_cpu_time() - cpu_start

    # What multiprocessing would pay per item: send input, get output back.
    # Measured outside the timed loop above so it does not inflate cpu_ratio.
    # A spread-out sample is enough for the mean; tiny items can mean
    # tens of thousands of probed items.
    step = max(1, len(results) // PICKLE_SAMPLE)
    for item, out in list(zip(items, results))[::step][:PICKLE_SAMPLE]:
        t0 = time.perf_counter()
        data = pickle.dumps(item)
        pickle.loads(data)
        pickle.loads(pickle.dumps(out))
        pickle_times.append(time.perf_counter() - t0)
        input_sizes.append(len(data))

    time_per_item = statistics.mean(item_times)
    features = {
        "input_bytes": statistics.mean(input_sizes),
        "time_per_item": time_per_item,
        "cpu_ratio": min(cpu / wall, 1.0) if wall > 0 else 0.0,
        "time_cv": (statistics.pstdev(item_times) / time_per_item
                    if time_per_item > 0 else 0.0),
        "pickle_time_ratio": (statistics.mean(pickle_times) / time_per_item
                              if time_per_item > 0 else 0.0),
    }
    return features, results, item_times


def thread_probe(func, items, time_per_item):
    """Run the NEXT items (not the probed ones again) on 2 threads.

    Compares their wall time with what they would take sequentially
    (len(items) * time_per_item from the probe):
    ~2.0 means threads really run in parallel (I/O or GIL-releasing code),
    ~1.0 or less means the GIL serialises them.
    The outputs are kept, so this work is not wasted. Items are never run
    twice, so functions with side effects (writing files, ...) stay correct.

    Returns (speedup, results). With fewer than 2 items left: (1.0, []).
    """
    if len(items) < 2:
        return 1.0, []
    results = [None] * len(items)

    def work(indexes):
        for i in indexes:
            results[i] = func(items[i])

    halves = [range(0, len(items), 2), range(1, len(items), 2)]
    threads = [threading.Thread(target=work, args=(h,)) for h in halves]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    speedup = len(items) * time_per_item / wall if wall > 0 else 1.0
    return speedup, results


# ---------------------------------------------------------------------- all

def extract_features(func, items, n_workers):
    """All 13 features for running func over items with n_workers workers.

    Returns (features, probe_results, overhead_s, extra_s).
    probe_results are the outputs of the first len(probe_results) items.
    overhead_s is the total time spent here; extra_s is the part that did
    no useful work (pickle test, AST, bookkeeping), i.e. overhead_s minus
    the time spent computing real items.
    """
    t0 = time.perf_counter()
    probe_feats, results, item_times = probe(func, items)
    t_probe = time.perf_counter()
    # Skip the first probed item when estimating sequential speed: it may
    # include one-off warm-up.
    steady = item_times[1:] if len(item_times) >= 3 else item_times
    next_items = items[len(results):len(results) + len(results)]
    speedup, thread_results = thread_probe(func, next_items, statistics.mean(steady))
    t_thread = time.perf_counter()
    results = results + thread_results

    features = {
        "n_items": len(items),
        "input_bytes": probe_feats["input_bytes"],
        "cpu_count": usable_cpu_count(),
        "n_workers": n_workers,
        **ast_features(func),
        "time_per_item": probe_feats["time_per_item"],
        "cpu_ratio": probe_feats["cpu_ratio"],
        "time_cv": probe_feats["time_cv"],
        "pickle_time_ratio": probe_feats["pickle_time_ratio"],
        "thread_probe_speedup": speedup,
    }
    overhead = time.perf_counter() - t0
    useful = sum(item_times) + (t_thread - t_probe if thread_results else 0.0)
    return features, results, overhead, max(overhead - useful, 0.0)
