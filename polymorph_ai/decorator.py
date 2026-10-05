"""@adaptive_exec: run a per-item function over a list in the best mode.

    from polymorph_ai import adaptive_exec

    @adaptive_exec
    def process(item):
        ...

    process(one_item)            # a normal call, nothing changes
    results = process.map(items) # polymorph_ai picks sequential / threading / multiprocessing
    print(process.last_run)      # what it picked and why

How one .map() call works:
  1. safety checks: empty / tiny list, one worker, nested call, can items and
     the function be sent to other processes at all?
  2. probe: run the first items for ~50 ms and measure them (features.py, the
     exact code used to collect the training data). Their results are kept.
  3. the MLP (model.py) turns the 13 features into a mode.
  4. the remaining items run in that mode on a pool that stays alive for the
     next call (the model was trained on warm-pool timings).
"""

import atexit
import functools
import importlib
import logging
import math
import multiprocessing
import os
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.pool import ThreadPool

try:
    import cloudpickle      # sends functions written in Jupyter/REPL to worker processes by value
except ImportError:         # pragma: no cover
    cloudpickle = None

from .features import PROBE_MIN_ITEMS, ast_features, extract_features, thread_cpu_time, usable_cpu_count
from .model import ModeModel

log = logging.getLogger("polymorph_ai")

MODES = ("sequential", "threading", "multiprocessing")
# Below this estimated sequential time (whole job, or what is left after the probe), skip the model:
# no pool can win back its own overhead (the training data starts at 10 ms).
SMALL_JOB_S = 0.01
# A cached decision is reused this many times, then the job is probed again.
CACHE_USES = 50
# Median time to start a process pool in our dataset (pool_startup_s):
# Windows (spawn) 0.25 s, Linux (fork/forkserver) 0.07 s. Used only when the pool is not running yet.
COLD_START_S = {"spawn": 0.25, "fork": 0.07, "forkserver": 0.07}


def _start_method():
    # Read the method without fixing it, so the user can still call set_start_method() later.
    # get_all_start_methods() lists the platform default first.
    return multiprocessing.get_start_method(allow_none=True) or multiprocessing.get_all_start_methods()[0]


# ------------------------------------------------------------------- result

@dataclass
class Decision:
    """What one .map() call did. `print(decision)` gives a one-line summary."""
    function: str
    mode: str
    reason: str
    n_items: int
    n_workers: int
    probabilities: dict = None      # model output, None if the model was not asked
    features: dict = None           # the 13 features, None if not measured
    probed_items: int = 0           # items already computed during the probe
    overhead_s: float = 0.0         # time spent deciding that did no useful work
    total_s: float = 0.0
    notes: list = field(default_factory=list)

    def __str__(self):
        p = f" p={self.probabilities[self.mode]:.2f}" if self.probabilities else ""
        return (f"[polymorph_ai] {self.function}: {self.n_items:,} items -> {self.mode}{p} "
                f"({self.reason}) | overhead {self.overhead_s * 1e3:.1f} ms | total {self.total_s:.3f} s")


# -------------------------------------------------------------------- pools

_pools = {}
_pools_lock = threading.Lock()
_worker = threading.local()          # .active is True inside our pool workers
_decisions = {}                      # cache key -> [mode, uses left]
_model = None
_model_lock = threading.Lock()


def _mark_worker():
    _worker.active = True


class _ThreadPool(ThreadPool):
    """ThreadPool that does not lock the program's default start method
    (the stock one calls multiprocessing.get_context(), which does)."""

    def __init__(self, processes, initializer):
        multiprocessing.pool.Pool.__init__(self, processes, initializer,
                                           context=multiprocessing.get_context(_start_method()))


def _get_pool(mode, n_workers):
    with _pools_lock:
        pool = _pools.get((mode, n_workers))
        if pool is None:
            if mode == "threading":
                pool = _ThreadPool(n_workers, _mark_worker)
            else:   # explicit context: same reason as _ThreadPool
                pool = multiprocessing.get_context(_start_method()).Pool(n_workers, initializer=_mark_worker)
            _pools[(mode, n_workers)] = pool
            _register_shutdown()
        return pool


_shutdown_registered = False


def _register_shutdown():
    # atexit runs the last registered function first. multiprocessing registers its own
    # exit hook when the first pool starts; if that hook ran before ours, it would kill
    # the pool's workers, the pool would start new ones, and those would be left running
    # after the program (or Jupyter kernel) exits. Registering after the first pool
    # exists makes our shutdown() run first.
    global _shutdown_registered
    if not _shutdown_registered:
        atexit.register(shutdown)
        _shutdown_registered = True


def _pool_is_warm(mode, n_workers):
    return (mode, n_workers) in _pools


def warm_up(n_workers=None, modes=("threading", "multiprocessing")):
    """Start the pools (and the Windows CPU-clock calibration) now, so the first
    .map() call does not pay these one-off costs."""
    thread_cpu_time()
    _get_model()
    n_workers = n_workers or usable_cpu_count()
    for mode in modes:
        _get_pool(mode, n_workers).map(int, range(n_workers), chunksize=1)


def shutdown():
    """Stop all pools (called automatically when the program exits)."""
    with _pools_lock:
        for pool in _pools.values():
            # pool.terminate() can hang while the interpreter is shutting down (seen in
            # Jupyter kernels on Windows), which left the worker processes running after
            # the kernel was gone. Give it a few seconds, then kill the workers directly.
            workers = list(getattr(pool, "_pool", []))
            stopper = threading.Thread(target=pool.terminate, daemon=True)
            stopper.start()
            stopper.join(timeout=3)
            for p in workers + list(getattr(pool, "_pool", [])):
                if hasattr(p, "kill") and p.is_alive():
                    p.kill()
        _pools.clear()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = ModeModel()
        return _model


# ------------------------------------------------- can we use processes?

def _find(module, qualname):
    """Unpickle helper: look a decorated function up by name in a worker process."""
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _defined_without_file(target):
    """True for a function typed into Jupyter or a REPL: worker processes started
    with spawn (Windows, macOS) cannot import it by name."""
    module = sys.modules.get(getattr(target, "__module__", None))
    return (module is not None and module.__name__ == "__main__" and not hasattr(module, "__file__")
            and _start_method() != "fork")


def _process_problem(target, sample_item):
    """None if target(item) can run in a worker process, else the reason why not."""
    if multiprocessing.parent_process() is not None or getattr(_worker, "active", False):
        return "already inside a worker"
    qualname = getattr(target, "__qualname__", "")
    if "<locals>" in qualname or "<lambda>" in qualname:
        return "function is not defined at module top level"
    module = sys.modules.get(getattr(target, "__module__", None))
    if module is None:
        return "function's module cannot be found"
    if _defined_without_file(target):
        if cloudpickle is None or not isinstance(target, AdaptiveExec):
            return "function defined in Jupyter/REPL cannot be sent to worker processes"
        try:                                   # it travels by value: check that this really works
            pickle.loads(pickle.dumps(target))
        except Exception as e:
            return f"function defined in Jupyter/REPL cannot be pickled ({type(e).__name__})"
    else:
        try:
            found = _find(module.__name__, qualname)
        except AttributeError:
            found = None
        if found is not target:
            return "function cannot be found by its name"
    try:
        pickle.dumps(sample_item)
    except Exception:
        return "items cannot be pickled"
    return None


def _pickled_size(obj):
    """Bytes after pickling, or None if obj cannot be pickled."""
    try:
        return len(pickle.dumps(obj))
    except Exception:
        return None


def _rule_on_source(func):
    """Fallback when the probe cannot run: threads if the code waits or calls GIL-free C code."""
    static = ast_features(func)
    return "threading" if static["has_io_call"] or static["has_gil_release_call"] else "sequential"


class _Recorder:
    """Wraps the user function during the probe.

    - keeps the outputs, so if probing fails half-way (an output that cannot be
      pickled) the items already computed are not lost or run twice;
    - keeps exceptions raised in the probe's helper threads, which Python would
      otherwise only print and swallow.
    It hashes/compares equal to the wrapped function, so the cached AST
    features of that function are reused.
    """

    def __init__(self, func):
        self.func = func
        self.__wrapped__ = func                   # inspect.getsource() follows this
        self.caller = threading.get_ident()
        self.outputs = []
        self.errors = []

    def __hash__(self):
        return hash(self.func)

    def __eq__(self, other):
        return other is self.func or (isinstance(other, _Recorder) and other.func is self.func)

    def __call__(self, item):
        in_caller = threading.get_ident() == self.caller
        try:
            out = self.func(item)
        except BaseException as e:
            self.errors.append(e)
            if in_caller:
                raise
            return None
        if in_caller:
            self.outputs.append(out)
        return out


# ----------------------------------------------------------------- the core

def _run_mode(mode, func, target, items, n_workers):
    if not items:
        return []
    if mode == "sequential":
        return [func(item) for item in items]
    pool = _get_pool(mode, n_workers)
    # default chunksize, same as in the data collection
    return pool.map(target if mode == "multiprocessing" else func, items)


def _best_allowed(probabilities, allowed):
    return max(allowed, key=lambda m: probabilities[m])


def run(func, items, n_workers=None, mode=None, cache=True, *, _target=None, _name=None):
    """Compute [func(item) for item in items] in the best mode.

    Returns (results, Decision). `mode` forces one mode (also settable for the
    whole program with the environment variable POLYMORPH_MODE). With `cache`,
    a later call that looks the same (similar item count, first-item time and
    item size) reuses the decision instead of probing again.
    """
    t_start = time.perf_counter()
    target = _target or func                    # what multiprocessing pickles
    if not isinstance(items, (list, tuple)):
        items = list(items)
    n_workers = max(1, int(n_workers or usable_cpu_count()))
    d = Decision(function=_name or getattr(func, "__qualname__", repr(func)),
                 mode="sequential", reason="", n_items=len(items), n_workers=n_workers)

    def finish(results):
        d.total_s = time.perf_counter() - t_start
        log.debug("%s", d)
        return results, d

    forced = mode or os.environ.get("POLYMORPH_MODE")
    if forced:
        if forced not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {forced!r}")
        if forced == "multiprocessing" and items:
            problem = _process_problem(target, items[0])
            if problem:
                raise ValueError(f"cannot use multiprocessing: {problem}")
        d.mode, d.reason = forced, "forced"
        return finish(list(_run_mode(forced, func, target, list(items), n_workers)))

    # ---- 1. cases where measuring is pointless or impossible
    if getattr(_worker, "active", False):
        d.reason = "nested call inside a polymorph_ai worker"
        return finish([func(item) for item in items])
    if len(items) <= PROBE_MIN_ITEMS or n_workers == 1:
        d.reason = "too few items" if n_workers > 1 else "only 1 worker"
        return finish([func(item) for item in items])

    allowed = list(MODES)
    problem = _process_problem(target, items[0])
    if problem:
        allowed.remove("multiprocessing")
        d.notes.append("no multiprocessing: " + problem)

    item_bytes = _pickled_size(items[0])
    if item_bytes is None:
        # The probe measures pickling, so it cannot run. Fall back to the code itself.
        d.mode, d.reason = _rule_on_source(func), "items cannot be pickled -> rule on the source code"
        return finish(_run_mode(d.mode, func, target, list(items), n_workers))

    # ---- 2. time the first item (useful work): is the whole job tiny?
    t0 = time.perf_counter()
    first = [func(items[0])]
    first_s = time.perf_counter() - t0
    if first_s * len(items) < SMALL_JOB_S:
        d.reason, d.probed_items = "tiny job (from the first item's time)", 1
        return finish(first + [func(item) for item in items[1:]])

    # ---- 3. same kind of call as before? reuse that decision, skip the probe
    key = (func, n_workers, round(math.log2(len(items))), round(math.log2(item_bytes + 1) / 2),
           round(math.log2(max(first_s, 1e-7)))) if cache else None
    cached = None
    if key:   # one timing can land just across a bucket edge, so accept the neighbouring buckets too
        cached = next((_decisions[k] for k in (key, key[:-1] + (key[-1] - 1,), key[:-1] + (key[-1] + 1,))
                       if k in _decisions), None)
    if cached and cached[1] > 0 and (cached[0] != "multiprocessing" or "multiprocessing" in allowed):
        cached[1] -= 1
        d.mode, d.reason, d.probed_items = cached[0], "same as an earlier similar call (cached)", 1
        d.overhead_s = time.perf_counter() - t0 - first_s
        return finish(first + _run_mode(d.mode, func, target, list(items[1:]), n_workers))

    # ---- 4. probe the next items (they are really computed and kept)
    ast_features(func)                          # cache under func itself, not the recorder
    rec = _Recorder(func)
    try:
        features, done, _, extra = extract_features(rec, items[1:], n_workers)
    except Exception:
        if rec.errors:                          # the user's function raised: behave like a plain loop
            raise
        # an output could not be pickled: keep what was computed, finish without processes
        done = first + rec.outputs
        d.mode, d.reason = _rule_on_source(func), "results cannot be pickled -> rule on the source code"
        d.probed_items = len(done)
        return finish(done + _run_mode(d.mode, func, target, list(items[len(done):]), n_workers))
    if rec.errors:                              # raised inside a thread-probe thread
        raise rec.errors[0]

    features["n_items"] = len(items)            # the model is about the whole job
    done = first + done
    d.features, d.probed_items = features, len(done)
    rest = list(items[len(done):])
    t_decide = time.perf_counter()

    # ---- 5. decide
    if not rest:
        d.reason = "probe already finished every item"
    elif features["time_per_item"] * len(rest) < SMALL_JOB_S:
        d.reason = "remaining work too small for any pool"
        if key:
            _decisions[key] = ["sequential", CACHE_USES]
    else:
        best, d.probabilities = _get_model().predict(features)
        d.mode, d.reason = _best_allowed(d.probabilities, allowed), "model"
        if d.mode != best:
            d.reason = f"model ({best} not possible)"
        if key:     # cached before the start-up check: a repeated job is worth starting the pool for
            _decisions[key] = [d.mode, CACHE_USES]
        if d.mode == "multiprocessing" and not _pool_is_warm("multiprocessing", n_workers):
            # The model learned on warm pools. If even a perfect speed-up would
            # save less than starting the pool costs, do not start it for this call.
            ideal_saving = features["time_per_item"] * len(rest) * (1 - 1 / min(n_workers, features["cpu_count"]))
            if ideal_saving < COLD_START_S.get(_start_method(), 0.25):
                d.mode = _best_allowed(d.probabilities, ["sequential", "threading"])
                d.reason = "model said multiprocessing, but pool start-up would cost more than it saves"
    d.overhead_s = extra + (time.perf_counter() - t_decide)

    # ---- 6. run the rest
    return finish(done + _run_mode(d.mode, func, target, rest, n_workers))


# ---------------------------------------------------------------- decorator

class AdaptiveExec:
    """The object @adaptive_exec puts in place of your function."""

    def __init__(self, func, n_workers=None, mode=None, cache=True, verbose=False):
        functools.update_wrapper(self, func)
        self.func = func
        self.n_workers = n_workers
        self.mode = mode
        self.cache = cache
        self.verbose = verbose
        self.last_run = None

    def __call__(self, item):
        return self.func(item)

    def map(self, items, n_workers=None, mode=None, cache=None):
        """[self(item) for item in items], run in the mode polymorph_ai picks."""
        results, self.last_run = run(self.func, items, n_workers or self.n_workers, mode or self.mode,
                                     self.cache if cache is None else cache,
                                     _target=self, _name=self.__qualname__)
        if self.verbose:
            print(self.last_run, file=sys.stderr)
        return results

    def __reduce__(self):
        if _defined_without_file(self) and cloudpickle is not None:
            # Written in Jupyter/REPL: there is no file to import it from, so send
            # the code itself. Inside that cloudpickle call, a reference back to
            # this wrapper (e.g. a recursive function calling itself by name) is
            # rebuilt from the function, which cloudpickle has already memoised.
            if getattr(_by_value, "active", False):
                return _rewrap, (self.func, self.n_workers, self.mode, self.cache, self.verbose)
            _by_value.active = True
            try:
                data = cloudpickle.dumps(self)
            finally:
                _by_value.active = False
            return cloudpickle.loads, (data,)
        # Worker processes import the module and look the function up by name,
        # which finds this object again (pickling self.func by name would fail:
        # its name now points at this wrapper).
        return _find, (self.__module__, self.__qualname__)

    def __repr__(self):
        return f"<adaptive_exec {self.__module__}.{self.__qualname__}>"


_by_value = threading.local()


def _rewrap(func, n_workers, mode, cache, verbose):
    return AdaptiveExec(func, n_workers, mode, cache, verbose)


def adaptive_exec(func=None, *, n_workers=None, mode=None, cache=True, verbose=False):
    """Decorator. Use as @adaptive_exec or @adaptive_exec(n_workers=4, verbose=True).

    n_workers: pool size (default: usable CPUs)   mode: force one mode
    cache: reuse decisions for similar calls       verbose: print every decision
    """
    if func is None:
        return lambda f: AdaptiveExec(f, n_workers, mode, cache, verbose)
    return AdaptiveExec(func, n_workers, mode, cache, verbose)
