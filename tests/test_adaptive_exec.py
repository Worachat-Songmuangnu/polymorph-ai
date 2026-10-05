"""Tests for the polymorph_ai library.  Run from the project root:  python -m pytest tests -v

Two kinds of tests:
  - correctness: whatever mode is picked, .map() must give exactly
    [f(x) for x in items], in order, run every item once, and raise the user's errors;
  - decisions: obvious workloads must get the obvious mode.
"""

import hashlib
import math
import os
import pickle
import threading
import time

import pytest

from polymorph_ai import adaptive_exec, run
from polymorph_ai.features import usable_cpu_count
from polymorph_ai.model import ModeModel

CPUS = usable_cpu_count()


# ---- workload functions (module level, so worker processes can import them)

@adaptive_exec
def square(x):
    return x * x


@adaptive_exec
def cpu_heavy(n):
    """Pure Python arithmetic: holds the GIL -> processes should win."""
    total = 0
    for i in range(n):
        total += (i * i) % 7
    return total


@adaptive_exec
def waits(ms):
    """Sleeps like a network call: releases the GIL -> threads should win."""
    time.sleep(ms / 1000)
    return ms


@adaptive_exec
def hashes(block):
    """hashlib releases the GIL on big buffers."""
    for _ in range(20):
        block = hashlib.sha256(block * 64).digest()
    return block


calls = []
calls_lock = threading.Lock()


@adaptive_exec
def counted(x):
    time.sleep(0.002)
    with calls_lock:
        calls.append(x)
    return x + 1


@adaptive_exec
def fails_on_13(x):
    time.sleep(0.001)
    if x == 13:
        raise KeyError("item 13")
    return x


@adaptive_exec
def returns_lock(x):
    time.sleep(0.001)
    with calls_lock:
        calls.append(x)
    return threading.Lock()


@adaptive_exec
def outer(n):
    return sum(square.map(range(n)))


# ---- correctness ------------------------------------------------------------

@pytest.mark.parametrize("mode", ["sequential", "threading", "multiprocessing", None])
def test_results_match_plain_loop(mode):
    items = list(range(500))
    assert square.map(items, mode=mode) == [x * x for x in items]


def test_auto_mode_results_correct_for_every_kind_of_work():
    for f, items in [(cpu_heavy, [20_000] * 64), (waits, [5] * 80), (hashes, [bytes([i]) * 256 for i in range(64)])]:
        assert f.map(items) == [f.func(x) for x in items], f.last_run


def test_decorated_function_still_works_as_a_normal_function():
    assert square(7) == 49
    assert square.__name__ == "square" and square.__wrapped__(3) == 9


def test_wrapper_pickles_by_name():
    assert pickle.loads(pickle.dumps(square)) is square


def test_empty_one_and_generator_inputs():
    assert square.map([]) == []
    assert square.map([3]) == [9]
    assert square.map(x for x in range(10)) == [x * x for x in range(10)]
    assert square.map(tuple(range(10))) == [x * x for x in range(10)]


def test_every_item_runs_exactly_once():
    calls.clear()
    items = list(range(300))
    assert counted.map(items) == [x + 1 for x in items]
    if counted.last_run.mode != "multiprocessing":     # calls made in child processes are not visible here
        assert sorted(calls) == items


@pytest.mark.parametrize("bad_index", [0, 5, 40, 299])   # in the probe, the thread probe, and the rest
def test_user_errors_are_raised(bad_index):
    items = [0] * 300
    items[bad_index] = 13
    with pytest.raises(KeyError, match="item 13"):
        fails_on_13.map(items)


def test_lambda_and_closure_never_use_processes():
    def local(x):
        total = 0
        for i in range(30_000):
            total += i % 3
        return total + x
    for f in (local, lambda x: sum(i % 3 for i in range(30_000)) + x):
        results, d = run(f, list(range(64)))
        assert results == [f(x) for x in range(64)]
        assert d.mode != "multiprocessing"
        assert any("module top level" in n for n in d.notes), d


def test_unpicklable_items_fall_back_without_processes():
    items = [(i, threading.Lock()) for i in range(50)]
    results, d = run(lambda it: it[0] * 2, items)
    assert results == [i * 2 for i in range(50)]
    assert d.mode != "multiprocessing" and "cannot be pickled" in d.reason


def test_unpicklable_results_keep_probed_work():
    calls.clear()
    items = list(range(200))
    results = returns_lock.map(items)
    assert len(results) == 200
    assert sorted(calls) == items                       # nothing ran twice
    assert "cannot be pickled" in returns_lock.last_run.reason


def test_nested_map_does_not_deadlock():
    assert outer.map([100] * 40) == [sum(x * x for x in range(100))] * 40


def test_forced_mode_from_environment(monkeypatch):
    monkeypatch.setenv("POLYMORPH_MODE", "threading")
    square.map(range(100))
    assert square.last_run.mode == "threading" and square.last_run.reason == "forced"


def test_forced_multiprocessing_on_lambda_gives_clear_error():
    with pytest.raises(ValueError, match="multiprocessing"):
        run(lambda x: x, [1, 2, 3], mode="multiprocessing")


def test_start_method_is_not_locked():
    import multiprocessing
    square.map(range(1000), mode="multiprocessing")
    assert multiprocessing.get_start_method(allow_none=True) is None


# ---- decisions --------------------------------------------------------------

def test_tiny_job_skips_the_probe():
    t0 = time.perf_counter()
    square.map(range(1000))
    d = square.last_run
    assert d.mode == "sequential" and d.reason.startswith("tiny job"), d
    assert time.perf_counter() - t0 < 0.005


def test_similar_call_reuses_decision():
    waits.map([10] * 200, cache=False)
    first = waits.last_run
    waits.map([10] * 210)                 # stored
    waits.map([10] * 190)                 # same size bucket -> no probe
    d = waits.last_run
    assert d.reason.endswith("(cached)") and d.mode == first.mode, d
    assert d.overhead_s < 0.002, d


def test_cache_off_always_probes():
    for _ in range(2):
        waits.map([10] * 200, cache=False)
        assert waits.last_run.reason == "model", waits.last_run


@pytest.mark.skipif(CPUS < 2, reason="needs 2+ CPUs")
def test_big_pure_python_job_uses_processes():
    cpu_heavy.map([200_000] * (8 * CPUS), cache=False)
    assert cpu_heavy.last_run.mode == "multiprocessing", cpu_heavy.last_run


def test_waiting_job_uses_threads():
    waits.map([10] * 200, cache=False)
    assert waits.last_run.mode == "threading", waits.last_run


def test_model_file_loads_and_gives_probabilities():
    m = ModeModel()
    assert len(m.feature_names) == 13
    mode, p = m.predict({f: 1.0 for f in m.feature_names})
    assert mode in p and math.isclose(sum(p.values()), 1.0, rel_tol=1e-6)


def test_decision_explains_itself():
    waits.map([10] * 200, cache=False)
    text = str(waits.last_run)
    assert "waits" in text and waits.last_run.mode in text and "overhead" in text
    assert waits.last_run.features["n_items"] == 200


JUPYTER_STYLE = """
from polymorph_ai import adaptive_exec, shutdown

@adaptive_exec
def fib(n):                      # recursive: its body refers to the decorated name
    return n if n < 2 else fib(n - 1) + fib(n - 2)

out = fib.map([18] * 16, mode="multiprocessing")
assert out == [fib(18)] * 16
auto = fib.map([22] * 48, cache=False)
assert not any("no multiprocessing" in note for note in fib.last_run.notes), fib.last_run.notes
shutdown()
print("OK")
"""


def test_function_without_a_file_can_use_processes():
    # python -c has no __main__.__file__, exactly like a Jupyter cell: the function
    # must travel to the worker processes by value (cloudpickle).
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run([sys.executable, "-c", JUPYTER_STYLE], cwd=root, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0 and "OK" in done.stdout, done.stderr
