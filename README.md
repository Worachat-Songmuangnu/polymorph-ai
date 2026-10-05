# polymorph-ai

`@adaptive_exec` runs a function over a list of items **sequentially, with threads, or with processes**, and a small trained neural network picks whichever mode should be fastest for that job on that machine.

```python
from polymorph_ai import adaptive_exec

@adaptive_exec
def process(item):
    ...

if __name__ == "__main__":            # required on Windows/macOS for processes
    results = process.map(items)      # same as [process(x) for x in items], in order
    print(process.last_run)
    # [polymorph_ai] process: 5,000 items -> multiprocessing p=0.97 (model) | overhead 0.7 ms | total 1.84 s
```

`process(x)` still works as a normal function call. Only `.map()` adds the automatic part.

## Install

```bash
pip install polymorph-ai
```

Dependencies: numpy and cloudpickle. Tested on Linux with Python 3.10, 3.12 and 3.14, and on Windows with Python 3.14.

Works in Jupyter too: a function written in a notebook cell is sent to the worker processes by value (with cloudpickle), so it can still use multiprocessing on Windows and macOS.

## How it decides

Every `.map(items)` call goes through these steps:

1. **Safety checks.** Empty or very short lists, one worker, or a call nested inside another polymorph_ai worker all just run as a plain loop. If the function or the items cannot be sent to another process (a lambda, a function defined inside another function, unpicklable items), multiprocessing is ruled out.
2. **First item.** The first item runs and is timed. If `time × number of items` is under 10 ms, the job is too small for any pool to pay off, so the rest runs as a plain loop.
3. **Cache.** A call that looks like an earlier one (similar number of items, first-item time, and item size) reuses that decision and skips the probe.
4. **Probe.** The next items run for about 50 ms, first one by one and then on 2 threads, to measure the 13 features: call site, static code analysis (AST), runtime probe, and thread probe. The probe uses the same code (`polymorph_ai/features.py`) that collected the training data. **Probed items are real work: their results are kept and no item ever runs twice.**
5. **Model.** An MLP (13 → 225 ReLU → 3 softmax, 3,828 weights) turns the features into probabilities. It is trained in Keras and runs here in numpy, at about 30 µs per decision.
6. **Run.** The remaining items run in the chosen mode on a pool that stays alive for the next call.

## Options

```python
@adaptive_exec(n_workers=4, verbose=True, cache=True)   # verbose prints every decision
def process(item): ...

process.map(items, mode="threading")      # force one mode for this call
# POLYMORPH_MODE=sequential python app.py  # force one mode for the whole program (A/B testing, debugging)

from polymorph_ai import run, warm_up
results, decision = run(some_function, items)   # for functions you cannot decorate
warm_up()                                      # start the pools early (e.g. at server start-up)
```

`process.last_run` (a `Decision`) records the chosen mode, why it was chosen, the model's probabilities, the 13 features, how many items the probe computed, and the overhead.

## Results

**Model** (13,603 jobs collected on 16 machine settings, see [`dataset/`](https://github.com/Worachat-Songmuangnu/polymorph-ai/tree/main/dataset), GroupKFold by workload template, so every test job comes from code the model never trained on). Time lost compared with always picking the fastest mode:

- Always multiprocessing: +22.1%
- if-else rules: +13.0%
- Random Forest: +5.8%
- **MLP: +4.3%**

**Library** ([`examples/demo.py`](https://github.com/Worachat-Songmuangnu/polymorph-ai/blob/main/examples/demo.py): 8 everyday jobs that are not in the training data, 8 CPUs). Total time compared with perfect picks:

| | Windows | Linux (WSL) |
|---|---|---|
| always sequential | +349% | +426% |
| always threading | +117% | +138% |
| always multiprocessing | +19% | +30% |
| polymorph_ai, first call | +52% | +66% |
| polymorph_ai, repeated call | **+13%** | **+24%** |

**When does it pay off?** The first call on a job pays about 0.2 s for the probe (`python examples/demo.py --overhead`).

- Compared with a plain loop, polymorph_ai is already faster on jobs from about 0.3–0.6 s.
- Compared with a perfect choice of mode, the first call is within 15% once the job takes 5 s or more.
- Repeated calls skip the probe, so they cost almost nothing.

## Limitations

- The decorated function takes **one item** and is applied to every item (`map`). polymorph_ai cannot parallelise code that does not have this shape.
- On Windows and macOS, the script that uses processes needs the `if __name__ == "__main__":` guard (a Python rule for processes, not specific to polymorph_ai).
- The first item and the probe items run in the calling thread, so a job with a few very long items loses one item's worth of parallel time.
- The model was trained on synthetic workloads from 13 families. Code that behaves unlike all of them can still be mispredicted. In the demo, sorting 20k-number lists stays sequential when processes would be twice as fast.
- asyncio is not supported: a decorator cannot turn ordinary code into `async def` code.
- In Jupyter on Windows, call `shutdown()` before you close or restart the kernel. If the kernel is stopped straight after the cell that started the process pool, the worker processes can be left running.

## Tests

```bash
python -m pytest tests -v    # 28 tests, pass on Windows and Linux
```

## Project layout

```
polymorph_ai/   the library: decorator.py, features.py, model.py + model.npz (the trained network)
tests/          pytest suite
examples/       demo.py (benchmark vs fixed modes) and its results/
dataset/        the training dataset (13,603 jobs) and the real-code results (92 jobs)
```
