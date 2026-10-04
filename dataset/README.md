# Dataset

## `polymorph_dataset.csv` — 13,603 jobs

Each row is one job: a function applied to a list of items. The job was timed in all three modes on one machine.

| column | meaning |
|---|---|
| `os` | Windows or Linux |
| `family` | kind of work (pure-Python CPU, I/O, numpy, hashing, ...) |
| 13 feature columns | the inputs of the model (see the main README): `n_items`, `input_bytes`, `cpu_count`, `n_workers`, `max_loop_depth`, `arith_op_count`, `has_io_call`, `has_gil_release_call`, `time_per_item`, `cpu_ratio`, `time_cv`, `pickle_time_ratio`, `thread_probe_speedup` |
| `t_sequential`, `t_threading`, `t_multiprocessing` | wall-clock seconds of the whole job in each mode (median of up to 3 runs) |
| `label` | the fastest mode; if another mode is within 5% of it, the cheaper one wins (sequential < threading < multiprocessing) |

## `realcode_results.csv` — 92 jobs of ordinary code

These are 23 functions from the pyperformance benchmark suite, the standard library and numpy, with 4 job sizes each. None of them were in the training data. They ran on a Linux machine with 4 vCPUs.

| column | meaning |
|---|---|
| `function`, `kind` | the function and where it comes from (`pyperformance`, `stdlib`, `numpy`) |
| `job_size_s`, `n_items` | target length of the job as a plain loop, and the number of items |
| `t_sequential`, `t_threading`, `t_multiprocessing`, `label` | as above |
| `t_adaptive`, `adaptive_mode` | the whole `@adaptive_exec` call on a first call (probe + model + run, no cache) and the mode it picked |
