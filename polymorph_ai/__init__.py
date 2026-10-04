"""polymorph_ai: pick sequential / threading / multiprocessing automatically.

    from polymorph_ai import adaptive_exec

    @adaptive_exec
    def process(item):
        ...

    results = process.map(items)
    print(process.last_run)
"""

from .decorator import Decision, adaptive_exec, run, shutdown, warm_up
from .features import FEATURE_NAMES, extract_features

__all__ = ["adaptive_exec", "run", "warm_up", "shutdown", "Decision",
           "FEATURE_NAMES", "extract_features"]
