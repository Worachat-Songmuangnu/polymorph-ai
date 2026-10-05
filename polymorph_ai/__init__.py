"""polymorph_ai: pick sequential / threading / multiprocessing automatically.

    from polymorph_ai import adaptive_exec

    @adaptive_exec
    def process(item):
        ...

    results = process.map(items)
    print(process.last_run)
"""

from importlib.metadata import PackageNotFoundError, version

from .decorator import Decision, adaptive_exec, run, shutdown, warm_up
from .features import FEATURE_NAMES, extract_features

try:
    __version__ = version("polymorph-ai")
except PackageNotFoundError:     # running from a source checkout that is not installed
    __version__ = "unknown"

__all__ = ["adaptive_exec", "run", "warm_up", "shutdown", "Decision",
           "FEATURE_NAMES", "extract_features"]
