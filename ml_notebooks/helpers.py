"""Shared helpers for the ML notebooks: paths, the label rule, the time-lost metric,
and one chart style so every figure in the project looks the same.

Tie rule (decided 2026-10-04): a mode within 5% of the fastest counts as a tie,
and the tie goes to the cheaper mode (sequential < threading < multiprocessing).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # repo root (holds polymorph_ai/)
if ROOT not in sys.path:                      # so notebooks can import polymorph_ai
    sys.path.insert(0, ROOT)

from polymorph_ai.features import FEATURE_NAMES  # noqa: E402

DATASET = os.path.join(HERE, "data", "dataset.csv")   # merged raw rows
CLEAN = os.path.join(HERE, "data", "clean.csv")       # output of 01_data_cleaning
FIGURES = os.path.join(HERE, "figures")

MODES = ["sequential", "threading", "multiprocessing"]   # cheapest first
TIME_COLS = ["t_" + m for m in MODES]
TIE = 1.05

# Features whose values span several orders of magnitude -> log before an MLP.
LOG_FEATURES = ["n_items", "input_bytes", "time_per_item", "time_cv",
                "pickle_time_ratio", "thread_probe_speedup"]


def tie_label(times):
    """times: (n, 3) array in MODES order -> array of mode names."""
    best = times.min(axis=1, keepdims=True)
    first_close = np.argmax(times <= best * TIE, axis=1)   # cheapest mode within 5%
    return np.array(MODES)[first_close]


def time_lost(times, chosen):
    """Extra total time (%) of the chosen modes vs always picking the fastest mode.

    times: (n, 3) array, chosen: array of mode names.
    """
    idx = np.array([MODES.index(m) for m in chosen])
    picked = times[np.arange(len(times)), idx]
    best = times.min(axis=1)
    return (picked.sum() / best.sum() - 1) * 100


def log_transform(X):
    """log10 of the heavy-tailed columns (stateless, safe to apply before splitting)."""
    X = X.copy()
    for c in LOG_FEATURES:
        if c in X:
            X[c] = np.log10(X[c].clip(lower=1e-9))
    return X


def load_clean():
    return pd.read_csv(CLEAN)


# ---- MLflow (local, no server needed) ------------------------------------------
# View the runs:  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db   (from this folder)
MLFLOW_DIR = os.path.join(HERE, "mlflow")
MLFLOW_URI = "sqlite:///" + os.path.join(MLFLOW_DIR, "mlflow.db").replace("\\", "/")
EXPERIMENT = "conductor-mode-selection"


def setup_mlflow():
    import mlflow
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        artifacts = "file:///" + os.path.join(MLFLOW_DIR, "artifacts").replace("\\", "/")
        mlflow.create_experiment(EXPERIMENT, artifact_location=artifacts)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


# ---- chart style -------------------------------------------------------------
# Palette checked with the dataviz validator (colorblind-safe for 3 classes).
COLORS = {"sequential": "#2a78d6", "threading": "#eb6834", "multiprocessing": "#1baf7a"}
OS_COLORS = {"Linux": "#2a78d6", "Windows": "#eb6834"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def setup_plots():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "figure.dpi": 110,
    })
    os.makedirs(FIGURES, exist_ok=True)


def save(fig, name):
    """Save a figure as figures/<name>.png."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, name + ".png"), dpi=150)


def stacked_share(ax, df, by, order=None, label_col="y"):
    """100% stacked horizontal bars: share of each mode per group, labels inside."""
    import matplotlib.ticker as mtick
    share = pd.crosstab(df[by], df[label_col], normalize="index").reindex(columns=MODES, fill_value=0)
    counts = df[by].value_counts()
    share = share.loc[order] if order is not None else share.loc[share["multiprocessing"].sort_values().index]
    left = np.zeros(len(share))
    for m in MODES:
        ax.barh(range(len(share)), share[m], left=left, color=COLORS[m], label=m,
                edgecolor=SURFACE, linewidth=2, height=0.72)
        for i, v in enumerate(share[m]):
            if v >= 0.08:
                ax.text(left[i] + v / 2, i, f"{v:.0%}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        left += share[m].to_numpy()
    ax.set_yticks(range(len(share)), [f"{g}  (n={counts[g]:,})" for g in share.index])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False, fontsize=9)
    return share
