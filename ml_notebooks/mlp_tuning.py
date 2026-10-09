"""Keras MLP for mode selection + nested Optuna tuning.

The notebook imports KerasMLP from here. Run the file to tune (resumable, results in mlflow/optuna.db):

    cd ml_notebooks
    python mlp_tuning.py --trials 30

For every outer fold (StratifiedGroupKFold by template, same folds as the notebook) Optuna searches
hyperparameters using only that fold's TRAIN rows: the model holds out 20% of the train templates,
early-stops on them and is scored (time lost) on them. The outer test fold is never touched here.
"""

import argparse
import os
import time

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from helpers import FEATURE_NAMES, MLFLOW_DIR, MODES, TIME_COLS, load_clean, log_transform, time_lost

df = load_clean()
X, y, groups = df[FEATURE_NAMES], df["y"], df["template_id"]
TIMES = df[TIME_COLS].to_numpy()
FOLDS = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups))

OPTUNA_DB = "sqlite:///" + os.path.join(MLFLOW_DIR, "optuna.db").replace("\\", "/")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class KerasMLP(BaseEstimator, ClassifierMixin):
    """log -> StandardScaler -> Keras MLP, early stopping on 20% of the train templates."""

    def __init__(self, units=64, layers=2, dropout=0.2, learning_rate=1e-3, l2=0.0,
                 batch_size=256, max_epochs=300, patience=20, seed=0):
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.l2 = l2
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

    def _build(self):
        reg = keras.regularizers.l2(self.l2) if self.l2 > 0 else None
        n_in = getattr(self, "n_in_", len(FEATURE_NAMES))      # fewer columns in the ablation study
        model = keras.Sequential([keras.Input(shape=(n_in,))])
        for i in range(self.layers):
            model.add(keras.layers.Dense(max(self.units // (2 ** i), 8), activation="relu",
                                         kernel_regularizer=reg))       # 64 -> 32 -> 16 ...
            model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(len(MODES), activation="softmax"))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y, extra_callbacks=()):
        keras.utils.set_random_seed(self.seed)
        self.classes_ = np.array(MODES)
        self.n_in_ = X.shape[1]
        y_int = pd.Series(y).map({m: i for i, m in enumerate(MODES)}).to_numpy().astype("int32")

        # validation = 20% of the TRAIN templates (never the test fold)
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        fit_idx, val_idx = next(split.split(X, y_int, groups.loc[X.index]))
        self.val_index_ = X.index[val_idx]

        self.scaler_ = StandardScaler().fit(log_transform(X.iloc[fit_idx]))
        Z = self.scaler_.transform(log_transform(X))
        weights = compute_class_weight("balanced", classes=np.arange(len(MODES)), y=y_int[fit_idx])

        self.model_ = self._build()
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                             restore_best_weights=True)
        self.history_ = self.model_.fit(
            Z[fit_idx], y_int[fit_idx], validation_data=(Z[val_idx], y_int[val_idx]),
            epochs=self.max_epochs, batch_size=self.batch_size,
            class_weight=dict(enumerate(weights)), callbacks=[stop, *extra_callbacks], verbose=0).history
        return self

    def predict(self, X):
        Z = self.scaler_.transform(log_transform(X))
        return self.classes_[self.model_.predict(Z, verbose=0, batch_size=4096).argmax(axis=1)]

    def n_weights(self):
        return int(sum(np.prod(w.shape) for w in self.model_.get_weights()))


# ---- Optuna ----------------------------------------------------------------------

def suggest(trial):
    """Search space (6 hyperparameters)."""
    return {
        "layers": trial.suggest_int("layers", 1, 4),
        "units": trial.suggest_int("units", 16, 256, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True),
    }


class ReportToOptuna(keras.callbacks.Callback):
    """Send val_loss to Optuna every epoch so the pruner can stop bad trials early."""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial
        self.pruned = False

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(logs["val_loss"], step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            self.model.stop_training = True


def make_objective(fold):
    tr, _ = FOLDS[fold]
    Xtr, ytr = X.iloc[tr], y.iloc[tr]

    def objective(trial):
        params = suggest(trial)
        reporter = ReportToOptuna(trial)
        t0 = time.time()
        m = KerasMLP(**params).fit(Xtr, ytr, extra_callbacks=[reporter])
        h = m.history_
        trial.set_user_attr("loss", [float(v) for v in h["loss"]])
        trial.set_user_attr("val_loss", [float(v) for v in h["val_loss"]])
        trial.set_user_attr("epochs", len(h["loss"]))
        trial.set_user_attr("n_weights", m.n_weights())
        trial.set_user_attr("seconds", round(time.time() - t0, 1))
        if reporter.pruned:
            raise optuna.TrialPruned()
        val = m.val_index_
        pred = m.predict(X.loc[val])
        trial.set_user_attr("val_accuracy", float((pred == y.loc[val]).mean()))
        return time_lost(df.loc[val, TIME_COLS].to_numpy(), pred)       # minimise time lost on validation

    return objective


def study_for(fold):
    return optuna.create_study(
        study_name=f"mlp_fold{fold + 1}", storage=OPTUNA_DB, load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=fold),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=15))


def best_params(fold):
    return optuna.load_study(study_name=f"mlp_fold{fold + 1}", storage=OPTUNA_DB).best_params


def log_study_to_mlflow(study, fold):
    """One parent run per fold, one nested run per trial (params, score, loss per epoch)."""
    from helpers import setup_mlflow
    mlflow = setup_mlflow()
    with mlflow.start_run(run_name=f"optuna MLP fold {fold + 1}"):
        mlflow.log_params({"n_trials": len(study.trials), **{f"best_{k}": v for k, v in study.best_params.items()}})
        mlflow.log_metric("best_val_time_lost_pct", study.best_value)
        for t in study.trials:
            with mlflow.start_run(run_name=f"fold{fold + 1}-trial{t.number}", nested=True):
                mlflow.log_params({**t.params, "state": t.state.name})
                if t.value is not None:
                    mlflow.log_metric("val_time_lost_pct", t.value)
                for k in ("val_accuracy", "n_weights", "seconds", "epochs"):
                    if k in t.user_attrs:
                        mlflow.log_metric(k, t.user_attrs[k])
                for e, (l, vl) in enumerate(zip(t.user_attrs.get("loss", []), t.user_attrs.get("val_loss", []))):
                    mlflow.log_metrics({"loss": l, "val_loss": vl}, step=e)


def export_csv():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frames = []
    for fold in range(len(FOLDS)):
        try:
            s = optuna.load_study(study_name=f"mlp_fold{fold + 1}", storage=OPTUNA_DB)
        except KeyError:
            continue
        d = s.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
        d = d.drop(columns=[c for c in d.columns if c in ("user_attrs_loss", "user_attrs_val_loss")])
        frames.append(d.assign(fold=fold + 1))
    out = os.path.join(RESULTS_DIR, "optuna_trials.csv")
    pd.concat(frames).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30, help="trials per outer fold")
    ap.add_argument("--folds", default="1,2,3,4,5")
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for fold in [int(f) - 1 for f in args.folds.split(",")]:
        study = study_for(fold)
        todo = args.trials - len([t for t in study.trials if t.state.is_finished()])
        t0 = time.time()
        if todo > 0:
            study.optimize(make_objective(fold), n_trials=todo)
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
        print(f"fold {fold + 1}: best val time lost {study.best_value:.2f}% | {states} | "
              f"{time.time() - t0:.0f} s | {study.best_params}", flush=True)
        log_study_to_mlflow(study, fold)
    export_csv()
