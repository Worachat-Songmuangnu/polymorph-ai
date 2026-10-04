"""The trained mode-selection MLP, run with plain numpy.

The weights come from notebooks/04_export_model.ipynb (Keras, trained on the
collected dataset). Re-implementing the forward pass here keeps the library's
only dependency at numpy: no Keras/JAX import (seconds) at run time, and one
decision takes microseconds.

Steps, identical to training:
  log10 of the heavy-tailed features -> standardise -> Dense+ReLU ... -> Dense -> softmax
"""

import json
import os

import numpy as np

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.npz")


class ModeModel:
    def __init__(self, path=DEFAULT_PATH):
        data = np.load(path)
        self.feature_names = [str(f) for f in data["feature_names"]]
        self.modes = [str(m) for m in data["modes"]]
        self.meta = json.loads(str(data["meta"]))
        log_features = {str(f) for f in data["log_features"]}
        self._is_log = np.array([f in log_features for f in self.feature_names])
        self._mean = data["scaler_mean"]
        self._scale = data["scaler_scale"]
        n = int(data["n_layers"])
        self._layers = [(data[f"w{2 * i}"], data[f"w{2 * i + 1}"]) for i in range(n)]

    def predict_proba(self, features):
        """features: dict with the 13 feature names -> probabilities in self.modes order."""
        x = np.array([float(features[f]) for f in self.feature_names])
        x[self._is_log] = np.log10(np.maximum(x[self._is_log], 1e-9))
        h = (x - self._mean) / self._scale
        for i, (w, b) in enumerate(self._layers):
            h = h @ w + b
            if i < len(self._layers) - 1:
                h = np.maximum(h, 0.0)                     # ReLU (dropout is off at inference)
        e = np.exp(h - h.max())
        return e / e.sum()                                 # softmax

    def predict(self, features):
        """Most likely mode and the probability of every mode (dict)."""
        p = self.predict_proba(features)
        return self.modes[int(p.argmax())], dict(zip(self.modes, p.tolist()))
