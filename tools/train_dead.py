"""Train the dead-ball classifier (mtah/deadmodel.py) on IDSSE Bundesliga matches 1-5.

    python tools/train_dead.py

Needs the IDSSE data (huggingface.co/datasets/pysport/idsse-data, CC BY 4.0) in ext-data/idsse/
next to this repository. Writes models/dead_ball.joblib. Labels are the tracking provider's own
per-frame ball status; nothing from BAR-ATH is used. Deterministic: same data, same model.
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import tune_external as te  # noqa: E402
from mtah import deadmodel  # noqa: E402

if __name__ == "__main__":
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    warnings.simplefilter("ignore", RuntimeWarning)
    Xs, ys = [], []
    for name in ("d1", "d2", "d3", "d4", "d5"):
        g = te.make(name, 0.11)
        g.score({"dead_model": 0})      # builds g.feat with this game's pitch and halves
        X = deadmodel.features(g.feat)
        idx = np.flatnonzero(g.truth < 3)[::4]
        Xs.append(X[idx])
        ys.append(g.truth[idx] == 2)
        print(name, len(idx), "frames", flush=True)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=6,
                                           min_samples_leaf=50, random_state=0)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    os.makedirs(os.path.dirname(deadmodel.MODEL), exist_ok=True)
    joblib.dump(model, deadmodel.MODEL)
    print("->", os.path.relpath(deadmodel.MODEL))
