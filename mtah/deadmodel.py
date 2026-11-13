"""Dead ball from a classifier trained on other matches.

The rules call the ball dead when it leaves the pitch or the fastest players slow down together,
with thresholds a person had to pick. The IDSSE Bundesliga matches (Bassek et al. 2025, CC BY 4.0)
carry the tracking provider's own ball status on every frame, so a model can learn the same thing
from the players' pace and the ball path, over a few seconds before and after each frame.

Trained by tools/train_dead.py on matches 1-5 and saved to models/dead_ball.joblib. On the held-out
matches 6 and 7 it separates dead from live frames with AUC 0.99, on Metrica sample game 3 with
0.81. Used in place of the rule mask it raised spellF1 there by 0.077 and 0.046, and left Metrica
game 3 unchanged (0.575).
"""

import os

import numpy as np

from . import config

MODEL = os.path.join(os.path.dirname(config.HERE), "models", "dead_ball.joblib")
_model = None


def _roll(a, w, shift=0):
    a = np.nan_to_num(np.asarray(a, dtype=float))
    c = np.concatenate(([0], np.cumsum(a)))
    out = np.full(len(a), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return np.roll(out, shift)


def features(feat):
    """One row per frame, built only from what rules.features() already has."""
    fps = config.FPS
    pf, pm, bs = feat["pace_fast"], feat["pace"], feat["ball_speed"]
    seen = (~np.isnan(feat["ball_xy"][:, 0])).astype(float)
    x, y = feat["ball_xy"][:, 0], feat["ball_xy"][:, 1]
    with np.errstate(invalid="ignore"):
        out_by = np.nan_to_num(np.maximum.reduce([-x, x - config.PITCH_LENGTH, -y, y - config.PITCH_WIDTH]), nan=-5.0)
        near = np.nanmin(feat["dist"], axis=1)
    cols = [pf, pm, np.nan_to_num(bs, nan=-1), seen, out_by, np.nan_to_num(near, nan=30)]
    for w in (fps, 2 * fps, 4 * fps):
        for src in (pf, pm, seen, np.nan_to_num(bs)):
            cols.append(_roll(src, w))              # the w frames before
            cols.append(_roll(src, w, -w + 1))      # the w frames after
    return np.stack(cols, axis=1)


def mask(feat, p):
    """Frames the model calls dead: its score smoothed over a second, above dead_thr, in runs of
    at least dead_min seconds (a shorter one is a pause in play)."""
    global _model
    if _model is None:
        import joblib
        _model = joblib.load(MODEL)
    fps = config.FPS
    score = _roll(_model.predict_proba(features(feat))[:, 1], fps, -fps // 2)
    dead = np.nan_to_num(score) > p["dead_thr"]
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s < p["dead_min"] * fps:
            dead[s:e] = False
    return dead
