"""Which team has the ball, from a classifier trained on other matches.

The control test in rules.evidence only speaks when a player is within reach of a slow ball, so
most frames give it nothing to say and possession just carries on. The IDSSE Bundesliga matches
(Bassek et al. 2025, CC BY 4.0) flag the owning team on every frame, so a model can learn it from
the same distances seen over a few seconds before and after, and from how both teams stand. It is
only asked where the rules are silent, and only trusted when it is very sure (poss_thr).

Trained by tools/train_poss.py on matches 1-5 and saved to models/possession.joblib. It ranks the
owning team with AUC 0.96-0.97 on Metrica sample game 3 and Bundesliga 6 and 7. Filling the silent
frames raised spellF1 there by 0.049, 0.057 and 0.098.
"""

import os
import warnings

import numpy as np

from . import config, touch
from .deadmodel import _roll

MODEL = os.path.join(os.path.dirname(config.HERE), "models", "possession.joblib")
_model = None


def features(data, feat, ball):
    """One row per frame. ball is the detector's ball path, the tracking's is already in feat."""
    fps = config.FPS
    d = feat["dist"].astype(float)
    v = touch.team_distance(data, ball)
    seen = (~np.isnan(feat["ball_xy"][:, 0])).astype(float)
    cols = [np.nan_to_num(d[:, 0], nan=40), np.nan_to_num(d[:, 1], nan=40),
            np.nan_to_num(d[:, 0] - d[:, 1], nan=0), np.nan_to_num(feat["ball_speed"], nan=-1), seen,
            np.nan_to_num(v[:, 0], nan=40), np.nan_to_num(v[:, 1], nan=40)]
    with np.errstate(invalid="ignore"):
        near0 = ((d[:, 0] < d[:, 1]) & (d[:, 0] < 2)).astype(float)
        near1 = ((d[:, 1] < d[:, 0]) & (d[:, 1] < 2)).astype(float)
    for w in (fps // 2, fps, 2 * fps, 4 * fps):
        for src in (cols[2], near0, near1):
            cols.append(_roll(src, w))              # the w frames before
            cols.append(_roll(src, w, -w + 1))      # the w frames after
    return np.nan_to_num(np.concatenate([np.stack(cols, axis=1), shape(data, feat)], axis=1), nan=0)


def shape(data, feat):
    """How both teams stand: the side with the ball spreads out and moves towards the other."""
    fps = config.FPS
    xy = data["players_xy"].astype(float)
    ball = feat["ball_xy"].astype(float)
    per = []
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team in (0, 1):
            pts = xy[:, data["team"] == team]
            c, s = np.nanmean(pts, axis=1), np.nanstd(pts, axis=1)
            v = np.full_like(c, np.nan)
            v[fps:] = c[fps:] - c[:-fps]            # centroid move over a second
            per.append((c, s, v))
        (c0, s0, v0), (c1, s1, v1) = per
        to1 = c1 - c0
        to1 /= np.maximum(np.hypot(*to1.T), 1)[:, None]
        adv0, adv1 = (v0 * to1).sum(axis=1), -(v1 * to1).sum(axis=1)
        cols = [s0[:, 0] - s1[:, 0], s0[:, 1] - s1[:, 1], s0[:, 0] * s0[:, 1] - s1[:, 0] * s1[:, 1],
                adv0, adv1, adv0 - adv1, ((ball - c0) * to1).sum(axis=1), -((ball - c1) * to1).sum(axis=1),
                np.hypot(*(c1 - c0).T)]
    cols = [np.nan_to_num(c) for c in cols]
    for w in (2 * fps, 4 * fps, 8 * fps):
        for src in cols[:6]:
            cols.append(_roll(src, w))
            cols.append(_roll(src, w, -w + 1))
    return np.stack(cols, axis=1)


def fill(ev, data, feat, p):
    """ev with the frames it leaves at -1 given to the team the model is at least poss_thr sure of."""
    global _model
    if _model is None:
        import joblib
        _model = joblib.load(MODEL)
    q = _model.predict_proba(features(data, feat, touch.video_ball()))[:, 1]
    out = ev.copy()
    silent = out < 0
    out[silent & (q > p["poss_thr"])] = 1
    out[silent & (q < 1 - p["poss_thr"])] = 0
    return out
