"""Ball positions from the video detector (tools/detect_ball.py), on the pitch in metres.

The homography maps normalised pitch coordinates to pixels, so its inverse takes a detected box
back to the grass. That is exact for a ball on the ground and wrong for one in the air, the same
flaw the tracking has, so the box size is kept: a lofted ball near the camera looks bigger.
"""

import glob
import os

import numpy as np
import pandas as pd

from . import config

DET_DIR = os.path.join(config.CACHE_DIR, "ball_det")
MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]
_hom = None


def homographies():
    """(N_FRAMES, 3, 3) pitch-to-pixel matrices, cached as .npy next to the other caches."""
    global _hom
    if _hom is None:
        cache = os.path.join(config.CACHE_DIR, "homography.npy")
        if os.path.exists(cache):
            _hom = np.load(cache)
        else:
            table = pd.read_csv(config.asset("homography"))
            _hom = np.full((config.N_FRAMES, 3, 3), np.nan)
            _hom[table["frame"].values] = table[MATRIX_COLUMNS].values.reshape(-1, 3, 3)
            np.save(cache, _hom)
    return _hom


def to_pitch(frames, px, py):
    """Pixel points on given frames -> metres on the pitch."""
    inv = np.linalg.inv(homographies()[frames])
    pts = np.stack((px, py, np.ones_like(px)), axis=1)[:, :, None]
    out = (inv @ pts)[:, :, 0]
    return np.stack((out[:, 0] / out[:, 2] * config.PITCH_LENGTH,
                     out[:, 1] / out[:, 2] * config.PITCH_WIDTH), axis=1)


def load():
    """All detections: dict of arrays frame, x, y (m), w, h (px), conf, one row per box."""
    parts = [np.load(p)["det"] for p in sorted(glob.glob(os.path.join(DET_DIR, "*.npz")))]
    det = np.concatenate(parts) if parts else np.zeros((0, 6), np.float32)
    frames = det[:, 0].astype(int)
    xy = to_pitch(frames, det[:, 1].astype(float), det[:, 2].astype(float))
    return {"frame": frames, "x": xy[:, 0], "y": xy[:, 1], "w": det[:, 3], "h": det[:, 4],
            "conf": det[:, 5], "px": det[:, 1], "py": det[:, 2]}


def covered():
    """Boolean mask of frames the detector has been run on, found or not."""
    mask = np.zeros(config.N_FRAMES, dtype=bool)
    for p in glob.glob(os.path.join(DET_DIR, "*.npz")):
        z = np.load(p)
        mask[int(z["lo"]):int(z["hi"]) + 1] = True
    return mask


def best_box(det, conf, margin=2.0):
    """(N_FRAMES, 2) ground point of the most confident box per frame on or near the pitch, and
    its confidence. The bottom of the box is where a ball on the grass touches it."""
    xy = np.full((config.N_FRAMES, 2), np.nan)
    c = np.zeros(config.N_FRAMES)
    ground = to_pitch(det["frame"], det["px"].astype(float), (det["py"] + det["h"] / 2).astype(float))
    inside = ((ground[:, 0] > -margin) & (ground[:, 0] < config.PITCH_LENGTH + margin) &
              (ground[:, 1] > -margin) & (ground[:, 1] < config.PITCH_WIDTH + margin))
    keep = (det["conf"] >= conf) & inside
    order = np.argsort(det["conf"][keep])          # ascending, the best box is written last
    f = det["frame"][keep][order]
    xy[f] = ground[keep][order]
    c[f] = det["conf"][keep][order]
    return xy, c


def fill(ball_xy, conf=0.3, max_speed=35.0, sure=0.6):
    """Tracking ball with its holes filled from the detector.

    A box is taken when it is confident (>= sure) or when it sits where the ball could have got
    to from the last known position at max_speed; a lone box far from everything is usually a
    head, a boot or the penalty spot.
    """
    det = load()
    box, c = best_box(det, conf)
    out = ball_xy.copy()
    last, last_f = None, None
    for i in range(len(out)):
        if not np.isnan(out[i, 0]):
            last, last_f = out[i], i
            continue
        if np.isnan(box[i, 0]):
            continue
        reach = np.inf if last is None else max_speed * (i - last_f) / config.FPS + 1.0
        if c[i] >= sure or np.hypot(*(box[i] - last)) <= reach:
            out[i] = box[i]
            last, last_f = box[i], i
    return out
