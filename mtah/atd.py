"""Load the raw tracking file into numpy arrays, in metres, indexed by video frame.

Raw line: "<frame_id>:<x,y,speed>;<x,y,speed>;...:<bx,by,bs>", NaN where nothing was tracked.
frame_id counts from 1, video frames (and submission rows) from 0.

First load parses ~200 MB of text and takes a while, so the result is cached as .npz in
work/cache and later loads are instant. Delete the cache if the data bundle ever changes.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config

HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """Team of each track slot, 0 home / 1 away. First team in the XML is home (Barcelona)."""
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    names = [t.findtext("Name") for t in root.iter("Team")]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, got {}".format(meta_path, len(teams)))
    slots = [p.get("teamId") for p in root.iter("Player")]
    team = np.array([HOME if s == teams[0] else AWAY for s in slots], dtype=np.int8)
    return team, tuple(names)


def keeper_slots(meta_path):
    """Boolean per slot: the XML marks one goalkeeper per team (position_type)."""
    root = ET.parse(meta_path).getroot()
    out = []
    for player in root.iter("Player"):
        kinds = [p.findtext("Value") for p in player.iter("ProviderParameter") if p.findtext("Name") == "position_type"]
        out.append("Goalkeeper" in kinds)
    return np.array(out, dtype=bool)


def _parse_raw(raw_path):
    with open(raw_path) as handle:
        text = handle.read()
    # three separators carry no extra information once the column count is known
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float32, na_values=["NaN"])
    values = table.values
    n_slots = (values.shape[1] - 4) // 3
    frame_id = values[:, 0].astype(np.int64)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:]
    return frame_id, players, ball


def load(source="tactical", use_cache=True):
    """dict with everything per video frame, positions already in metres.

    frame        (N,)      video frame number, 0-based
    ball_xy      (N, 2)    NaN when the ball was not tracked
    players_xy   (N, S, 2) NaN for empty slots
    players_speed(N, S)    m/s as reported by the tracker
    team         (S,)      0 = home (Barcelona), 1 = away (Athletic)
    keeper       (S,)      True for the two goalkeeper slots
    team_names   (home, away)
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache = os.path.join(config.CACHE_DIR, source + "_atd.npz")
    if use_cache and os.path.isfile(cache):
        data = np.load(cache, allow_pickle=False)
        out = {k: data[k] for k in data.files} | {"team_names": tuple(data["team_names"])}
        out["keeper"] = keeper_slots(config.asset("atd_meta", source))
        return out

    team, names = slot_teams(config.asset("atd_meta", source))
    frame_id, players, ball = _parse_raw(config.asset("atd_raw", source))

    scale = np.array([config.PITCH_LENGTH, config.PITCH_WIDTH], dtype=np.float32)
    out = {
        "frame": frame_id - 1,
        "ball_xy": ball[:, :2] * scale,
        "players_xy": players[:, :, :2] * scale,
        "players_speed": players[:, :, 2],
        "team": team,
        "keeper": keeper_slots(config.asset("atd_meta", source)),
    }
    if use_cache:
        np.savez(cache, team_names=np.array(names), **out)
    out["team_names"] = names
    return out


def ball_visible(data):
    return ~np.isnan(data["ball_xy"][:, 0])


def nearest_player(data):
    """Per frame: (team of nearest tracked player, distance in metres). -1 / NaN when no answer."""
    diff = data["players_xy"] - data["ball_xy"][:, None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))          # (N, S), NaN where either side is missing
    has_any = ~np.all(np.isnan(dist), axis=1)
    idx = np.full(len(dist), -1)
    best = np.full(len(dist), np.nan, dtype=np.float32)
    idx[has_any] = np.nanargmin(dist[has_any], axis=1)
    best[has_any] = dist[has_any, idx[has_any]]
    team = np.where(idx >= 0, data["team"][np.clip(idx, 0, None)], -1).astype(np.int8)
    return team, best
