"""Touches seen in the video ball, used to re-time the rule model's possession switches.

The admin switches possession on the other team's first touch. The rules switch once the tracking
shows control (ball slow, player within reach, opponent further away), which lands a median
second after the touch and later still when the two teams are tangled. The video detector sees
the ball itself, frame by frame and unsmoothed, so a touch shows as a kink: the ball's velocity
changes by several m/s within two frames. A kink with one team's player clearly nearest is a touch
by that team.

Touches are not trusted to make switches on their own: a kink the detector misses, or a
deflection the admin ignores, splits a possession, and spellF1 punishes that harder than a late
switch. They only move a switch the rules already made back to
the touch frame, at most `back` seconds earlier, and never across a frame where the old team is
plainly on the ball. Moving early switches later was tried too and loses on missed touches.
"""

import os

import numpy as np

from . import balldet, balltrack, config

DEFAULTS = {
    "kink": 4.0,     # m/s change of the video ball's velocity across two frames
    "reach": 2.0,    # m, the touching team's player this close to the ball ...
    "margin": 0.0,   # m, ... and the other team's nearest at least this much further
    "back": 5.0,     # s, how far a switch may move back to its touch
    "hold": 0.5,     # m, old team's player this close to the ball ...
    "free": 3.0,     # m, ... with the new team's further than this = old team still on it, stop
    # a sharp touch by the team not in possession starts a possession of its own (v5)
    "start": 1,      # 0 = v4, switches are only moved
    "start_kink": 4.0,
    "start_reach": 3.0,
    "angle": 60.0,  # degrees the ball turns at the touch; a header or clearance sends it back
    "settle": 1.0,  # s after the touch during which the old team's control is not believed:
                     # the tracking ball lags the real one by a few frames and still sits at the
                     # feet of the player who just lost it
    "min_len": 1.0,  # s, a possession that would end sooner than this is not written
}
CACHE = os.path.join(config.CACHE_DIR, "video_ball.npz")
SPAN = 2


def video_ball(refresh=False):
    """(N_FRAMES, 2) detector-only ball track; the tracking's point is left out because its
    smoothing hides the very kinks this looks for."""
    det = balldet.load()
    key = "{}-{}".format(int(balldet.covered().sum()), len(det["frame"]))
    if not refresh and os.path.exists(CACHE):
        z = np.load(CACHE)
        if str(z["key"]) == key:
            return z["ball"]
    ball, _ = balltrack.track(np.full((config.N_FRAMES, 2), np.nan), det=det)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE, ball=ball, key=key)
    return ball


def team_distance(data, ball):
    """(N, 2) distance from the given ball to each team's nearest tracked player."""
    out = np.full((len(ball), 2), np.nan)
    diff = data["players_xy"] - ball[:, None, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    for team in (0, 1):
        slots = data["team"] == team
        has = ~np.all(np.isnan(dist[:, slots]), axis=1)
        out[has, team] = np.nanmin(dist[has][:, slots], axis=1)
    return out


def touches(ball, dteam, p):
    """int8 per frame: team that touched the ball there, else -1. One frame per kink."""
    v_in = np.full_like(ball, np.nan)
    v_in[SPAN:] = (ball[SPAN:] - ball[:-SPAN]) / SPAN * config.FPS
    v_out = np.roll(v_in, -SPAN, 0)
    dv = np.hypot(*(v_out - v_in).T)
    kink = dv > p["kink"]
    who = np.full(len(ball), -1, np.int8)
    for team, other in ((0, 1), (1, 0)):
        mine, theirs = dteam[:, team], dteam[:, other]
        clear = np.isnan(theirs) | (theirs > mine + p["margin"])
        who[kink & (mine <= p["reach"]) & clear] = team
    out = np.full(len(ball), -1, np.int8)
    m = who >= 0
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        j = s + int(np.nanargmax(dv[s:e + 1]))
        out[j] = who[j]
    return out


def shift(labels, tch, dteam, p):
    """Move each H<->A switch back to the new team's touch, if one precedes it closely enough."""
    code = {"H": 0, "A": 1}
    out = labels.copy()
    back = int(p["back"] * config.FPS)
    for s in np.flatnonzero(out[1:] != out[:-1]) + 1:
        if out[s] not in code or out[s - 1] not in code:
            continue
        new, old = code[out[s]], code[out[s - 1]]
        first = None
        for j in range(s - 1, max(0, s - back) - 1, -1):
            if tch[j] == old or out[j] not in code:
                break
            if dteam[j, old] < p["hold"] and not (dteam[j, new] < p["free"]):
                break
            if tch[j] == new:
                first = j
        if first is not None:
            out[first:s] = out[s]
    return out


def sharp(ball, dteam, p):
    """int8 per frame: team whose player turned the ball by at least `angle` degrees, else -1.

    Velocities are central differences over SPAN frames taken SPAN frames before and after, so
    the turn is measured over about a third of a second of the detector-only track."""
    v = np.full_like(ball, np.nan)
    v[SPAN:-SPAN] = (ball[2 * SPAN:] - ball[:-2 * SPAN]) / (2 * SPAN) * config.FPS
    v_in, v_out = np.roll(v, SPAN, 0), np.roll(v, -SPAN, 0)
    dv = np.hypot(*(v_out - v_in).T)
    s_in, s_out = np.hypot(*v_in.T), np.hypot(*v_out.T)
    cos = (v_in * v_out).sum(axis=1) / np.maximum(s_in * s_out, 1e-6)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    kink = dv > p["start_kink"]
    who = np.full(len(ball), -1, np.int8)
    for team, other in ((0, 1), (1, 0)):
        mine, theirs = dteam[:, team], dteam[:, other]
        clear = np.isnan(theirs) | (theirs > mine + p["margin"])
        who[kink & (mine <= p["start_reach"]) & clear] = team
    out = np.full(len(ball), -1, np.int8)
    m = who >= 0
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
        j = s + int(np.nanargmax(dv[s:e + 1]))
        if turn[j] >= p["angle"]:
            out[j] = who[j]
    return out


def start(labels, tch, ev, p, confirm, gap):
    """Write a possession for the team not in possession from each of its sharp touches.

    It lasts until the old team touches sharply again, shows control the way decode() counts it
    (`confirm` evidence frames with `gap` allowed, none of them inside `settle`), or until the
    labels change on their own. The rules' evidence (ev) comes from the tracking ball, which is
    why the first settle seconds are ignored."""
    code = {"H": 0, "A": 1}
    names = "HA"
    out = labels.copy()
    settle = int(round(p["settle"] * config.FPS))
    min_len = int(round(p["min_len"] * config.FPS))
    limit = 20 * config.FPS
    for j in np.flatnonzero(tch >= 0):
        new, cur = int(tch[j]), out[j]
        if cur not in code or code[cur] == new:
            continue
        old = code[cur]
        stop, end = j + 1, min(len(out), j + limit)
        run_start, run, last = None, 0, -10 ** 9
        while stop < end and out[stop] == cur:
            if tch[stop] == old:
                break
            if stop - j > settle and ev[stop] == old:
                if run_start is None or stop - last > gap + 1:
                    run_start, run = stop, 0
                run += 1
                last = stop
                if run >= confirm:
                    stop = run_start
                    break
            stop += 1
        if stop - j >= min_len:
            out[j:stop] = names[new]
    return out


def apply(data, labels, params=None, ev=None, confirm=5, gap=5):
    p = dict(DEFAULTS, **(params or {}))
    ball = video_ball()
    dteam = team_distance(data, ball)
    out = shift(labels, touches(ball, dteam, p), dteam, p)
    if p["start"] and ev is not None:
        out = start(out, sharp(ball, dteam, p), ev, p, confirm, gap)
    return out
