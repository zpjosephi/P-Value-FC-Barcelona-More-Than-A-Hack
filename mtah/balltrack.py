"""One ball position per frame from the tracking and the video detector together.

Every frame has up to six guesses for the ball: the tracking's point and up to five boxes from the
detector (tools/detect_ball.py), projected to the grass. Each source is wrong in its own way: the
tracking jumps onto players and drops the ball for seconds, the detector fires on heads, boots and
the penalty spot. Taken one frame at a time neither can be trusted where they disagree, and they
disagree by more than 10 m on a fifth of the frames where both have an answer.

A ball cannot jump, though. So the track is the cheapest path through the guesses, solved exactly
with dynamic programming (Viterbi):

- each guess costs something: little for the tracking point, more for a weak box
- moving between guesses k frames apart costs d^2 / (k sigma^2), a random walk, so a 1 m step per
  frame (a 25 m/s pass) is cheap and a 10 m jump in one frame is not; faster than max_speed is
  impossible
- a frame can be left empty at a price per frame, and a path may jump straight across up to
  `bridge` empty frames if the distance fits; after a longer gap the path re-enters anywhere at a
  fixed price, since the ball can have gone anywhere by then

The result replaces data["ball_xy"] and everything downstream runs unchanged.
"""

import numpy as np

from . import balldet, config

DEFAULTS = {
    "track_cost": 0.3,   # choosing the tracking point
    "det_cost": 2.5,     # times (1 - conf) for choosing a detector box
    "det_min": 0.1,      # boxes below this confidence are not candidates
    "skip": 2.0,         # per frame left without a ball
    "sigma": 1.0,        # m per sqrt(frame), the random walk's step
    "max_speed": 40.0,   # m/s
    "bridge": 12,        # frames a path may jump across without a candidate
    "reenter": 6.0,      # starting again after the ball was lost
}
K = 6                    # candidates per frame: the tracking point and five boxes


def candidates(ball_xy, det, p):
    """(N, K, 2) positions and (N, K) costs, inf where a slot is empty. Slot 0 is the tracking."""
    n = config.N_FRAMES
    xy = np.full((n, K, 2), np.nan)
    cost = np.full((n, K), np.inf)
    ok = ~np.isnan(ball_xy[:, 0])
    xy[ok, 0] = ball_xy[ok]
    cost[ok, 0] = p["track_cost"]

    ground = balldet.to_pitch(det["frame"], det["px"].astype(float), (det["py"] + det["h"] / 2).astype(float))
    m = 2.0
    keep = ((det["conf"] >= p["det_min"]) & (ground[:, 0] > -m) & (ground[:, 0] < config.PITCH_LENGTH + m) &
            (ground[:, 1] > -m) & (ground[:, 1] < config.PITCH_WIDTH + m))
    f, g, c = det["frame"][keep], ground[keep], det["conf"][keep]
    order = np.lexsort((-c, f))
    f, g, c = f[order], g[order], c[order]
    first = np.searchsorted(f, f)
    rank = np.arange(len(f)) - first
    use = rank < K - 1
    xy[f[use], 1 + rank[use]] = g[use]
    cost[f[use], 1 + rank[use]] = p["det_cost"] * (1 - c[use])
    return xy, cost


def viterbi(xy, cost, lo, hi, p):
    """Track for frames lo..hi: (len, 2) positions (NaN when lost) and (len,) chosen slot (-1)."""
    n = hi - lo + 1
    B = int(p["bridge"])
    reach = p["max_speed"] / config.FPS
    inv = 1.0 / p["sigma"] ** 2
    skip, reenter = p["skip"], p["reenter"]

    best = np.full((n, K), np.inf)
    lost = np.zeros(n)
    back_k = np.zeros((n, K), np.int16)       # 0 = came from the lost state
    back_i = np.zeros((n, K), np.int8)
    lost_from = np.full(n, -1, np.int8)       # -1 = was lost already, else slot at t-1

    for t in range(n):
        node = cost[lo + t]
        here = xy[lo + t]
        prev_lost = lost[t - 1] if t else 0.0
        entry = prev_lost + reenter + node
        best[t] = entry
        if t:
            ks = np.arange(1, min(B, t) + 1)
            src_best = best[t - ks]                                    # (k, K)
            src_xy = xy[lo + t - ks]                                   # (k, K, 2)
            d2 = ((src_xy[:, :, None, :] - here[None, None, :, :]) ** 2).sum(-1)   # (k, K, K)
            kk = ks[:, None, None].astype(float)
            move = d2 * inv / kk
            move[d2 > (reach * kk + 1.0) ** 2] = np.inf
            total = src_best[:, :, None] + move + (kk - 1) * skip + node[None, None, :]
            total = np.where(np.isnan(total), np.inf, total)
            flat = total.reshape(-1, K)
            arg = flat.argmin(0)
            val = flat[arg, np.arange(K)]
            better = val < best[t]
            best[t] = np.where(better, val, best[t])
            back_k[t] = np.where(better, ks[arg // K], 0)
            back_i[t] = np.where(better, arg % K, 0)
        # lost: stay lost, or drop the ball from a candidate one frame back
        stay = prev_lost + skip
        if t:
            j = int(np.argmin(best[t - 1]))
            drop = best[t - 1, j] + skip
            if drop < stay:
                lost[t], lost_from[t] = drop, j
                continue
        lost[t] = stay

    out = np.full((n, 2), np.nan)
    slot = np.full(n, -1, np.int8)
    t = n - 1
    j = int(np.argmin(best[t]))
    in_lost = lost[t] < best[t, j]
    while t >= 0:
        if in_lost:
            if lost_from[t] >= 0:
                j, in_lost = int(lost_from[t]), False
            t -= 1
            continue
        out[t] = xy[lo + t, j]
        slot[t] = j
        k = int(back_k[t, j])
        if k == 0:
            in_lost = True
            t -= 1
        else:
            j = int(back_i[t, j])
            t -= k
    return out, slot


def track(ball_xy, params=None, det=None):
    """(N_FRAMES, 2) ball and (N_FRAMES,) source: 0 tracking, 1-5 detector box rank, -1 none."""
    p = dict(DEFAULTS, **(params or {}))
    det = balldet.load() if det is None else det
    xy, cost = candidates(ball_xy, det, p)
    out = np.full((config.N_FRAMES, 2), np.nan)
    src = np.full(config.N_FRAMES, -1, np.int8)
    covered = balldet.covered()
    for lo, hi in config.HALVES.values():
        if not covered[lo:hi + 1].all():
            raise ValueError("detector has not been run on frames {}-{}".format(lo, hi))
        out[lo:hi + 1], src[lo:hi + 1] = viterbi(xy, cost, lo, hi, p)
    return out, src
