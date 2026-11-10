"""Rule model: possession from the tracking alone, following notes/ATURAN-ADMIN.md.

The baseline gives the ball to whoever is nearest on every frame and calls a lost ball dead. The
admin does neither, so this model:

- holds possession with the last team until the other team shows control for a few frames, then
  moves the switch back to the first of those frames (the admin switches on the first touch,
  rules 2-4), so a ball flying past a player, a ball that vanishes, or a ring that drifts onto an
  opponent for a moment changes nothing (rules 1, 5, 7)
- only counts a player as touching when the ball is slow enough to be at his feet; a fast ball is
  a pass or a ball in the air, and its 2D ring can sit right on a player it only flies over
- marks the ball dead when the tracked ball leaves the pitch or when both teams slow down together
  for a while (the whistle, rule 8), and ignores every touch until play restarts (rule 11)
- ends D at the kick of a ball that sat still (or was not seen) and gives the ball to the team
  nearest it, since the restart taker nearly always is; a kick-off is a slow tap, so off the
  centre spot any movement counts (rules 12-16)

Touches seen in the video ball (mtah/touch.py) pull a possession switch back to the other team's
first touch, and a sharp turn of the ball by the team not in possession starts a possession. The
video detector is a public model trained on other footage. It never creates a stoppage.
"""

import warnings

import numpy as np

from . import atd, config, touch

DEFAULTS = {
    "reach": 1.5,       # m, ball this close to a player counts as at his feet
    "margin": 0.5,      # m, and the nearest opponent at least this much further away
    "max_speed": 9.0,   # m/s, faster than this the ball is travelling, nobody controls it
    "confirm": 5,       # frames of control before the other team gets the ball
    "gap": 5,           # frames without evidence allowed inside one run of control
    "slow": 1.6,        # m/s, player speed (see fast) below this ...
    "slow_for": 2.0,    # s, ... for this long means the ball is dead
    "resume": 2.6,      # m/s, speed back above this ends it
    "lead": 1.0,        # s, the slowdown trails the whistle, so D starts this much earlier
    "out_margin": 0.5,  # m beyond the line before a tracked ball counts as out
    "kick": 6.0,        # m/s, ball this fast after it sat still or went out = restart taken
    "settle": 2.0,      # s, no restart is looked for sooner than this after the ball went out
    "restart": 1,       # 1 = end D at the kick of a ball that sat still, 0 = wait for the players
    "still": 1.0,       # s the ball has to sit still before that kick
    "seed": 2.0,        # m, after a restart the nearest team within this takes the ball
    "spot": 1.5,        # m from the centre spot where a slow tap already counts (kick-off)
    "fast": 1,          # 1 = watch the fastest quarter of the players, 0 = the median one (v1, v2)
    "veto": 0.3,        # share of the first veto_for s with a player on the ball that cancels a slowdown
    "veto_for": 4.0,
    "veto_seen": 0.9,   # ... counted only when the tracking sees the ball this much of that time
    "touch": 1,         # 1 = move switches back to the video touch, 0 = v3
    "touch_start": 1,   # 1 = a sharp video touch by the other team starts a possession (v5), 0 = v4
    "fake": 6.0,        # s, a restart kick after which nobody gets going for this long was not the
                        # restart (ball handed back, keeper placing it); 0 = off
    "fake_pace": 1.75,  # m/s, "getting going" = the fast quarter's pace reaching this
    "gk_far": 50.0,     # m, a keeper's point further than this from his own goal is a swapped
                        # track sitting on somebody else (often the other keeper); 0 = keep
}


def clean(data, params=None):
    """The tracking with its known-impossible points removed.

    The two keeper slots wander: on about 1% of frames each sits 90-100 m from its own goal, on
    an outfield player or on the other keeper (case 01: the Athletic keeper's point on ter Stegen
    as he picks the ball up, so A). A keeper that far from his goal in play does not happen, so
    those frames are dropped. Which goal is his comes from the slot's median x per half.
    """
    p = dict(DEFAULTS, **(params or {}))
    if not p["gk_far"] or "keeper" not in data:
        return data
    xy = data["players_xy"].copy()
    speed = data["players_speed"].copy()
    for j in np.flatnonzero(data["keeper"]):
        for lo, hi in config.HALVES.values():
            x = xy[lo:hi + 1, j, 0]
            seen = ~np.isnan(x)
            if not seen.any():
                continue
            goal = 0.0 if np.median(x[seen]) < config.PITCH_LENGTH / 2 else config.PITCH_LENGTH
            far = seen & (np.abs(x - goal) > p["gk_far"])
            xy[lo:hi + 1][far, j] = np.nan
            speed[lo:hi + 1][far, j] = np.nan
    return dict(data, players_xy=xy, players_speed=speed)


def _rolling(values, window, fn):
    pad = window // 2
    padded = np.pad(values, (pad, window - 1 - pad), mode="edge")
    return fn(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)


def ball_speed(data):
    """m/s per frame from a centred difference, median over 5 frames to kill single-frame jumps."""
    xy = data["ball_xy"]
    speed = np.full(len(xy), np.nan, dtype=np.float32)
    speed[1:-1] = np.hypot(*(xy[2:] - xy[:-2]).T) * config.FPS / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN windows while the ball is lost
        return _rolling(speed, 5, np.nanmedian)


def team_distances(data):
    """(N, 2) distance from the ball to the nearest tracked player of each team, NaN if none."""
    out = np.full((len(data["ball_xy"]), 2), np.nan, dtype=np.float32)
    for team in (atd.HOME, atd.AWAY):
        slots = data["team"] == team
        diff = data["players_xy"][:, slots] - data["ball_xy"][:, None, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        has = ~np.all(np.isnan(dist), axis=1)
        out[has, team] = np.nanmin(dist[has], axis=1)
    return out


def features(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_speed = np.nanmedian(data["players_speed"], axis=1)
        fast_speed = np.nanpercentile(data["players_speed"], 75, axis=1)
    median_speed = np.where(np.isnan(median_speed), np.nanmean(median_speed), median_speed)
    fast_speed = np.where(np.isnan(fast_speed), np.nanmean(fast_speed), fast_speed)
    return {
        "ball_xy": data["ball_xy"],
        "ball_speed": ball_speed(data),
        "dist": team_distances(data),
        "pace": _rolling(median_speed, config.FPS, np.mean),
        # after a whistle the median player only slows to a jog (~1.25 m/s); the fastest quarter
        # drops from ~3 to ~1.5 m/s within two seconds, far clearer against open play (~2.7)
        "pace_fast": _rolling(fast_speed, config.FPS, np.mean),
    }


def evidence(feat, p):
    """Per frame: team clearly in control of the ball, or -1."""
    dist, speed = feat["dist"], feat["ball_speed"]
    slow_ball = ~(speed > p["max_speed"])            # NaN speed (ball just reappeared) passes
    out = np.full(len(dist), -1, dtype=np.int8)
    for team, other in ((atd.HOME, atd.AWAY), (atd.AWAY, atd.HOME)):
        mine, theirs = dist[:, team], dist[:, other]
        clear = np.isnan(theirs) | (theirs > mine + p["margin"])
        out[slow_ball & (mine <= p["reach"]) & clear] = team
    return out


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)


def dead_ball(feat, p):
    """Boolean mask of frames the model calls dead, before restarts are decided."""
    n = len(feat["pace"])
    dead = np.zeros(n, dtype=bool)

    # both teams slowing down together: the whistle has gone
    pace = feat["pace_fast"] if p["fast"] else feat["pace"]
    slow = pace < p["slow"]
    lead = int(p["lead"] * config.FPS)
    ev = evidence(feat, p) if p["veto"] else None
    seen = ~np.isnan(feat["ball_xy"][:, 0])
    for s, e in _runs(slow):
        if e - s + 1 < p["slow_for"] * config.FPS:
            continue
        # players also slow down in open play (a keeper with the ball, a slow build-up), but then
        # somebody is still on a ball the tracking sees; after a whistle the ball is left alone
        # or picked up and lost from the tracking
        if ev is not None:
            w = slice(s, s + int(p["veto_for"] * config.FPS))
            if seen[w].mean() >= p["veto_seen"] and (ev[w] >= 0).mean() >= p["veto"]:
                continue
        end = e
        while end + 1 < n and pace[end + 1] < p["resume"]:
            end += 1
        dead[max(0, s - lead):end + 1] = True

    # tracked ball beyond a line: dead from that frame until somebody kicks it back in
    x, y = feat["ball_xy"][:, 0], feat["ball_xy"][:, 1]
    m = p["out_margin"]
    with np.errstate(invalid="ignore"):
        outside = (x < -m) | (x > config.PITCH_LENGTH + m) | (y < -m) | (y > config.PITCH_WIDTH + m)
        inside = (x >= 0) & (x <= config.PITCH_LENGTH) & (y >= 0) & (y <= config.PITCH_WIDTH)
        kicked = inside & (feat["ball_speed"] > p["kick"])
    settle = int(p["settle"] * config.FPS)
    i = 0
    while i < n:
        if not outside[i]:
            i += 1
            continue
        later = np.flatnonzero(kicked[i + settle:])
        end = i + settle + later[0] - 1 if len(later) else n - 1
        dead[i:end + 1] = True
        i = end + 1

    # a ball that sat still and is then struck is the restart, even while the players are still
    # walking; without this the slowdown keeps D going seconds into the new possession. At half
    # the public restarts the ball is not tracked while it waits, so "not seen" counts as still
    if p["restart"]:
        still = int(p["still"] * config.FPS)
        speed = feat["ball_speed"]
        quiet = (np.isnan(speed) | (speed < 1.5)).astype(np.float32)
        before = np.concatenate((np.zeros(still), _rolling(quiet, still, np.mean)[:-still]))
        # a kick-off is a short tap, far below a set-piece strike, so off the centre spot any
        # movement counts
        cx, cy = config.PITCH_LENGTH / 2, config.PITCH_WIDTH / 2
        with np.errstate(invalid="ignore"):
            on_spot = np.hypot(feat["ball_xy"][:, 0] - cx, feat["ball_xy"][:, 1] - cy) < p["spot"]
            struck = ((speed > p["kick"]) | (on_spot & (speed > 1.5))) & (before > 0.8)
        for s, e in list(_runs(dead)):
            hit = np.flatnonzero(struck[s + settle:e + 1])
            if len(hit):
                dead[s + settle + hit[0]:e + 1] = False
        # a struck ball is not always the restart: during a stoppage the ball gets kicked back to
        # the taker, or the keeper strikes it to the spot. After every real restart in the keys
        # the fast quarter reaches 2.0 m/s within 8 s; after a hand-back they stay under 1.65. So
        # a strike nobody gets going after was not it, and the stoppage goes on to the next strike
        # of a ball that sat still, checked the same way
        if p["fake"]:
            fake = int(p["fake"] * config.FPS)
            with np.errstate(invalid="ignore"):
                fast = speed > p["kick"]
            kicks = np.flatnonzero(fast & ~np.roll(fast, 1))
            for s, e in list(_runs(dead)):
                j = e + 1
                if j >= n or dead[j] or np.nanmax(pace[j:j + fake]) >= p["fake_pace"]:
                    continue
                # the real restart is the next strike the players do get going after; the ball
                # need not have sat still for it, the taker often nudges it into place first
                end = n
                for k in kicks[kicks > j + settle]:
                    if dead[k] or np.nanmax(pace[k:k + fake]) >= p["fake_pace"]:
                        end = k
                        break
                dead[j:end] = True
    return dead


def decode(ev, dead, lo, hi, p, near=None):
    """Labels for frames lo..hi of one half from evidence and dead-ball mask.

    near: (team, distance) of the nearest player per frame. Right after a restart nobody may be
    on the ball for seconds (a kick-off goes straight back), so until real control shows up the
    team nearest the ball takes it; the restart taker is nearly always that player.
    """
    out = np.full(hi - lo + 1, "D", dtype="<U1")
    code = {atd.HOME: "H", atd.AWAY: "A"}
    state = cand = None
    run_start = run_len = 0
    last_seen = -10 ** 9
    for i in range(lo, hi + 1):
        if dead[i]:
            state = cand = None
            run_len = 0
            continue
        e = int(ev[i])
        if e >= 0:
            if e == cand and i - last_seen <= p["gap"] + 1:
                run_len += 1
            else:
                cand, run_start, run_len = e, i, 1
            last_seen = i
        elif cand is not None and i - last_seen > p["gap"] + 1:
            cand, run_len = None, 0
        if cand is not None and cand != state and run_len >= p["confirm"]:
            # the switch belongs to the first touch, not to the frame the model became sure
            out[run_start - lo:i - lo] = code[cand]
            state = cand
        if state is None and near is not None and near[1][i] <= p["seed"]:
            state = int(near[0][i])
        if state is not None:
            out[i - lo] = code[state]
    return out


def predict(data, params=None, feat=None):
    """Full-match label array. Outside the two halves the answer is X (not graded).

    feat, when given, must come from features(clean(data, params)) with the same params."""
    p = dict(DEFAULTS, **(params or {}))
    data = clean(data, p)
    feat = features(data) if feat is None else feat
    ev = evidence(feat, p)
    dead = dead_ball(feat, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        near = (np.nanargmin(np.nan_to_num(feat["dist"], nan=np.inf), axis=1), np.nanmin(feat["dist"], axis=1))
    labels = np.full(config.N_FRAMES, "X", dtype="<U1")
    for half, (lo, hi) in config.HALVES.items():
        kickoff = config.KICKOFF[half]
        labels[lo:kickoff] = "D"
        # the players stand still before a kick-off, so the slowdown would carry D past it; the
        # kick-off frames are published boundaries, not answers
        live = dead.copy()
        end = kickoff
        while end <= hi and live[end]:
            end += 1
        live[kickoff:end] = False
        labels[kickoff:hi + 1] = decode(ev, live, kickoff, hi, p, near)
    if p["touch"]:
        labels = touch.apply(data, labels, {"start": p["touch_start"]}, ev, p["confirm"], p["gap"])
    return labels
