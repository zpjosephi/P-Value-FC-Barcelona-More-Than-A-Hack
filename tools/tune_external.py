"""Pick the rule constants on Metrica's public sample games, never on BAR-ATH.

    python tools/tune_external.py --check    # scores with the constants now in the code
    python tools/tune_external.py            # coordinate descent, prints the chosen values

Data: github.com/metrica-sports/sample-data, cloned next to this repo as metrica-sample/. Those
games ship tracking and synchronised events, so possession labels can be derived from the events
with no human input of ours: each on-ball event gives its team the ball from its start frame, a
ball out, foul or goal is dead until the next set piece. The search starts from the middle of each
grid so that it does not inherit earlier values.
"""

import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mtah import config, rules, score, touch  # noqa: E402

SAMPLE = os.path.join(config.ROOT, "metrica-sample", "data")
LENGTH, WIDTH = 105.0, 68.0
ON_BALL = {"PASS", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST"}
RULE_GRID = {
    "reach": [1.0, 1.5, 2.0, 2.5, 3.0], "margin": [0.0, 0.25, 0.5, 1.0],
    "max_speed": [6.0, 9.0, 12.0, 15.0], "confirm": [2, 3, 5, 8, 12], "gap": [2, 5, 10],
    "slow": [1.2, 1.6, 2.0, 2.4], "slow_for": [1.0, 2.0, 3.0, 4.0], "resume": [2.0, 2.6, 3.2],
    "lead": [0.0, 0.5, 1.0, 2.0], "out_margin": [0.0, 0.5, 1.0], "kick": [4.0, 6.0, 8.0, 10.0],
    "settle": [1.0, 2.0, 3.0], "still": [0.5, 1.0, 2.0], "seed": [1.0, 2.0, 3.0, 5.0],
    "spot": [1.0, 1.5, 3.0], "veto": [0.1, 0.3, 0.5], "veto_for": [2.0, 4.0, 6.0],
    "fake": [0.0, 4.0, 6.0, 8.0], "fake_pace": [1.5, 1.75, 2.0, 2.5],
}
TOUCH_GRID = {
    "kink": [4.0, 6.0, 8.0, 12.0], "reach": [1.0, 1.5, 2.0, 3.0], "margin": [0.0, 0.5, 1.0],
    "back": [1.0, 2.0, 3.0, 5.0], "hold": [0.5, 1.0, 1.5], "free": [1.5, 2.0, 3.0],
    "start": [0, 1], "start_kink": [4.0, 6.0, 8.0, 12.0], "start_reach": [1.0, 1.5, 2.0, 3.0],
    "angle": [60.0, 90.0, 120.0, 150.0], "settle": [0.0, 0.25, 0.5, 1.0],
    "min_len": [0.0, 0.25, 0.5, 1.0],
}


def read_team(game, side):
    folder = "Sample_Game_{}".format(game)
    name = "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side)
    return pd.read_csv(os.path.join(SAMPLE, folder, name), skiprows=2)


def load(game):
    home, away = read_team(game, "Home"), read_team(game, "Away")
    frames = home["Frame"].values
    n = int(frames.max()) + 2
    scale = np.array([LENGTH, WIDTH], np.float32)
    blocks, team = [], []
    for code, table in ((0, home), (1, away)):
        for c in [c for c in table.columns if c.startswith("Player")]:
            i = table.columns.get_loc(c)
            blocks.append(table.iloc[:, i:i + 2].values.astype(np.float32))
            team.append(code)
    xy = np.full((n, len(blocks), 2), np.nan, np.float32)
    xy[frames] = np.stack(blocks, axis=1) * scale
    i = home.columns.get_loc("Ball")
    ball = np.full((n, 2), np.nan, np.float32)
    ball[frames] = home.iloc[:, i:i + 2].values.astype(np.float32) * scale
    speed = np.full((n, xy.shape[1]), np.nan, np.float32)
    speed[2:-2] = np.hypot(*(xy[4:] - xy[:-4]).transpose(2, 0, 1)) * config.FPS / 4
    periods = {int(p): (int(g["Frame"].min()), int(g["Frame"].max()))
               for p, g in home.groupby("Period") if p in (1, 2)}
    data = {"ball_xy": ball, "players_xy": xy, "players_speed": speed,
            "team": np.array(team), "keeper": np.zeros(len(team), bool)}
    return data, periods, n


def truth(game, periods, n):
    folder = "Sample_Game_{}".format(game)
    events = pd.read_csv(os.path.join(SAMPLE, folder, folder + "_RawEventsData.csv"))
    events = events.sort_values("Start Frame", kind="stable")
    sets = set(int(f) for f in events[events["Type"] == "SET PIECE"]["Start Frame"])
    marks = []
    for team, kind, sub, start, end in zip(events["Team"], events["Type"], events["Subtype"],
                                           events["Start Frame"], events["End Frame"]):
        start, end = int(start), int(end)
        sub = sub if isinstance(sub, str) else ""
        if kind in ("BALL OUT", "FAULT RECEIVED"):
            marks.append((start, "D"))
        elif kind == "SHOT" and ("GOAL" in sub or "OUT" in sub):
            marks.append((start, team[0]))
            marks.append((max(end, start + 1), "D"))
        elif kind in ON_BALL:
            marks.append((start, team[0]))
    out = np.full(n, "X", dtype="<U1")
    kick = {}
    for p, (lo, hi) in periods.items():
        inside = sorted(m for m in marks if lo <= m[0] <= hi)
        kick[p] = inside[0][0]
        out[lo:kick[p]] = "D"
        dead = False
        for k, (f, lab) in enumerate(inside):
            nxt = inside[k + 1][0] if k + 1 < len(inside) else hi + 1
            if lab == "D":
                dead = True
            elif f in sets:
                dead = False
            out[f:nxt] = "D" if dead else lab
    return out, kick


def holes(ball, seed, share=0.33, mean_len=300):
    """Copy of the ball with random missing stretches covering about `share` of the frames."""
    rng = np.random.default_rng(seed)
    out = ball.copy()
    i = 0
    while i < len(out):
        i += int(rng.exponential(mean_len * (1 - share) / share))
        length = int(rng.exponential(mean_len))
        out[i:i + length] = np.nan
        i += length
    return out


def jitter(ball, seed, sigma=0.11):
    """The detector's ball is a box centre found again on every frame, so it is not smooth like the
    sample tracking. Each frame gets its own error of about half a ball's diameter (0.11 m), set
    from the ball's size and not from BAR-ATH."""
    rng = np.random.default_rng(seed)
    return ball.astype(float) + rng.normal(0.0, sigma, ball.shape)


class Game:
    def __init__(self, number, sigma=0.11):
        self.data, self.periods, self.n = load(number)
        labels, self.kick = truth(number, self.periods, self.n)
        self.truth = score.pm.label_codes(labels)
        # the sample tracking sees the ball far more often than BAR-ATH's automatic tracking (about
        # 90% against 59% of live frames), so holes are punched to match; the video ball gets its own
        full = self.data["ball_xy"]
        self.data["ball_xy"] = holes(full, number)
        self.ball = jitter(holes(full, number + 100), number + 200, sigma)
        self.feat = None

    def score(self, rule_params):
        config.N_FRAMES, config.PITCH_LENGTH, config.PITCH_WIDTH = self.n, LENGTH, WIDTH
        config.HALVES, config.KICKOFF = self.periods, self.kick
        touch.video_ball = lambda: self.ball
        if self.feat is None:
            self.feat = rules.features(self.data)
        pred = rules.predict(self.data, rule_params, self.feat)
        return score.pm.evaluate(self.truth, score.pm.label_codes(pred))["spell_f1_mean"]


class HeldOut(Game):
    """Sample Game 3, in the EPTS format, kept out of the search to test what it picked."""

    def __init__(self, sigma=0.11):
        root = os.path.join(SAMPLE, "Sample_Game_3", "Sample_Game_3_")
        meta = open(root + "metadata.xml", encoding="utf-8").read()
        owner = dict(re.findall(r'<Player id="(\w+)" teamId="(\w+)"', meta))
        channel = dict(re.findall(r'id="(player\d+)_x" playerId="(\w+)"', meta))
        specs = meta.split("<DataFormatSpecification ")[1:]
        # substitutions change which players each stretch of the file lists
        layouts = [(int(re.search(r'startFrame="(\d+)"', s).group(1)), int(re.search(r'endFrame="(\d+)"', s).group(1)),
                    re.findall(r'playerChannelId="(player\d+)_x"', s)) for s in specs]
        order = list(dict.fromkeys(c for lay in layouts for c in lay[2]))
        slot = {c: i for i, c in enumerate(order)}
        teams = sorted(set(owner.values()))
        team = np.array([teams.index(owner[channel[c]]) for c in order])
        text = open(root + "tracking.txt").read().replace(":", ";")
        rows = [line.split(";") for line in text.splitlines() if line]
        n = max(int(r[0]) for r in rows) + 2
        scale = np.array([LENGTH, WIDTH], np.float32)
        xy = np.full((n, len(order), 2), np.nan, np.float32)
        ball = np.full((n, 2), np.nan, np.float32)
        for r in rows:
            f = int(r[0])
            lay = next(l[2] for l in layouts if l[0] <= f <= l[1])
            cells = [c.split(",") for c in r[1:]]
            pts = np.array(cells[:len(lay)], np.float32)
            xy[f, [slot[c] for c in lay]] = pts * scale
            ball[f] = np.array(cells[len(lay)][:2], np.float32) * scale
        speed = np.full((n, len(order)), np.nan, np.float32)
        speed[2:-2] = np.hypot(*(xy[4:] - xy[:-4]).transpose(2, 0, 1)) * config.FPS / 4
        param = dict(re.findall(r"<Name>(\w+_half_\w+)</Name>\s*<Value>(\d+)</Value>", meta))
        self.periods = {1: (int(param["first_half_start"]), int(param["first_half_end"])),
                        2: (int(param["second_half_start"]), int(param["second_half_end"]))}
        self.n = n
        self.data = {"ball_xy": holes(ball, 3), "players_xy": xy, "players_speed": speed,
                     "team": team, "keeper": np.zeros(len(team), bool)}
        self.ball = jitter(holes(ball, 103), 203, sigma)
        self.feat = None
        events = json.load(open(root + "events.json", encoding="utf-8"))["data"]
        letter = {teams[0]: "H", teams[1]: "A"}
        marks, sets = [], set()
        for e in events:
            kind, start, end = e["type"]["name"], e["start"]["frame"], e["end"]["frame"] or e["start"]["frame"]
            subs = e["subtypes"] or []
            subs = [s["name"] for s in (subs if isinstance(subs, list) else [subs])]
            who = letter.get((e.get("team") or {}).get("id"))
            if kind == "SET PIECE":
                sets.add(start)
            if kind in ("BALL OUT", "FAULT RECEIVED"):
                marks.append((start, "D"))
            elif kind == "SHOT" and ("GOAL" in subs or "OUT" in subs):
                marks.append((start, who))
                marks.append((max(end, start + 1), "D"))
            elif (kind in ON_BALL or kind == "CARRY") and who:
                marks.append((start, who))
        out = np.full(n, "X", dtype="<U1")
        self.kick = {}
        for p, (lo, hi) in self.periods.items():
            inside = sorted(m for m in marks if lo <= m[0] <= hi)
            self.kick[p] = inside[0][0]
            out[lo:self.kick[p]] = "D"
            dead = False
            for k, (f, lab) in enumerate(inside):
                nxt = inside[k + 1][0] if k + 1 < len(inside) else hi + 1
                if lab == "D":
                    dead = True
                elif f in sets:
                    dead = False
                out[f:nxt] = "D" if dead else lab
        self.truth = score.pm.label_codes(out)


IDSSE = os.path.join(config.ROOT, "ext-data", "idsse")


def idsse_matches():
    return sorted(f.split("_DFL-COM-")[1][:-4] for f in os.listdir(IDSSE)
                  if f.startswith("DFL_04_03_positions_raw_observed") and f.endswith(".xml"))


def read_idsse(match):
    """Positions and the tracking provider's own ball status and possession, per frame."""
    import xml.etree.ElementTree as ET

    cache = os.path.join(IDSSE, "cache", match + "_v2.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    info = open(os.path.join(IDSSE, "DFL_02_01_matchinformation_DFL-COM-{}.xml".format(match)), encoding="utf-8").read()
    home = re.search(r'HomeTeamId="([^"]+)"', info).group(1)
    path = os.path.join(IDSSE, "DFL_04_03_positions_raw_observed_DFL-COM-{}.xml".format(match))
    tracks, ball, current = {}, {}, None
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            team, person = el.get("TeamId"), el.get("PersonId")
            current = None if team == "referee" else (team, person, el.get("GameSection"))
        elif event == "end" and el.tag == "Frame" and current is not None:
            n = int(el.get("N"))
            if current[0] == "BALL":
                ball[n] = (float(el.get("X")), float(el.get("Y")), int(el.get("BallStatus")), int(el.get("BallPossession")), float(el.get("Z")))
            else:
                tracks.setdefault((current[0], current[1]), {})[n] = (float(el.get("X")), float(el.get("Y")), float(el.get("S")) / 3.6)
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            el.clear()
    frames = np.array(sorted(ball))
    lo = frames.min()
    n = int(frames.max() - lo) + 2
    keys = sorted(tracks)
    xy = np.full((n, len(keys), 2), np.nan, np.float32)
    speed = np.full((n, len(keys)), np.nan, np.float32)
    for j, k in enumerate(keys):
        f = np.array(list(tracks[k]))
        v = np.array(list(tracks[k].values()), np.float32)
        xy[f - lo, j] = v[:, :2] + [LENGTH / 2, WIDTH / 2]
        speed[f - lo, j] = v[:, 2]
    b = np.array([ball[f] for f in frames], np.float32)
    bxy = np.full((n, 2), np.nan, np.float32)
    bxy[frames - lo] = b[:, :2] + [LENGTH / 2, WIDTH / 2]
    status = np.zeros(n, np.int8)
    status[frames - lo] = b[:, 2]
    owner = np.zeros(n, np.int8)
    owner[frames - lo] = b[:, 3]
    seen = np.zeros(n, bool)
    seen[frames - lo] = True
    height = np.full(n, np.nan, np.float32)
    height[frames - lo] = b[:, 4]
    out = {"xy": xy, "speed": speed, "team": np.array([0 if k[0] == home else 1 for k in keys]),
           "ball": bxy, "status": status, "owner": owner, "seen": seen, "height": height}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


class Idsse(Game):
    """A Bundesliga match from the IDSSE dataset (Bassek et al. 2025, CC BY 4.0). Its TRACAB
    tracking carries the provider's per-frame ball status and owning team, the same kind of
    corrected-tracking flags the organisers' ground truth comes from."""

    def __init__(self, k, sigma=0.11):
        z = read_idsse(idsse_matches()[k - 1])
        n = len(z["status"])
        self.n = n + 2
        pad = lambda a, fill: np.concatenate((a, np.full((2,) + a.shape[1:], fill, a.dtype)))
        xy, speed, ball = pad(z["xy"], np.nan), pad(z["speed"], np.nan), pad(z["ball"], np.nan)
        status, owner, seen = pad(z["status"], 0), pad(z["owner"], 0), pad(z["seen"], False)
        # the halves are the two runs of frames the ball is tracked on; each starts at its kick-off
        edges = np.diff(np.concatenate(([0], seen.astype(np.int8), [0])))
        runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))[:2]
        self.periods = {i + 1: (int(a), int(b)) for i, (a, b) in enumerate(runs)}
        self.kick = {p: int(a + np.argmax(status[a:b + 1] == 1)) for p, (a, b) in self.periods.items()}
        labels = np.full(self.n, "X", dtype="<U1")
        for p, (a, b) in self.periods.items():
            seg = np.where(status[a:b + 1] == 1, np.where(owner[a:b + 1] == 1, "H", "A"), "D")
            labels[a:b + 1] = seg
        self.truth = score.pm.label_codes(labels)
        self.data = {"ball_xy": holes(ball, 300 + k), "players_xy": xy, "players_speed": speed,
                     "team": z["team"], "keeper": np.zeros(len(z["team"]), bool)}
        self.ball = jitter(holes(ball, 400 + k), 500 + k, sigma)
        self.height = pad(z["height"], np.nan)
        self.feat = None


def make(name, sigma):
    """m1, m2: Metrica sample games 1-2; m3: Metrica sample game 3; d1..d7: IDSSE matches."""
    if name in ("m1", "m2"):
        return Game(int(name[1]), sigma)
    if name == "m3":
        return HeldOut(sigma)
    return Idsse(int(name[1:]), sigma)


def total(games, rule_params, touch_params):
    touch.DEFAULTS.update(touch_params)
    return [g.score(rule_params) for g in games]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="score Sample Game 3, not searched")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.11],
                        help="detector errors (m) to search and test over; the search maximises their mean")
    parser.add_argument("--search", nargs="+", default=["m1", "m2"], help="games the search runs on")
    parser.add_argument("--test", nargs="+", default=["m3"], help="held-out games for --holdout")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    if args.holdout:
        held = [make(g, s) for s in args.sigmas for g in args.test]
        t = held[0].truth
        print("H/A/D", [int((t == c).sum()) for c in range(3)], "spells", int(((t[1:] != t[:-1]) & (t[1:] < 2)).sum()))
        got = total(held, {}, {})
        names = ["{}@{}m".format(g, s) for s in args.sigmas for g in args.test]
        print("chosen constants", " ".join("{} {:.4f}".format(a, v) for a, v in zip(names, got)), "mean {:.4f}".format(np.mean(got)))
        mid_r = {k: v[len(v) // 2] for k, v in RULE_GRID.items()}
        mid_t = {k: v[len(v) // 2] for k, v in TOUCH_GRID.items()}
        print("middle of grid   mean {:.4f}".format(np.mean(total(held, mid_r, mid_t))))
        sys.exit()
    games = [make(g, s) for s in args.sigmas for g in args.search]
    if args.check:
        for g in games:
            t = g.truth
            print("frames", g.n, "H/A/D", [int((t == c).sum()) for c in range(3)],
                  "spells", int(((t[1:] != t[:-1]) & (t[1:] < 2)).sum()),
                  "ball seen in play %.1f%%" % (100 * (~np.isnan(g.ball[t < 2, 0])).mean()),
                  "ball seen dead %.1f%%" % (100 * (~np.isnan(g.ball[t == 2, 0])).mean()))
        print("current constants", total(games, {}, {}))
        sys.exit()
    rp = {k: v[len(v) // 2] for k, v in RULE_GRID.items()}
    tp = {k: v[len(v) // 2] for k, v in TOUCH_GRID.items()}
    best = float(np.mean(total(games, rp, tp)))
    print("start", round(best, 4), flush=True)
    for sweep in range(2):
        changed = False
        for grid, params in ((RULE_GRID, rp), (TOUCH_GRID, tp)):
            for key, values in grid.items():
                for value in values:
                    if value == params[key]:
                        continue
                    old = params[key]
                    params[key] = value
                    s = float(np.mean(total(games, rp, tp)))
                    if s > best + 1e-9:
                        best, changed = s, True
                        print("{} = {} -> {:.4f}".format(key, value, s), flush=True)
                    else:
                        params[key] = old
        if not changed:
            break
    print("final", total(games, rp, tp))
    print(json.dumps({"rules": rp, "touch": tp}, indent=1))
