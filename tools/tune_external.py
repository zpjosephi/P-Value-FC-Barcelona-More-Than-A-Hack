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


class Game:
    def __init__(self, number):
        self.data, self.periods, self.n = load(number)
        labels, self.kick = truth(number, self.periods, self.n)
        self.truth = score.pm.label_codes(labels)
        # the sample tracking sees the ball far more often than BAR-ATH's automatic tracking (about
        # 90% against 59% of live frames), so holes are punched to match; the video ball gets its own
        full = self.data["ball_xy"]
        self.data["ball_xy"] = holes(full, number)
        self.ball = holes(full, number + 100).astype(float)
        self.feat = None

    def score(self, rule_params):
        config.N_FRAMES, config.PITCH_LENGTH, config.PITCH_WIDTH = self.n, LENGTH, WIDTH
        config.HALVES, config.KICKOFF = self.periods, self.kick
        touch.video_ball = lambda: self.ball
        if self.feat is None:
            self.feat = rules.features(self.data)
        pred = rules.predict(self.data, rule_params, self.feat)
        return score.pm.evaluate(self.truth, score.pm.label_codes(pred))["spell_f1_mean"]


def total(games, rule_params, touch_params):
    touch.DEFAULTS.update(touch_params)
    return [g.score(rule_params) for g in games]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    games = [Game(1), Game(2)]
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
