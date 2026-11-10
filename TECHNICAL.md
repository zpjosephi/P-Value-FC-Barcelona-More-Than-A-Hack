# P-Value FC: technical document

More than a Hack 2027, Phase 1. Labels every frame of the BAR-ATH tactical video as H, A, D or X.

## Approach

A rule model over the automatic tracking, plus one signal from the video. We chose rules over a
learned model for two reasons. The only possession labels on this match are the public 5 minutes,
and they may not be used for training. And the task is mostly about conventions (when a switch
counts, when a stoppage starts), which rules can state directly and a reader can check.

The rules hold possession with the last team until the other team shows control, so a lost ball,
a ball flying past a player or a brief overlap changes nothing. spellF1 punishes a possession
split in two harder than a switch that lands a little late, so everything that could create a
false switch or a false stoppage is deliberately restricted.

## Assets

Used:

- **Tactical automatic tracking.** Player and ball positions, player speeds, team of each slot.
- **Tactical homography.** Projects detected ball boxes onto the pitch.
- **Tactical video, as input only.** `football-ball-detection.pt` from github.com/roboflow/sports
  (YOLOv8, MIT, trained on other footage) finds the ball frame by frame. The tracking ball is
  smoothed and hides touches, the raw detections show them as kinks.
- **Metrica Sports public sample data** (github.com/metrica-sports/sample-data). External data,
  used to select every searched constant. Thanks to Metrica for publishing it.

Tried and dropped:

- **Whistle detector from the match sound.** It was trained on whistles we marked by ear, which the
  organisers' clarification forbids, so it was removed.
- **Smart tagging.** The model already has a stoppage within 10 s of 53 of the 58 set-piece tags,
  and the tags have no external counterpart to set a rule on.
- **Broadcast tracking.** It sees the ball more often while the camera is wide, but it has cuts and
  replays and no frame mapping to the tactical video.
- **Filling the tracking's ball holes from the video.** No gain.
- **Jersey colour to catch swapped tracks.** Shadows made it discard correct points.
- **Public player detector with ByteTrack.** Positions agree with the tracking, but it has no teams
  and its identities break every few seconds.

## Possession rules

- **Switch.** The other team must show control for `confirm` frames: a player within `reach` of a
  ball slower than `max_speed`, with the nearest opponent `margin` further away. The switch is then
  dated back to the first of those frames.
- **Touches.** A kink in the video ball with one team clearly nearest is a touch by that team. A
  switch moves back to such a touch, up to `back` seconds. A sharp turn of the ball by the team not
  in possession starts a possession, which ends when the old team touches or shows control again.
- **Loose balls and challenges.** Possession stays with the last team until the above is met.
- **Dead ball.** The tracked ball beyond a line, or the fastest quarter of the players slowing
  together for `slow_for` seconds, unless a player is still on a ball the tracking sees.
- **Restarts.** A dead ball ends when a ball that sat still is struck, and the nearest team takes
  it. A strike nobody gets going after is a hand-back and the stoppage continues.
- **Bad tracks.** A keeper's point more than 50 m from his own goal is discarded.

## How the constants were chosen

`tools/tune_external.py` runs coordinate descent on Sample Games 1 and 2. Labels there are derived
from the games' own event data, and the search starts from the middle of each grid. Random holes
are punched in the sample ball, since its tracking is cleaner than BAR-ATH's. Sample Game 3 was
kept out of the search as a test.

| Game | spellF1 |
|---|---|
| Sample 1 (searched) | 0.595 |
| Sample 2 (searched) | 0.591 |
| Sample 3 (held out) | 0.577 |
| Sample 3 with the grid's middle values | 0.448 |
| BAR-ATH public 5 minutes (looked at once, not used to choose) | 0.571 |

## History

This project began before the organisers' clarification, under documentation that allowed hand
labels for validation. Earlier versions used team-made labels on this match and chose constants
while looking at scores on them and on the public ground truth. After the clarification we removed
the whistle detector, stopped using those labels, and re-selected every searched constant on the
external data above. The shape of the rules came from watching the match and studying the public
ground truth, including where earlier versions went wrong on it. An AI coding assistant helped
write the code. No AI model was trained on the match data.

## Limitations

- Short contested possessions, under 5 s, are the main weakness. In headers and clearances the ball
  is too fast for the control test, and it is often missing from both the tracking and the video.
- The tracking misses the ball on 41% of live frames and the detector on 42%.
- Dead-ball detection got worse with the external constants (public F1 for D 0.59), because the
  sample games are event-labelled and stop differently.
- The sample games have no video, so the touch constants were selected on their tracking ball, a
  different signal from the detector's path.
- With more time: a detector made for small balls, a whistle detector trained only on external
  audio, and an alignment of the broadcast tracking to fill ball holes.

## Reproduction

See `README.md`. With the data bundle next to this repository:

    pip install -r requirements.txt
    python tools/detect_ball.py --all    # once, GPU, about 11 hours, see README
    python run_rules.py

This writes `submissions/submission.csv`, md5 `e397d8b04bb2aaf8129b742a72f5c360`. Two runs from an
empty cache (keeping the ball detections) give the same file. The detections are frame-level data
derived from the video, so they are not in the public repository. We can provide our copy to the
organisers on request.
