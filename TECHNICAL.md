# P-Value FC: technical document

More than a Hack 2027, Phase 1. Labels every frame of the BAR-ATH tactical video as H, A, D or X.

## Approach

Rules over the automatic tracking for who has the ball, one signal from the video for touches, and
one classifier, trained on other matches, for when the ball is dead. Possession is mostly a matter
of convention (when a switch counts, when a stoppage starts), which rules can state directly and a
reader can check. Dead ball is a pattern in how 22 players move, which a model learns better than a
threshold, and public matches with the tracking provider's own ball status exist to learn it from.
Nothing is trained or tuned on BAR-ATH.

The rules hold possession with the last team until the other team shows control, so a lost ball,
a ball flying past a player or a brief overlap changes nothing. spellF1 punishes a possession
split in two harder than a switch that lands a little late, so everything that could create a
false switch is deliberately restricted.

## Assets

Used:

- **Tactical automatic tracking.** Player and ball positions, player speeds, team of each slot.
- **Tactical homography.** Projects detected ball boxes onto the pitch.
- **Tactical video, as input only.** `football-ball-detection.pt` from github.com/roboflow/sports
  (YOLOv8, MIT, trained on other footage) finds the ball frame by frame. The tracking ball is
  smoothed and hides touches, the raw detections show them as kinks.
- **Metrica Sports public sample data** (github.com/metrica-sports/sample-data). Three matches with
  tracking and events. Used to select every searched constant. Thanks to Metrica for publishing it.
- **IDSSE Bundesliga data** (Bassek, Rein, Weber and Memmert, Scientific Data 12, 195, 2025;
  CC BY 4.0, via huggingface.co/datasets/pysport/idsse-data). Seven matches of official TRACAB
  tracking with the provider's ball status, owning team and ball height on every frame. Used to
  train the dead-ball classifier and as extra test matches.

Tried and dropped:

- **Whistle detector from the match sound.** It was trained on whistles we marked by ear, which the
  organisers' clarification forbids, so it was removed.
- **Smart tagging.** The rule model already had a stoppage within 10 s of 53 of the 58 set-piece
  tags, and the tags have no external counterpart to set a rule on.
- **Broadcast tracking.** It sees the ball more often while the camera is wide, but it has cuts and
  replays and no frame mapping to the tactical video.
- **Filling the tracking's ball holes from the video.** No gain.
- **Jersey colour to catch swapped tracks.** Shadows made it discard correct points.
- **A classifier deciding which touches start a possession**, in place of the fixed angle test.
  Trained on 4,359 touches from seven matches, it lost on all three test matches (-0.029, -0.050,
  -0.005).
- **A classifier for "ball in the air"**, trained on the Bundesliga ball height from the 2D path
  only. It finds airborne balls well (AUC 0.95 on held-out matches) but changed nothing (+0.002 at
  most): the control test already ignores a fast ball, and an airborne one nearly always is.
- **Constants searched on all seven training matches.** Better on the Bundesliga test matches
  (+0.050, +0.041) but worse on Metrica game 3 (-0.027), the test match whose automatic tracking is
  closest to BAR-ATH's, so they were not used.
- **A pass to the taker is not the restart**: a strike after which the ball soon sits still again.
  The search on the sample games switched it off.
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
- **Dead ball.** `mtah/deadmodel.py`: a gradient-boosted classifier scores every frame from the
  players' pace (median and fastest quarter), the ball's speed, whether it is seen, how far it is
  outside the lines and how close the nearest player is, each over 1, 2 and 4 s before and after.
  The score is smoothed over a second; above 0.5, in runs of at least 2 s, the ball is dead. The
  earlier rule mask (ball beyond a line, players slowing together) is still in `rules.dead_ball`
  and comes back with `--set dead_model=0`.
- **Restarts.** After a dead stretch the team nearest the ball takes it, since the restart taker
  nearly always is.
- **Bad tracks.** A keeper's point more than 50 m from his own goal is discarded.

## How the choices were made

Three matches were kept out of every search and every training run: Metrica game 3 and Bundesliga
matches 6 and 7. A change was used only if, on those three, the mean spellF1 rose by at least 0.01
and no match fell by more than 0.02, with the threshold fixed before the test was run.

`tools/tune_external.py` runs coordinate descent on Metrica games 1 and 2. Labels there are derived
from the games' own event data, and the search starts from the middle of each grid. Random holes
are punched in the sample ball, since its tracking is cleaner than BAR-ATH's. The ball used for
touches also gets a random error of 0.11 m on every frame, about half a ball's diameter, because the
detector finds a box centre anew on each frame and is not smooth like the sample tracking.
`tools/train_dead.py` trains the dead-ball classifier on Bundesliga matches 1 to 5.

| Test match | Rules only | With the dead-ball classifier |
|---|---|---|
| Metrica 3 | 0.575 | 0.575 |
| Bundesliga 6 | 0.524 | 0.601 |
| Bundesliga 7 | 0.519 | 0.565 |

For reference, Metrica 3 with the grid's middle values scores 0.454, and with constants picked
without the simulated ball error 0.525. The submitted file scores 0.509 on the BAR-ATH public five
minutes; that number was looked at, never used to choose.

## History

This project began before the organisers' clarification, under documentation that allowed hand
labels for validation. Earlier versions used team-made labels on this match and chose constants
while looking at scores on them and on the public ground truth. After the clarification we removed
the whistle detector, stopped using those labels, and re-selected every searched constant on the
external data above. The shape of the rules came from watching the match and studying the public
ground truth, including where earlier versions went wrong on it. Two late changes came from watching
our output on the public minutes against the video: the simulated ball error, and the
pass-to-the-taker test, which the search then switched off. An AI coding assistant helped write the
code. No AI model was trained on the match data.

## Limitations

- Short contested possessions, under 5 s, are the main weakness. In headers and clearances the ball
  is too fast for the control test, and it is often missing from both the tracking and the video.
- The tracking misses the ball on 41% of live frames and the detector on 42%.
- A single camera puts an airborne ball on the ground where it covers the grass, away from the
  camera, so a lofted ball can look as if it sits at an opponent's feet. Neither external dataset
  has that error, so nothing here was trained to handle it.
- The dead-ball classifier separates dead from live frames far better on the Bundesliga test
  matches (AUC 0.99) than on Metrica game 3 (0.81), and only the Bundesliga matches gained from it.
- The sample games have no video, so the touch constants were selected on their tracking ball with a
  simulated detector error, which only approximates the detector's path.
- The constants of the ball path (`mtah/balltrack.py`: cost of a weak box, step size, top speed
  40 m/s, gaps of up to 12 frames bridged) were set by hand early on, while working on this match,
  and were not re-selected: no external dataset has video to select them on. Most are physical
  bounds, but they are the one set of constants not chosen on external data.
- With more time: a detector made for small balls, a whistle detector trained only on external
  audio, and an alignment of the broadcast tracking to fill ball holes.

## Reproduction

See `README.md`. With the data bundle next to this repository:

    pip install -r requirements.txt
    python tools/detect_ball.py --all    # once, GPU, see README
    python run_rules.py

This writes `submissions/submission.csv`, md5 `6a0598331be4d6ac5489b0b9c9fd7f41`. Two runs from an
empty cache (keeping the ball detections) give the same file. The trained classifier is in
`models/dead_ball.joblib`; `python tools/train_dead.py` rebuilds it from the IDSSE data and gives
the same file again. The ball detections are frame-level data derived from the video, so they are
not in the public repository. We can provide our copy to the organisers on request.
