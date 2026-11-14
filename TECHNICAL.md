# P-Value FC: technical document

More than a Hack 2027, Phase 1. Labels every frame of the BAR-ATH tactical video as H, A, D or X.

## Approach

Rules over the automatic tracking for who has the ball, one signal from the video for touches, and
two classifiers trained on other matches: one for when the ball is dead, one for which team has it
where the rules cannot tell. Possession is mostly a matter of convention (when a switch counts, when
a stoppage starts), which rules can state directly and a reader can check. Dead ball, and who owns a
ball nobody is touching, are patterns in how 22 players move, which a model learns better than a
threshold, and public matches with the tracking provider's own flags exist to learn them from.
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
  (YOLOv8, MIT, trained on other footage) finds the ball frame by frame at 1920 px. The tracking
  ball is smoothed and hides touches, the raw detections show them as kinks.
- **Metrica Sports public sample data** (github.com/metrica-sports/sample-data). Three matches with
  tracking and events. Used to select every searched constant. Thanks to Metrica for publishing it.
- **IDSSE Bundesliga data** (Bassek, Rein, Weber and Memmert, Scientific Data 12, 195, 2025;
  CC BY 4.0, via huggingface.co/datasets/pysport/idsse-data). Seven matches of official TRACAB
  tracking with the provider's ball status, owning team and ball height on every frame. Used to
  train both classifiers and as extra test matches.
- **Roboflow football-ball-detection test split** (CC BY 4.0, via huggingface.co/datasets/
  martinjolif/football-ball-detection). Used to choose the detector's input size (the ball found
  on 114 of 125 images at 1920 px, 99 at 1280, two false boxes at both) and to measure how far its
  box lands from the ball.

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
- **Dead-ball classifier with more data and the players' shape** (share of players in the boxes,
  near the ball, far from it, their spread), adding Metrica games 1 and 2 to the training matches.
  +0.003 on the test matches, under the threshold.
- **Constants searched at the detector's measured error.** On the external test split the box lands
  about 5% of a ball's width from the labelled ball, at most 0.03 m on the pitch. Constants searched
  at 0.03 m gained +0.028 at that error but lost 0.021 at 0.11 m, and Metrica game 3 fell by
  0.055: they only hold if nothing but the detector adds error, which on BAR-ATH the camera and
  lofted balls do. Searching at 0.03 and 0.11 m together gained +0.001 and dropped Metrica game 3
  by 0.022.
- **Constants searched again with the possession classifier on.** -0.006 on the test matches.
- **More uses of the possession classifier.** Letting it overrule the control test, using it alone,
  or letting it pick the restart team did no better on Metrica games 1 and 2, where such choices are
  made, so none reached the test matches. Training it on Metrica games 1 and 2 as well gained
  +0.002. Feeding its doubt to the dead-ball classifier gained +0.006 (+0.003, +0.002, +0.011),
  positive on all three but under the threshold, so it was dropped too.
- **Public player detector with ByteTrack.** Positions agree with the tracking, but it has no teams
  and its identities break every few seconds.

## Possession rules

- **Switch.** The other team must show control for `confirm` frames: a player within `reach` of a
  ball slower than `max_speed`, with the nearest opponent `margin` further away. The switch is then
  dated back to the first of those frames.
- **Touches.** A kink in the video ball with one team clearly nearest is a touch by that team. A
  switch moves back to such a touch, up to `back` seconds. A sharp turn of the ball by the team not
  in possession starts a possession, which ends when the old team touches or shows control again.
- **Frames the control test leaves open.** `mtah/possmodel.py`: a gradient-boosted classifier,
  trained on the owning team the Bundesliga tracking flags on every live frame, scores each frame
  from both teams' distance to the tracking ball and to the video ball, the ball's speed, which team
  was close over 0.5, 1, 2 and 4 s before and after, and how both teams stand: the side with the
  ball spreads out and moves towards the other. Where it is at least 90% sure (`poss_thr`, picked
  on Metrica games 1 and 2, which it was not trained on) the frame counts as control by that team,
  the same as a frame that passes the test above.
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
`tools/train_dead.py` and `tools/train_poss.py` train the two classifiers on Bundesliga matches 1
to 5.

| Test match | Rules only | + dead-ball classifier | + possession, distances | + team shape |
|---|---|---|---|---|
| Metrica 3 | 0.575 | 0.575 | 0.604 | 0.623 |
| Bundesliga 6 | 0.524 | 0.601 | 0.633 | 0.658 |
| Bundesliga 7 | 0.519 | 0.565 | 0.600 | 0.663 |

BAR-ATH's tracking has about 9 of 11 players or fewer on one frame in ten, while the test matches
are complete, so the team-shape features were also checked with players punched out of the test
matches (in 1-10 s stretches, until about half of all player frames were missing). They still beat
the distance-only model on all three (+0.034, +0.003, +0.034).

For reference, Metrica 3 with the grid's middle values scores 0.454, and with constants picked
without the simulated ball error 0.525. The submitted file scores 0.556 on the BAR-ATH public five
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
- The tracking misses the ball on 41% of live frames and the video ball path on 24% (42% at the
  earlier 1280 px). Both miss it on 15%.
- A single camera puts an airborne ball on the ground where it covers the grass, away from the
  camera, so a lofted ball can look as if it sits at an opponent's feet. Neither external dataset
  has that error, so nothing here was trained to handle it.
- The dead-ball classifier separates dead from live frames far better on the Bundesliga test
  matches (AUC 0.99) than on Metrica game 3 (0.81), and only the Bundesliga matches gained from it.
- The sample games have no video, so the touch constants were selected on their tracking ball with a
  simulated detector error, which only approximates the detector's path.
- The ball path from the video (`mtah/balltrack.py`) has a few constants that were set once when
  the module was written and never searched. The physical ones were checked against external data
  afterwards. A top speed of 40 m/s is above a full-power instep kick by elite players (mean 104
  km/h, 29 m/s, by radar; Sporis, Vucetic and Jukic, Journal of Sports Science and Medicine 6,
  Suppl. 10, 2007) and above 99.9% of ball speeds in the IDSSE tracking (31.6 m/s). A step of about
  1 m per frame is the 99th percentile there (0.98 m). The rest (a gap of up to 12 frames, 0.5 s,
  can be bridged; boxes under 0.1 confidence are ignored; a weak box costs more than a confident
  one) could not be checked, because no external dataset has video to run the detector on.
- With more time: a detector made for small balls, a whistle detector trained only on external
  audio, and an alignment of the broadcast tracking to fill ball holes.

## Reproduction

See `README.md`. With the data bundle next to this repository:

    pip install -r requirements.txt
    python tools/detect_ball.py --all    # once, GPU, see README
    python run_rules.py

This writes `submissions/submission.csv`, md5 `703e1e5371c961b7f8f5269cad595944`. Two runs from an
empty cache (keeping the ball detections) give the same file. The trained classifiers are in
`models/dead_ball.joblib` and `models/possession.joblib`; `python tools/train_dead.py` and
`python tools/train_poss.py` rebuild them from the IDSSE data and give the same file again. The
ball detections are frame-level data derived from the video, so they are not in the public
repository. We can provide our copy to the organisers on request.
