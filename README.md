# P-Value FC: More than a Hack 2027, Phase 1

Labels every frame of the BAR-ATH tactical video as H, A, D or X. Rule-based, no model is trained
in this repository.

## Run

The match data is not in this repo. Put this folder next to the organisers' folders, or set
`MTAH_DATA_DIR`:

    hackaton/
      MoreThanAHack/{Tactical,Broadcast}/
      more-than-a-hack-2027-GitHub/
      P-Value-FC-Barcelona-More-Than-A-Hack/

    pip install -r requirements.txt
    # make the ball detections first, see below
    python run_rules.py

The result is `submissions/submission.csv`. It is deterministic: two runs from an empty cache give
the same file. The script also prints the score on the public 5 minutes.

## How it works

`mtah/rules.py` reads the automatic tracking and holds possession with the last team until the
other team shows control. Dead ball comes from the ball leaving the pitch or the fastest players
slowing down together, and ends at the restart kick.

`mtah/touch.py` uses a ball path from the video to find touches. It moves a switch back to the
first touch, and a sharp turn of the ball by the team not in possession starts a possession.

## Ball detections (run once, needs a GPU)

`run_rules.py` reads per-frame ball boxes from `cache/ball_det/`. They are frame-level data derived
from the match video, so they are not published here. Make them once:

    pip install -r requirements-gpu.txt     # torch built for CUDA 12.1
    # download football-ball-detection.pt from github.com/roboflow/sports to cache/models/
    python tools/detect_ball.py --all       # about 11 hours on an RTX 2060

The detector is YOLOv8 (MIT), trained on other footage, run in FP32 with deterministic cuDNN. Boxes
from an RTX 2060 and an RTX 3050 agreed to 0.0004 px. The organisers can have our copy of these
intermediate files on request.

## Where the constants come from

The numeric constants in `DEFAULTS` (`mtah/rules.py`, `mtah/touch.py`) were selected by
`tools/tune_external.py` on Sample Games 1 and 2 of Metrica Sports' public sample data
(github.com/metrica-sports/sample-data, used with acknowledgement as its README asks). Possession
labels there are derived from the games' own event data. The search starts from the middle of each
grid and does not use BAR-ATH. Sample Game 3 is held out (`--holdout`). Random holes are punched in the sample ball because its tracking is
cleaner than BAR-ATH's. `gk_far` (a keeper more than 50 m from his own goal is a bad track) and
`veto_seen` were not searched.

The sample games have no video, so the touch constants were selected on their tracking ball, which
is not the same signal as the detector's ball path on BAR-ATH.

History, stated plainly. The organisers' clarification of November 2026 forbids tuning on the
ground truth or on the provided video, and forbids hand labels on this match. This project started
before that email, and earlier versions did those things. Because of it:

- A whistle detector trained on whistles we marked by ear was removed.
- Hand-labelled clips are no longer used for anything and are not in this repo.
- All constants above were re-selected on external data, replacing the earlier values.

The shape of the rules (switch on the first touch, a loose ball stays with the last team, restart
to the kicker, a ball handed back is not a restart) came from watching the match and studying the
public ground truth, including where earlier versions went wrong on it.
