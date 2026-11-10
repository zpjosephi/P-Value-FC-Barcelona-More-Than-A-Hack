# P-Value: More than a Hack 2027, Phase 1

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
    python run_rules.py

The result is `submissions/submission.csv`. It is deterministic: two runs from an empty cache give
the same file. The script also prints the score on the public 5 minutes.

## How it works

`mtah/rules.py` reads the automatic tracking and holds possession with the last team until the
other team shows control. Dead ball comes from the ball leaving the pitch or the fastest players
slowing down together, and ends at the restart kick.

`mtah/touch.py` uses a ball path from the video to find touches. It moves a switch back to the
first touch, and a sharp turn of the ball by the team not in possession starts a possession.

## Intermediate files

`cache/ball_det/` holds the per-frame ball boxes from `tools/detect_ball.py`, so the GPU step
(about 11 hours on an RTX 2060) does not have to be repeated. The detector is
`football-ball-detection.pt` from github.com/roboflow/sports (MIT), trained on other footage. To
redo it, download the model to `cache/models/`, install torch and ultralytics, then run
`python tools/detect_ball.py --all`.

## Where the constants come from

The organisers' clarification of November 2026 forbids tuning on the ground truth or on the
provided video, and forbids hand labels on this match. This code was started before that email.
What we changed because of it:

- A whistle detector trained on whistles we marked by ear was removed.
- Hand-labelled clips are no longer used for anything and are not in this repo.

What is still open: the numeric constants in `DEFAULTS` (`mtah/rules.py`, `mtah/touch.py`) were
chosen earlier while looking at scores on the public ground truth and on our hand-labelled clips.
They have not yet been re-selected on external data.
