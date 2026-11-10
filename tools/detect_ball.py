"""Find the ball in the tactical video with a pre-trained detector, frame by frame, on the GPU.

The tracking loses the ball on 41% of the graded frames and smooths the rest. The detector looks
at the picture itself, so a touch shows up as a kink in the ball path.

    .venv/Scripts/python tools/detect_ball.py 11185 18684          # public window
    .venv/Scripts/python tools/detect_ball.py --all                 # both halves, hours

Model: football-ball-detection.pt from github.com/roboflow/sports (MIT), YOLOv8 at 1280 px,
downloaded once to cache/models/ (see README). Output: cache/ball_det/<lo>-<hi>.npz with up to
TOP boxes per frame in pixels, ordered by confidence: frame, cx, cy, w, h, conf. Projection to
the pitch happens later in mtah/balldet.py, which needs no GPU.

Runs in .venv (torch + ultralytics), not in the plain Python the rest of work/ uses. Inference
is FP32 with cuDNN in deterministic mode so a rerun gives the same boxes (RULES.md, rule 3).
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mtah import config  # noqa: E402

MODEL = os.path.join(config.CACHE_DIR, "models", "football-ball-detection.pt")
OUT_DIR = os.path.join(config.CACHE_DIR, "ball_det")
TOP = 5


def video_file():
    folder = os.path.dirname(config.asset("atd_raw"))
    return os.path.join(folder, sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))[0])


def frames(lo, hi):
    """(frame index, RGB array) for lo..hi, index from the timestamp like make_clip.py."""
    import av

    src = av.open(video_file())
    stream = src.streams.video[0]
    stream.thread_type = "AUTO"
    src.seek(int(max(0, lo - 2 * config.FPS) / config.FPS / stream.time_base), stream=stream, backward=True)
    for frame in src.decode(stream):
        idx = int(round(float(frame.pts * stream.time_base) * config.FPS))
        if idx < lo:
            continue
        if idx > hi:
            break
        yield idx, frame.to_ndarray(format="bgr24")
    src.close()


def detect(lo, hi, batch, conf):
    import torch
    from ultralytics import YOLO

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = YOLO(MODEL)
    rows = []
    buf, idxs = [], []
    t0 = time.time()

    def flush():
        results = model.predict(buf, imgsz=1280, conf=conf, half=False, verbose=False, max_det=TOP)
        for idx, r in zip(idxs, results):
            boxes = r.boxes
            if len(boxes) == 0:
                continue
            xywh = boxes.xywh.cpu().numpy()
            c = boxes.conf.cpu().numpy()
            for k in np.argsort(-c)[:TOP]:
                rows.append((idx, *xywh[k], c[k]))
        buf.clear()
        idxs.clear()

    for idx, img in frames(lo, hi):
        buf.append(img)
        idxs.append(idx)
        if len(buf) == batch:
            flush()
            done = idx - lo + 1
            if done % (batch * 250) < batch:
                rate = done / (time.time() - t0)
                print("frame {}  {:.1f} fps  eta {:.0f} min".format(idx, rate, (hi - idx) / rate / 60), flush=True)
    if buf:
        flush()
    arr = np.array(rows, dtype=np.float32).reshape(-1, 6)
    return arr, time.time() - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", type=int, nargs="?")
    parser.add_argument("end", type=int, nargs="?")
    parser.add_argument("--all", action="store_true", help="both halves, kickoff to end")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.05, help="keep weak boxes, filtering comes later")
    args = parser.parse_args()

    if args.all:
        spans = [(config.HALVES[h][0], config.HALVES[h][1]) for h in (1, 2)]
    elif args.start is not None and args.end is not None:
        spans = [(args.start, args.end)]
    else:
        parser.error("give start end, or --all")

    os.makedirs(OUT_DIR, exist_ok=True)
    for lo, hi in spans:
        out = os.path.join(OUT_DIR, "{}-{}.npz".format(lo, hi))
        arr, secs = detect(lo, hi, args.batch, args.conf)
        np.savez_compressed(out, det=arr, lo=lo, hi=hi)
        n_frames = len(np.unique(arr[:, 0])) if len(arr) else 0
        print("{}: {} boxes on {} of {} frames, {:.1f} min".format(
            os.path.relpath(out), len(arr), n_frames, hi - lo + 1, secs / 60))
