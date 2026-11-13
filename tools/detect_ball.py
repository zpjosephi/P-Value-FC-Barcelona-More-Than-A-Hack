"""Find the ball in the tactical video with a pre-trained detector, frame by frame, on the GPU.

The tracking loses the ball on 41% of the graded frames and smooths the rest. The detector looks
at the picture itself, so a touch shows up as a kink in the ball path.

    python tools/detect_ball.py --all                 # both halves, resumable
    python tools/detect_ball.py --all --imgsz 1280    # the earlier, faster setting
    python tools/detect_ball.py 11185 18684           # one stretch, for a quick look

Model: football-ball-detection.pt from github.com/roboflow/sports (MIT), YOLOv8, trained on other
footage. Input size: on the test split of the public football-ball-detection dataset (CC BY 4.0,
Roboflow Universe) it found the ball on 114 of 125 images at 1920 px against 99 at 1280, with the
same two false boxes. Tiling into four crops found 112 with six false boxes at twice the cost.

Output: cache/ball_det_<imgsz>/<lo>-<hi>.npz, one file per 100 s, up to TOP boxes per frame in
pixels ordered by confidence: frame, cx, cy, w, h, conf. Finished pieces are skipped, so a run
that stops can be started again. FP32 with deterministic cuDNN so a rerun gives the same boxes.
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mtah import config  # noqa: E402

MODEL = os.path.join(config.CACHE_DIR, "models", "football-ball-detection.pt")
PIECE = 2500
TOP = 5


def video_file():
    folder = os.path.dirname(config.asset("atd_raw"))
    return os.path.join(folder, sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))[0])


def frames(lo, hi):
    """(frame index, BGR array) for lo..hi, index from the timestamp."""
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


def keep_awake(on):
    # Windows would sleep through an overnight run; the setting ends with the process anyway
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def split(spans):
    return [(a, min(hi, a + PIECE - 1)) for lo, hi in spans for a in range(lo, hi + 1, PIECE)]


def run(todo, out_dir, imgsz, batch, conf):
    import torch
    from ultralytics import YOLO

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = YOLO(MODEL)
    groups = []
    for p in todo:
        if groups and groups[-1][-1][1] + 1 == p[0]:
            groups[-1].append(p)
        else:
            groups.append([p])
    done, t0, left = 0, time.time(), sum(hi - lo + 1 for lo, hi in todo)
    for group in groups:
        queue = list(group)
        rows, buf, idxs = [], [], []

        def flush():
            for idx, r in zip(idxs, model.predict(buf, imgsz=imgsz, conf=conf, half=False, verbose=False, max_det=TOP)):
                if len(r.boxes):
                    xywh, c = r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()
                    rows.extend((idx, *xywh[k], c[k]) for k in np.argsort(-c)[:TOP])
            buf.clear()
            idxs.clear()

        for idx, img in frames(group[0][0], group[-1][1]):
            buf.append(img)
            idxs.append(idx)
            if len(buf) == batch:
                flush()
            if idx >= queue[0][1]:
                if buf:
                    flush()
                lo, hi = queue.pop(0)
                arr = np.array([r for r in rows if lo <= r[0] <= hi], dtype=np.float32).reshape(-1, 6)
                path = os.path.join(out_dir, "{}-{}.npz".format(lo, hi))
                np.savez_compressed(path + ".tmp.npz", det=arr, lo=lo, hi=hi, imgsz=imgsz)
                os.replace(path + ".tmp.npz", path)
                rows[:] = [r for r in rows if r[0] > hi]
                done += hi - lo + 1
                rate = done / (time.time() - t0)
                print("{} {}-{} done, {:.1f} frames/s, about {:.0f} min left".format(
                    time.strftime("%H:%M"), lo, hi, rate, (left - done) / rate / 60), flush=True)
                if not queue:
                    break
        for lo, hi in queue:
            print("video ended before frame {}, piece {}-{} not saved".format(hi, lo, hi), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("start", type=int, nargs="?")
    parser.add_argument("end", type=int, nargs="?")
    parser.add_argument("--all", action="store_true", help="both halves, kickoff to end")
    parser.add_argument("--imgsz", type=int, default=config.BALL_IMGSZ)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.05, help="keep weak boxes, filtering comes later")
    args = parser.parse_args()

    if args.all:
        spans = [config.HALVES[1], config.HALVES[2]]
    elif args.start is not None and args.end is not None:
        spans = [(args.start, args.end)]
    else:
        parser.error("give start end, or --all")
    out_dir = config.ball_det_dir(args.imgsz)
    os.makedirs(out_dir, exist_ok=True)
    todo = [p for p in split(spans) if not os.path.exists(os.path.join(out_dir, "{}-{}.npz".format(*p)))]
    print("{} of {} pieces to do, {} px, into {}".format(len(todo), len(split(spans)), args.imgsz, out_dir), flush=True)
    keep_awake(True)
    try:
        run(todo, out_dir, args.imgsz, args.batch, args.conf)
    finally:
        keep_awake(False)
