"""Paths and match constants shared by everything in work/.

The data bundle sits next to this folder as ../MoreThanAHack (Tactical/ and Broadcast/), not in
the data/tactical layout the official repo expects. MTAH_DATA_DIR overrides it, same env var the
official tooling uses, so one setting serves both.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
ROOT = os.path.dirname(WORK)

DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "MoreThanAHack")
OFFICIAL_DIR = os.path.join(ROOT, "more-than-a-hack-2027-GitHub")
CACHE_DIR = os.path.join(WORK, "cache")
SUBMISSIONS_DIR = os.path.join(WORK, "submissions")
FRAMES_DIR = os.path.join(WORK, "frames")

PUBLIC_GT = os.path.join(OFFICIAL_DIR, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

N_FRAMES = 199319
FPS = 25
PITCH_LENGTH, PITCH_WIDTH = 105.3, 68.0

# video frames, inclusive. Graded blocks and the actual kickoffs (2nd half is graded from
# 110808 but the ball is only struck at 113742, everything in between is D).
HALVES = {1: (11185, 90651), 2: (110808, 190197)}
KICKOFF = {1: 11185, 2: 113742}
PUBLIC_WINDOW = (11185, 18684)

SUFFIX = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}


def _source_dir(source):
    # the bundle arrived as Tactical/ and Broadcast/ but the official docs say lowercase;
    # Windows does not care, Linux and macOS do
    for name in (source.capitalize(), source.lower()):
        path = os.path.join(DATA_DIR, name)
        if os.path.isdir(path):
            return path
    raise IOError("no {} folder under {} (set MTAH_DATA_DIR)".format(source, DATA_DIR))


def asset(name, source="tactical"):
    path = os.path.join(_source_dir(source), "BAR-ATH_" + source + SUFFIX[name])
    if not os.path.isfile(path):
        raise IOError("missing data file: " + path)
    return path


def half_of(frame):
    for half, (lo, hi) in HALVES.items():
        if lo <= frame <= hi:
            return half
    return None


def match_clock(frame):
    """Human readable match time for a video frame, e.g. '1H 23:14' or 'HT'."""
    half = half_of(frame)
    if half is None:
        if frame < HALVES[1][0]:
            return "pre-match"
        if frame > HALVES[2][1]:
            return "full-time"
        return "HT"
    seconds = (frame - KICKOFF[half]) / FPS
    if seconds < 0:
        return "2H pre-kickoff" if half == 2 else "1H pre-kickoff"
    if half == 2:
        seconds += 45 * 60
    return "{}H {:02d}:{:02d}".format(half, int(seconds // 60), int(seconds % 60))


def video_clock(frame):
    seconds = frame / FPS
    return "{:d}:{:02d}:{:02d}".format(int(seconds // 3600), int(seconds % 3600 // 60),
                                       int(seconds % 60))
