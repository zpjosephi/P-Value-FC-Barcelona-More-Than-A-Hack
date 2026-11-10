"""Read and write frame,label CSVs and turn label arrays into possession segments."""

import os

import numpy as np
import pandas as pd

from . import config


def read(path):
    table = pd.read_csv(path)
    labels = table["label"].astype(str).str.strip().str.upper().values.astype("<U1")
    if len(labels) != config.N_FRAMES:
        raise ValueError("{}: {} rows, expected {}".format(path, len(labels), config.N_FRAMES))
    return labels


def public_gt():
    return read(config.PUBLIC_GT)


def write(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def segments(labels, lo=0, hi=None):
    """[(start, end_inclusive, label)] maximal runs inside [lo, hi]."""
    hi = config.N_FRAMES - 1 if hi is None else hi
    out = []
    start = lo
    for i in range(lo + 1, hi + 2):
        if i > hi or labels[i] != labels[start]:
            out.append((start, i - 1, labels[start]))
            start = i
    return out
