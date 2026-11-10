"""Score a submission with the official metric code, imported from the organisers' repo.

Nothing is reimplemented here on purpose: the numbers must be the ones the board will show.
"""

import os
import sys

from . import config

sys.path.insert(0, os.path.join(config.OFFICIAL_DIR, "evaluation"))
import possession_metrics as pm  # noqa: E402

_gt_codes = {}


def gt_codes(path=None):
    """Codes of a ground-truth CSV, the public one by default. Frames labelled X are not graded."""
    path = path or config.PUBLIC_GT
    if path not in _gt_codes:
        _gt_codes[path] = pm.label_codes(pm.read_label_csv(path))
    return _gt_codes[path]


def score_file(path, gt=None):
    """Official validation, then the full evaluate() dict, or {'error': reason}.

    gt: a hand-made key from tools/make_gt.py instead of the public ground truth.
    """
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, config.N_FRAMES)
    if reason:
        return {"error": reason}
    return pm.evaluate(gt_codes(gt), pm.label_codes(table))


def table(results):
    """[(name, evaluate dict)] -> aligned text, ranked on spellF1 like the official board."""
    rows = sorted(results, key=lambda r: -r[1].get("spell_f1_mean", -1))
    head = "{:<26} {:>8} {:>8} {:>5} {:>6}   {:>5} {:>5} {:>5}".format(
        "submission", "macroF1", "spellF1", "poss", "dur_s", "F1_H", "F1_A", "F1_D")
    lines = [head, "-" * len(head)]
    for name, r in rows:
        if "error" in r:
            lines.append("{:<26} INVALID: {}".format(name, r["error"]))
            continue
        pc = r["per_class"]
        lines.append("{:<26} {:>8.4f} {:>8.4f} {:>5d} {:>6.1f}   {:>5.2f} {:>5.2f} {:>5.2f}".format(
            name, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"],
            r["mean_spell_duration_s"], pc["H"]["f1"], pc["A"]["f1"], pc["D"]["f1"]))
    any_ok = next((r for _, r in rows if "error" not in r), None)
    if any_ok:
        lines.append("ground truth: {} possessions, mean {:.1f} s (public 5 min only)".format(
            any_ok["n_gt_spells"], any_ok["gt_mean_spell_duration_s"]))
    return "\n".join(lines)
