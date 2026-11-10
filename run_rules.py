"""Run the rule model (mtah/rules.py), write the CSV and score it next to the baseline.

    python run_rules.py                          # submissions/submission.csv + public-window score
    python run_rules.py --set confirm=8 reach=2  # try other values without editing the code
    python run_rules.py --sweep confirm 3 5 8 12 # one parameter at a time, scores only

The public window is 17 possessions, so a sweep is for seeing which way a parameter pushes and
whether the score is flat around the chosen value, not for picking the top row: differences under
0.04 spellF1 are noise (EVALUATION.md), and the official docs show a 5-minute winner losing on
the full match.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mtah import atd, config, labels, rules, score  # noqa: E402


def parse_set(pairs):
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if key not in rules.DEFAULTS:
            sys.exit("unknown parameter {!r}, known: {}".format(key, ", ".join(rules.DEFAULTS)))
        out[key] = type(rules.DEFAULTS[key])(float(value))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sweep", nargs="+", metavar="KEY VALUE", help="parameter then its values")
    parser.add_argument("--name", default="submission")
    args = parser.parse_args()

    t0 = time.time()
    params = parse_set(args.set)
    data = atd.load()
    feat = None if args.sweep else rules.features(rules.clean(data, params))
    print("data + features in {:.1f}s".format(time.time() - t0))

    results = []
    if args.sweep:
        key, values = args.sweep[0], args.sweep[1:]
        if key not in rules.DEFAULTS:
            sys.exit("unknown parameter {!r}".format(key))
        for value in values:
            trial = dict(params, **parse_set(["{}={}".format(key, value)]))
            path = labels.write(os.path.join(config.SUBMISSIONS_DIR, "sweep.csv"), rules.predict(data, trial, feat))
            results.append(("{}={}".format(key, value), score.score_file(path)))
        os.remove(path)
    else:
        path = labels.write(os.path.join(config.SUBMISSIONS_DIR, args.name + ".csv"), rules.predict(data, params, feat))
        results.append((args.name, score.score_file(path)))
        print("-> {}".format(os.path.relpath(path)))

    print(score.table(results))
