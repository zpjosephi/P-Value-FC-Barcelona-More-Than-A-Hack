"""Our rebuild of the official baseline on top of the cached arrays.

Nearest tracked player to the ball gets it, D when the ball is not seen, then a centred majority
vote. Same algorithm as starter/baseline.py so the numbers can be compared one to one, but
vectorised and taking seconds instead of minutes. This is the floor every later model is
measured against, nothing more.
"""

import numpy as np

from . import atd

NOBODY = -1


def nearest_codes(data):
    team, _ = atd.nearest_player(data)
    return team


def smooth(codes, window):
    """Majority vote over +/- window frames. Ties go home, like the official one."""
    n = len(codes)
    if window <= 0:
        return np.where(codes == atd.HOME, "H", np.where(codes == atd.AWAY, "A", "D")).astype("<U1")

    home_c = np.concatenate(([0], np.cumsum(codes == atd.HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == atd.AWAY)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = np.minimum(n, idx + window + 1)
    home = home_c[hi] - home_c[lo]
    away = away_c[hi] - away_c[lo]

    labels = np.full(n, "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels
