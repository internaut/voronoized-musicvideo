"""
General helper functions

2018, Markus Konrad <post@mkonrad.net>
"""

import numpy as np


def restrict_line(a, b, w, h):
    """
    Restrict a line from `a` to `b` to the rectangle (0, `w`), (0, `h`).
    """
    def restrict_coord(g0, g1, h0, h1, limit):
        g1 = h1 + ((limit - h0) / (g0 - h0)) * (g1 - h1)
        return limit, g1

    pts = [a, b, a]

    restricted = []
    for i in range(2):
        gx, gy = pts[i]  # this point
        hx, hy = pts[i + 1]  # other point

        if gx < 0:
            gx, gy = restrict_coord(gx, gy, hx, hy, 0)
        elif gx > w:
            gx, gy = restrict_coord(gx, gy, hx, hy, w)

        if gy < 0:
            gy, gx = restrict_coord(gy, gx, hy, hx, 0)
        elif gy > h:
            gy, gx = restrict_coord(gy, gx, hy, hx, h)

        restricted.append((gx, gy))

    # handle edge cases
    restricted = np.array(restricted)
    xs = restricted[:, 0].clip(0, w)
    ys = restricted[:, 1].clip(0, h)

    return np.vstack((xs, ys)).T.tolist()
