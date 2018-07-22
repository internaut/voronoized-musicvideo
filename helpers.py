"""
General helper functions

2018, Markus Konrad <post@mkonrad.net>
"""

import gizeh as gz
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


def draw_lines(ctx, ab_lines, stroke, stroke_width=1):
    """
    Draw many straight lines at once using cairo.
    :param ctx: cairo drawing context
    :param ab_lines: array with (a, b) tuples where a and b are the two end points of a straight line
    :param stroke: stroke to use for drawing each line (can be RGB(A) tuple, gz.ColorGradient or gz.ImagePattern
    :param stroke_width: stroke width
    """
    ctx.set_line_width(stroke_width)

    if isinstance(stroke, gz.ColorGradient):
        stroke.set_source(ctx)
    elif isinstance(stroke, gz.ImagePattern):
        ctx.set_source(stroke.make_cairo_pattern())
    elif hasattr(stroke, '__len__'):
        if len(stroke) == 3:
            ctx.set_source_rgb(*stroke)
        elif len(stroke) == 4:
            ctx.set_source_rgba(*stroke)

    for a, b in ab_lines:
        ctx.move_to(*a)
        ctx.line_to(*b)
        ctx.stroke()


def nparray_from_surface(surface, transparent=False, y_origin="top"):
    """
    get_npimage() COPIED AND ADAPTED FROM GIZEH PACKAGE

    Returns a WxHx[3-4] numpy array representing the RGB picture.

    If `transparent` is True the image is WxHx4 and represents a RGBA picture,
    i.e. array[i,j] is the [r,g,b,a] value of the pixel at position [i,j].
    If `transparent` is false, a RGB array is returned.

    Parameter y_origin ("top" or "bottom") decides whether point (0,0) lies in
    the top-left or bottom-left corner of the screen.
    """

    im = 0 + np.frombuffer(surface.get_data(), np.uint8)
    im.shape = (surface.get_height(), surface.get_width(), 4)
    im = im[:, :, [2, 1, 0, 3]]  # put RGB back in order
    if y_origin == "bottom":
        im = im[::-1]
    return im if transparent else im[:, :, :3]