import numpy as np
import gizeh as gz


def draw_lines(ctx, ab_lines, stroke, stroke_width=1):
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
