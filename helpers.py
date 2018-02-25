import numpy as np
import gizeh as gz
import cv2
from matplotlib import pyplot as plt


def display_img(img):
    if len(img.shape) == 2:
        cmap = 'gray'
    else:
        cmap = None
    plt.imshow(img, interpolation='bicubic', cmap=cmap)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


#def draw_lines(img, lines, color, **kwargs):
#    for a, b in lines:
#        cv2.line(img, pt2tuple(a), pt2tuple(b), color, **kwargs)


def draw_lines(surface, lines, color, width=1, **kwargs):
    gz_lines = [gz.polyline(a_to_b, stroke=color, stroke_width=width, **kwargs) for a_to_b in lines]
    gz.Group(gz_lines).draw(surface)

#def create_frame(w, h, fill=(0, 0, 0), dtype=np.uint8):
#    return np.full((h, w, 3), fill, dtype=dtype)


def create_surface(w, h, fill=(0, 0, 0)):
    return gz.Surface(w, h, bg_color=fill)


def pt2tuple(p):
    """Convert numpy array point to tuple (needed for OpenCV)"""
    return tuple(p.round().astype(np.int32))
