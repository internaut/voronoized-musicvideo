import numpy as np
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


def draw_lines(img, lines, color, **kwargs):
    for a, b in lines:
        cv2.line(img, pt2tuple(a), pt2tuple(b), color, **kwargs)


def create_frame(w, h):
    return np.full((h, w, 3), (0, 0, 0), np.uint8)


def pt2tuple(p):
    """Convert numpy array point to tuple (needed for OpenCV)"""
    return tuple(p.round().astype(np.int32))
