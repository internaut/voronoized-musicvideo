"""
Functions for generating Voronoi cells from a video frame.

2018, Markus Konrad <post@mkonrad.net>
"""

import random

import numpy as np
import cv2
from scipy.spatial import Voronoi


def lines_for_voronoi(vor, img_w, img_h):
    """
    Construct voronoi region lines from SciPy Voronoi object for image with given dimensions.
    :param vor: SciPy Voronoi object
    :param img_w: image width
    :param img_h: image height
    :return: list of voronoi lines, each as end-to-end (a, b) line
    """

    # taken and adopted from https://github.com/scipy/scipy/blob/v1.0.0/scipy/spatial/_plotutils.py
    center = np.array([img_w, img_h]) / 2
    max_dim_extend = max(img_w, img_h)

    lines = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            lines.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * max_dim_extend

            lines.append([vor.vertices[i], far_point])
    return lines


def features_from_img(img, blur_radius, features_where=0):
    """
    Retrieve features from input image. These are the white or black pixels in a binarized version of the input image.
    :param img: input image
    :param blur_radius: apply blurring with this radius to reduce noise
    :param features_where: 0 or 255 for black or white pixels in binary image as features
    :return tuple (gray image, binarized image, features as coordinates into input image)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if blur_radius > 0:
        gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray, binimg, np.where(binimg == features_where)


def voronoi_from_feature_samples(features, sample_size):
    """
    From a set of features, sample them according to `sample_size` and return a SciPy Voronoi object.
    :param features: features as 2D-coordinates
    :param sample_size: sample size
    :return: SciPy Voronoi
    """

    pts = random.sample(list(zip(features[1], features[0])), sample_size)

    return Voronoi(pts)
