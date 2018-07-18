"""
Functions for generating Voronoi cells from a video frame.

2018, Markus Konrad <post@mkonrad.net>
"""

import random

import numpy as np
import cv2
from scipy.spatial import Voronoi


def lines_for_voronoi(vor, img_w, img_h):
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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if blur_radius > 0:
        gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return gray, binimg, np.where(binimg == features_where)


def voronoi_from_feature_samples(features, sample_size):
    pts = random.sample(list(zip(features[1], features[0])), sample_size)
    return Voronoi(pts)
